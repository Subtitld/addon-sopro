"""Sopro's signal processing: reference features, levels, trims and the iSTFT.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

A numpy port of upstream's web/src/dsp.js (Apache-2.0, Samuel Vitorino), the
code the ONNX export was validated with. The graphs take mel spectrograms in,
and the vocoder gives iSTFT features out, so these must match it: the same
windows, mel banks, reflect padding and crops. Only the offline path is here.
"""

from __future__ import annotations

import math

import numpy as np

PROMPT_LEVEL_DB = -19.8
OUTPUT_LEVEL_DB = -23.0
REF_GAIN_LIMIT_DB = 30.0
PAUSE_MIN_SECONDS = 0.10
PAUSE_KEEP_SECONDS = 0.15
CROP_FORWARD_SECONDS = 5.0
CROP_BACKWARD_SECONDS = 5.0
MIN_KEEP_FRACTION = 0.75
ROOM_TONE_SECONDS = 0.25
FADE_SECONDS = 0.02
ONSET_THRESHOLD_DB = -45.0
ONSET_OVER_FLOOR_DB = 15.0
ONSET_WINDOW_FRAMES = 6
ONSET_MIN_FRAMES = 5
LEAD_IN_SECONDS = 0.08
SEGMENT_LEAD_SECONDS = 0.30
SEGMENT_SKIP_SECONDS = 0.10
TRAIL_SECONDS = 0.30
JOIN_FADE_SECONDS = 0.01
FINAL_FADE_SECONDS = 0.08


class Random:
    """xorshift32 with Box-Muller, as upstream's, so seeds give the same noise."""

    def __init__(self, seed: int = 1):
        self.state = (int(seed) & 0xFFFFFFFF) or 1
        self.spare: float | None = None

    def value(self) -> float:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x
        return (x + 0.5) / 4294967296

    def normal(self) -> float:
        if self.spare is not None:
            value, self.spare = self.spare, None
            return value
        r = math.sqrt(-2 * math.log(max(1e-12, self.value())))
        a = 2 * math.pi * self.value()
        self.spare = r * math.sin(a)
        return r * math.cos(a)

    def normals(self, count: int) -> np.ndarray:
        out = np.empty(count, dtype=np.float32)
        for i in range(count):
            out[i] = self.normal()
        return out


# ---------------------------------------------------------------------------
# Spectrograms
# ---------------------------------------------------------------------------
def _reflect(index: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        return np.zeros_like(index)
    period = 2 * length - 2
    index = np.abs(index) % period
    return np.where(index >= length, period - index, index)


def _hz_to_mel(hz, scale):
    hz = np.asarray(hz, dtype=np.float64)
    if scale == 'htk':
        return 2595 * np.log10(1 + hz / 700)
    step = math.log(6.4) / 27
    return np.where(hz < 1000, 3 * hz / 200, 15 + np.log(np.maximum(hz, 1e-12) / 1000) / step)


def _mel_to_hz(mel, scale):
    mel = np.asarray(mel, dtype=np.float64)
    if scale == 'htk':
        return 700 * (10 ** (mel / 2595) - 1)
    step = math.log(6.4) / 27
    return np.where(mel < 15, 200 * mel / 3, 1000 * np.exp(step * (mel - 15)))


def _mel_bank(sample_rate, n_fft, n_mels, f_min, f_max, scale, norm) -> np.ndarray:
    bins = n_fft // 2 + 1
    lo, hi = _hz_to_mel(f_min, scale), _hz_to_mel(f_max, scale)
    points = _mel_to_hz(lo + (hi - lo) * np.arange(n_mels + 2) / (n_mels + 1), scale)
    hz = sample_rate * np.arange(bins) / n_fft
    left, center, right = points[:-2, None], points[1:-1, None], points[2:, None]
    bank = np.maximum(0, np.minimum((hz - left) / (center - left), (right - hz) / (right - center)))
    if norm == 'slaney':
        bank *= 2 / (right - left)
    return bank


def mel_spectrogram(wav: np.ndarray, *, sample_rate: int, n_fft: int, hop_length: int, n_mels: int,
                    win_length: int | None = None, f_min: float = 0.0, f_max: float | None = None,
                    power: int = 2, scale: str = 'htk', norm: str | None = None,
                    frames: int | None = None) -> np.ndarray:
    """(n_mels, frames), centred frames with reflect padding, a periodic Hann window."""
    win_length = win_length or n_fft
    f_max = sample_rate / 2 if f_max is None else f_max
    count = frames if frames is not None else len(wav) // hop_length + 1
    window = np.zeros(n_fft)
    start = (n_fft - win_length) // 2
    window[start:start + win_length] = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win_length) / win_length)
    bank = _mel_bank(sample_rate, n_fft, n_mels, f_min, f_max, scale, norm)
    index = np.arange(count)[:, None] * hop_length - n_fft // 2 + np.arange(n_fft)[None, :]
    framed = np.asarray(wav, dtype=np.float64)[_reflect(index, len(wav))] * window
    spectrum = np.abs(np.fft.rfft(framed, axis=1)) ** 2
    if power == 1:
        spectrum = np.sqrt(spectrum)
    return (spectrum @ bank.T).T.astype(np.float32)


def resample(wav: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Hann-windowed sinc with 16 taps a side, normalized by the weight sum."""
    wav = np.asarray(wav, dtype=np.float32)
    if source_rate == target_rate:
        return wav.copy()
    length = max(1, round(len(wav) * target_rate / source_rate))
    radius, ratio = 16, source_rate / target_rate
    out = np.empty(length, dtype=np.float32)
    taps = np.arange(-radius + 1, radius + 1)
    block = 8192
    for first in range(0, length, block):
        center = (np.arange(first, min(length, first + block)) + 0.5) * ratio - 0.5
        source = np.floor(center).astype(np.int64)[:, None] + taps[None, :]
        x = center[:, None] - source
        with np.errstate(invalid='ignore', divide='ignore'):
            sinc = np.where(np.abs(x) < 1e-8, 1.0, np.sin(np.pi * x) / (np.pi * x))
        weight = sinc * (0.5 + 0.5 * np.cos(np.pi * x / radius))
        valid = (source >= 0) & (source < len(wav))
        weight = np.where(valid, weight, 0.0)
        values = wav[np.clip(source, 0, len(wav) - 1)]
        total = weight.sum(axis=1)
        out[first:first + len(center)] = np.where(total != 0, (values * weight).sum(axis=1) / np.where(total != 0, total, 1), 0)
    return out


# ---------------------------------------------------------------------------
# Levels and crops
# ---------------------------------------------------------------------------
def _frame_rms(wav: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, int]:
    win, hop = int(sample_rate * 0.025), int(sample_rate * 0.01)
    if len(wav) < win:
        return np.zeros(0), win, hop
    starts = np.arange(0, len(wav) - win + 1, hop)
    cumsum = np.concatenate(([0.0], np.cumsum(np.asarray(wav, dtype=np.float64) ** 2)))
    values = np.maximum(1e-6, np.sqrt(np.maximum(0, cumsum[starts + win] - cumsum[starts]) / win))
    return values, win, hop


def _quantile(values: np.ndarray, q: float) -> float:
    a = np.sort(np.asarray(values, dtype=np.float64))
    if not len(a):
        return 1e-6
    position = q * (len(a) - 1)
    low = int(math.floor(position))
    weight = position - low
    return float(a[low] * (1 - weight) + a[low + 1] * weight) if low + 1 < len(a) else float(a[low])


def _median(values: np.ndarray) -> float:
    a = np.sort(np.asarray(values, dtype=np.float64))
    return float(a[(len(a) - 1) // 2]) if len(a) else 1e-6


def _pause_runs(quiet: np.ndarray, min_run: int) -> list[tuple[int, int]]:
    runs, i = [], 0
    while i < len(quiet):
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < len(quiet) and quiet[j]:
            j += 1
        if j - i >= min_run:
            runs.append((i, j))
        i = j
    return runs


def _finish_with_room_tone(wav: np.ndarray, sample_rate: int, floor: float) -> np.ndarray:
    fade = int(FADE_SECONDS * sample_rate)
    out = np.zeros(len(wav) + int(ROOM_TONE_SECONDS * sample_rate), dtype=np.float32)
    out[:len(wav)] = wav
    if fade > 1 and len(wav) >= fade:
        out[len(wav) - fade:len(wav)] *= 1 - np.arange(fade) / (fade - 1)
    out[len(wav):] = Random(1).normals(len(out) - len(wav)) * floor
    return out


def crop_on_pause(wav: np.ndarray, seconds: float, sample_rate: int) -> np.ndarray:
    """Crop near `seconds` at a pause, or pad a short clip with room tone."""
    win, hop, total = int(sample_rate * 0.025), int(sample_rate * 0.010), len(wav)
    if total < 4 * win:
        return wav
    values, _, _ = _frame_rms(wav, sample_rate)
    floor = _quantile(values, 0.1)
    runs = _pause_runs(values < floor * 4, max(1, round(PAUSE_MIN_SECONDS / 0.010)))
    keep, target = round(PAUSE_KEEP_SECONDS / 0.010), round(seconds * sample_rate)

    def cut_at(run):
        a, b = run
        return wav[:min(total, (a + min(b - a, keep)) * hop + win)]

    if total <= target:
        if runs and runs[-1][1] * hop + win >= total - hop:
            return wav
        inside = [r for r in runs if r[0] * hop >= total * MIN_KEEP_FRACTION]
        return cut_at(inside[-1]) if inside else _finish_with_room_tone(wav, sample_rate, floor)
    forward = [r for r in runs if target <= r[0] * hop <= target + CROP_FORWARD_SECONDS * sample_rate]
    if forward:
        return cut_at(forward[0])
    backward = [r for r in runs if r[0] * hop < target and r[1] * hop >= target - CROP_BACKWARD_SECONDS * sample_rate]
    if backward:
        return cut_at(backward[-1])
    return _finish_with_room_tone(wav[:target], sample_rate, floor)


def speech_level_db(wav: np.ndarray, sample_rate: int) -> tuple[float, float]:
    win, hop = int(sample_rate * 0.025), int(sample_rate * 0.010)
    if len(wav) < win:
        rms = math.sqrt(float(np.sum(np.asarray(wav, dtype=np.float64) ** 2)) / max(1, len(wav)))
        return 20 * math.log10(max(rms, 1e-6)), 0.0
    values, _, _ = _frame_rms(wav, sample_rate)
    threshold = _quantile(values, 0.2) * 1.5
    active = values[values > threshold]
    kept = active if len(active) else values
    return 20 * math.log10(_median(kept)), len(kept) * hop / sample_rate


def normalize_reference(wav: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    level_db, _ = speech_level_db(wav, sample_rate)
    gain_db = min(max(PROMPT_LEVEL_DB - level_db, 0), REF_GAIN_LIMIT_DB)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak > 0:
        gain_db = min(gain_db, max(0, 20 * math.log10(0.95 / peak)))
    return (wav * 10 ** (gain_db / 20)).astype(np.float32), level_db + gain_db


def crop_and_normalize_reference(wav: np.ndarray, sample_rate: int, seconds: float = 10) -> tuple[np.ndarray, float]:
    return normalize_reference(crop_on_pause(np.asarray(wav, dtype=np.float32), seconds, sample_rate), sample_rate)


def output_gain(prompt_level_db: float = PROMPT_LEVEL_DB) -> float:
    return 10 ** ((OUTPUT_LEVEL_DB - prompt_level_db) / 20)


def _short_rms(wav: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    win = int(sample_rate * 0.010)
    frames = len(wav) // win
    if not frames:
        return np.zeros(0), win
    blocks = np.asarray(wav[:frames * win], dtype=np.float64).reshape(frames, win)
    return np.sqrt(np.mean(blocks ** 2, axis=1)), win


def _onset_threshold(values: np.ndarray) -> float:
    threshold = 10 ** (ONSET_THRESHOLD_DB / 20)
    if len(values) >= 30:
        threshold = max(threshold, _quantile(values, 0.1) * 10 ** (ONSET_OVER_FLOOR_DB / 20))
    return threshold


def speech_onset(wav: np.ndarray, sample_rate: int) -> int | None:
    values, win = _short_rms(wav, sample_rate)
    if len(wav) < win * ONSET_WINDOW_FRAMES:
        return None
    hits = (values > _onset_threshold(values)).astype(np.int32)
    windowed = np.convolve(hits, np.ones(ONSET_WINDOW_FRAMES, dtype=np.int32), mode='valid')
    found = np.nonzero(windowed >= ONSET_MIN_FRAMES)[0]
    return int(found[0]) * win if len(found) else None


def trim_lead(wav: np.ndarray, sample_rate: int, lead: float = LEAD_IN_SECONDS, skip: float = 0) -> np.ndarray:
    onset = speech_onset(wav, sample_rate)
    if onset is None:
        return wav
    cut = max(onset - int(lead * sample_rate), int(skip * sample_rate))
    cut = min(cut, max(0, onset - int(0.02 * sample_rate)))
    return wav[cut:]


def trim_trail(wav: np.ndarray, sample_rate: int, trail: float = TRAIL_SECONDS) -> np.ndarray:
    values, win = _short_rms(wav, sample_rate)
    if len(wav) < win:
        return wav
    loud = np.nonzero(values > _onset_threshold(values))[0]
    if not len(loud):
        return wav
    return wav[:min(len(wav), (int(loud[-1]) + 1) * win + int(trail * sample_rate))]


def fade_edges(wav: np.ndarray, sample_rate: int, fade_in: bool, fade_out: bool,
               fade_seconds: float = JOIN_FADE_SECONDS) -> np.ndarray:
    fade = int(fade_seconds * sample_rate)
    if len(wav) <= 2 * fade:
        return wav
    out = np.array(wav, dtype=np.float32)
    ramp = np.arange(fade) / (fade - 1)
    if fade_in:
        out[:fade] *= ramp
    if fade_out:
        out[len(out) - fade:] *= ramp[::-1]
    return out


def join_segments(parts: list[np.ndarray], sample_rate: int) -> np.ndarray:
    faded = [fade_edges(p, sample_rate, i > 0, i + 1 < len(parts)) for i, p in enumerate(parts)]
    return np.concatenate(faded) if faded else np.zeros(0, dtype=np.float32)


def soft_limit(samples: np.ndarray) -> np.ndarray:
    a = np.abs(samples)
    limited = np.sign(samples) * (0.9 + 0.1 * np.tanh((a - 0.9) / 0.1))
    return np.where(a > 0.9, limited, samples).astype(np.float32)


# ---------------------------------------------------------------------------
# Graph inputs and outputs
# ---------------------------------------------------------------------------
def reference_features(wav24: np.ndarray) -> dict:
    """What the reference graph takes, and the acoustic prompt's log mel."""
    wav16 = resample(wav24, 24000, 16000)
    semantic_frames = math.ceil(len(wav16) / 160)
    padded = np.concatenate([wav16, np.zeros(400, dtype=np.float32)])
    semantic = mel_spectrogram(padded, sample_rate=16000, n_fft=400, hop_length=160, n_mels=80,
                               scale='slaney', norm='slaney', frames=semantic_frames + 2)
    semantic = np.log10(np.maximum(1e-10, semantic))
    semantic = (np.maximum(semantic, semantic.max() - 8) + 4) / 4

    n_tokens, n50 = math.ceil(len(wav24) / 1024), math.ceil(semantic_frames / 2)
    source = np.clip((np.arange(n_tokens) + 0.5) * n50 / n_tokens - 0.5, 0, n50 - 1)
    left = np.floor(source).astype(np.int64)
    right = np.minimum(n50 - 1, left + 1)

    speaker = mel_spectrogram(wav16, sample_rate=16000, n_fft=1024, win_length=400, hop_length=160,
                              n_mels=80, f_min=20, f_max=7600, scale='slaney', norm='slaney')
    speaker = np.log(np.maximum(1e-5, speaker.astype(np.float64)))
    speaker = (speaker - speaker.mean(axis=0)) / np.sqrt(speaker.var(axis=0) + 1e-5)

    acoustic = mel_spectrogram(wav24, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100, power=1)
    acoustic = np.log(np.maximum(1e-7, acoustic))
    return {
        'semantic': semantic.astype(np.float32),
        'speaker': speaker.astype(np.float32),
        'acoustic': acoustic.astype(np.float32),
        'interp_left': left, 'interp_right': right,
        'interp_weight': (source - left).astype(np.float32),
        'n_tokens': n_tokens,
    }


def istft(features: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """(frames, n_fft + 2) log magnitudes then phases → samples, centre-trimmed."""
    frames = features.shape[0]
    bins = n_fft // 2 + 1
    if not frames:
        return np.zeros(0, dtype=np.float32)
    magnitude = np.exp(np.minimum(math.log(100), features[:, :bins].astype(np.float64)))
    phase = features[:, bins:].astype(np.float64)
    waves = np.fft.irfft(magnitude * np.exp(1j * phase), n=n_fft, axis=1)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    size = (frames - 1) * hop + n_fft
    ola, env = np.zeros(size), np.zeros(size)
    for t in range(frames):
        ola[t * hop:t * hop + n_fft] += waves[t] * window
        env[t * hop:t * hop + n_fft] += window * window
    pad = n_fft // 2
    return (ola[pad:size - pad] / np.maximum(1e-8, env[pad:size - pad])).astype(np.float32)
