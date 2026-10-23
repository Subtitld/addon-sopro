"""Sopro V2 Turbo on ONNX Runtime: clone a reference voice, speak a line.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The graphs
----------
Upstream's ONNX export (samuel-vitorino/sopro-v2-turbo-onnx, Apache-2.0) is
split into graphs that each take plain tensors; the loops between them live
in its JavaScript runtime (web/src/index.js). This module is a port of that
runtime's offline path, on the CPU profile ("wasm-uint8": float32 graphs
with 8-bit weights):

- reference: the clip's mel spectrograms → a speaker vector and the clip's
  semantic tokens.
- semantic_prefix + semantic_core: the autoregressive model. The prefix
  embeds the text, a style prompt and a token prompt; the core is run once on
  that and then once per token, with a KV cache, and returns the next logits.
- acoustic_condition: semantic tokens → a mel-rate conditioning.
- acoustic_offline_2: two flow steps from noise to the mel of the whole line,
  continuing the reference's mel.
- vocoder: mel → iSTFT features, which dsp.istft turns into samples.

The streaming graphs are not used: a dub is made a line at a time, and
upstream says the offline path sounds better.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from sopro_addon import dsp
from sopro_addon.text import Tokenizer, split_text

log = logging.getLogger('sopro_addon.engine')

PROFILE = 'wasm-uint8'
GRAPHS = ('reference', 'semantic_prefix', 'semantic_core', 'acoustic_condition', 'acoustic_offline_2', 'vocoder')
SAMPLE_RATE = 24000


class Cancelled(Exception):
    """The host cancelled the line."""


@dataclass
class Reference:
    cond_vec: np.ndarray          # (1, 512)
    semantic_tokens: np.ndarray   # (n,) int64
    mel: np.ndarray               # (100, frames), normalized
    level_db: float

    @property
    def mel_frames(self) -> int:
        return self.mel.shape[1]


def _rotary(length: int, dim: int, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    j = np.arange(dim)
    frequency = 1 / (10000 ** ((j % (dim // 2)) * 2 / dim))
    angle = (np.arange(length)[:, None] + offset) * frequency[None, :]
    return np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)


def _causal_bias(length: int) -> np.ndarray:
    return (np.triu(np.ones((length, length), dtype=np.float32), 1) * -1e4)[None, None]


def _sample(logits: np.ndarray, random: dsp.Random, temperature: float, top_p: float, top_k: int,
            bos: int, eos: int, allow_eos: bool) -> int:
    scores = logits.astype(np.float64).copy()
    scores[bos] = -np.inf
    if not allow_eos:
        scores[eos] = -np.inf
    if temperature <= 0:
        return int(np.argmax(scores))
    limit = min(top_k, len(scores)) if top_k > 0 else len(scores)
    candidates = np.argpartition(-scores, limit - 1)[:limit]
    candidates = candidates[np.argsort(-scores[candidates], kind='stable')]
    candidates = candidates[np.isfinite(scores[candidates])]
    probabilities = np.exp(scores[candidates] / temperature - scores[candidates[0]] / temperature)
    probabilities /= probabilities.sum()
    if top_p < 1:
        keep = int(np.searchsorted(np.cumsum(probabilities), top_p) + 1)
        candidates, probabilities = candidates[:keep], probabilities[:keep]
        probabilities /= probabilities.sum()
    draw = random.value()
    for index, p in zip(candidates, probabilities):
        draw -= p
        if draw <= 0:
            return int(index)
    return int(candidates[-1])


class Sopro:
    def __init__(self, model_dir: Path, threads: int = 0):
        import onnxruntime as ort

        self.model_dir = Path(model_dir)
        self.manifest = json.loads((self.model_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.config = self.manifest['config']
        self.profile = self.manifest['profiles'][PROFILE]
        self.tokenizer = Tokenizer.from_file(self.model_dir / self.manifest['tokenizer'])
        self.mel_mean = np.asarray(self.config['model']['acoustic_mel_mean'], dtype=np.float32)[:, None]
        self.mel_std = np.asarray(self.config['model']['acoustic_mel_std'], dtype=np.float32)[:, None]
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads > 0:
            options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.sessions = {}
        for name in GRAPHS:
            # Graphs load their weights from the shard next to them.
            path = graph_path(self.model_dir, self.profile, name)
            self.sessions[name] = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])

    # ------------------------------------------------------------------
    def prepare_reference(self, wav24: np.ndarray) -> Reference:
        seconds = self.config['generation']['ref_seconds']
        wav, level_db = dsp.crop_and_normalize_reference(wav24, SAMPLE_RATE, seconds)
        f = dsp.reference_features(wav)
        out = self.sessions['reference'].run(None, {
            'semantic_mel': f['semantic'][None],
            'interp_left': f['interp_left'],
            'interp_right': f['interp_right'],
            'interp_weight': f['interp_weight'],
            'speaker_mel': f['speaker'][None],
        })
        cond_vec, tokens = out
        mel = (f['acoustic'] - self.mel_mean) / self.mel_std
        return Reference(cond_vec.astype(np.float32).reshape(1, -1),
                         np.asarray(tokens, dtype=np.int64).reshape(-1), mel.astype(np.float32), level_db)

    # ------------------------------------------------------------------
    def synthesize(self, text: str, reference: Reference, *, language: str | None = None,
                   temperature: float | None = None, top_p: float | None = None, top_k: int | None = None,
                   seed: int = 1, progress: Callable[[float], None] | None = None,
                   cancelled: Callable[[], bool] | None = None) -> np.ndarray:
        g = self.config['generation']
        params = {
            'language': language,
            'temperature': g['temperature'] if temperature is None else float(temperature),
            'top_p': g['top_p'] if top_p is None else float(top_p),
            'top_k': g['top_k'] if top_k is None else int(top_k),
            'seed': int(seed),
        }
        segments = split_text(text, g['max_segment_chars'])
        parts: list[np.ndarray] = []
        carry = None
        for index, segment in enumerate(segments):
            def step(fraction, index=index):
                if progress:
                    progress((index + fraction) / len(segments))
            wav, tokens = self._segment(segment, reference, params, carry, step, cancelled)
            if len(tokens) and g['prompt_tokens'] > 0:
                carry = tokens[-g['prompt_tokens']:]
            if len(wav):
                parts.append(wav)
        if not parts:
            return np.zeros(self.manifest['tokenSamples'], dtype=np.float32)
        gain = dsp.output_gain(reference.level_db)
        trimmed = []
        for index, part in enumerate(parts):
            scaled = part * gain
            if index == 0:
                led = dsp.trim_lead(scaled, SAMPLE_RATE)
            else:
                led = dsp.trim_lead(scaled, SAMPLE_RATE, dsp.SEGMENT_LEAD_SECONDS, dsp.SEGMENT_SKIP_SECONDS)
            trimmed.append(dsp.trim_trail(led, SAMPLE_RATE))
        joined = dsp.soft_limit(dsp.join_segments(trimmed, SAMPLE_RATE))
        return dsp.fade_edges(joined, SAMPLE_RATE, False, True, dsp.FINAL_FADE_SECONDS)

    # ------------------------------------------------------------------
    def _semantic(self, text, reference, params, prompt_tokens, progress, cancelled) -> np.ndarray:
        model, g = self.config['model'], self.config['generation']
        vocab = model['semantic_vocab_size']
        bos, eos = vocab, vocab + 1
        head_dim = model['ar_model_dim'] // model['ar_heads']
        text_ids = np.asarray([self.tokenizer.encode(text, params['language'])], dtype=np.int64)
        style = reference.semantic_tokens[:g['style_tokens']][None]
        prompt = (reference.semantic_tokens[:g['prompt_tokens']] if prompt_tokens is None else prompt_tokens)[None]
        prefix_hidden = self.sessions['semantic_prefix'].run(['hidden'], {
            'text_ids': text_ids, 'style_tokens': style.astype(np.int64), 'prompt_tokens': prompt.astype(np.int64),
        })[0]
        length = model['style_prefix_tokens'] + text_ids.shape[1] + prompt.shape[1] + 1
        core = self.sessions['semantic_core']
        empty = np.zeros((model['ar_blocks'], 1, model['ar_kv_heads'], 0, head_dim), dtype=np.float32)
        cos, sin = _rotary(length, head_dim)
        logits, past_k, past_v = core.run(None, {
            'hidden': prefix_hidden, 'token': np.array([[bos]], dtype=np.int64),
            'past_k': empty, 'past_v': empty, 'cos': cos, 'sin': sin,
            'attention_bias': _causal_bias(length),
        })
        rate = SAMPLE_RATE / self.manifest['tokenSamples']
        max_steps = math.ceil(g['max_seconds'] * rate)
        min_steps = max(1, math.ceil(g['min_seconds'] * rate))
        # A line usually needs one token per ~15 characters' worth of speech;
        # the progress estimate only has to move, not be exact.
        expected = max(min_steps, min(max_steps, len(text) * 1.1))
        empty_hidden = np.zeros((1, 0, model['ar_model_dim']), dtype=np.float32)
        random = dsp.Random(params['seed'])
        generated: list[int] = []
        position = length
        for index in range(max_steps):
            if cancelled and cancelled():
                raise Cancelled()
            allow_eos = index + 1 >= min_steps
            token = _sample(logits[0], random, params['temperature'], params['top_p'], params['top_k'], bos, eos, allow_eos)
            if allow_eos and token == eos:
                break
            generated.append(min(vocab - 1, max(0, token)))
            if index + 1 >= max_steps:
                break
            if progress and index % 8 == 0:
                progress(min(0.95, index / expected) * 0.8)
            cos, sin = _rotary(1, head_dim, position)
            position += 1
            logits, past_k, past_v = core.run(None, {
                'hidden': empty_hidden, 'token': np.array([[token]], dtype=np.int64),
                'past_k': past_k, 'past_v': past_v, 'cos': cos, 'sin': sin,
                'attention_bias': np.zeros((1, 1, 1, position), dtype=np.float32),
            })
        return np.asarray(generated, dtype=np.int64)

    def _condition(self, tokens: np.ndarray, frames: int) -> np.ndarray:
        frame_to_token = np.minimum(len(tokens) - 1, (np.arange(frames) * len(tokens)) // frames).astype(np.int64)
        return self.sessions['acoustic_condition'].run(['mu'], {
            'semantic_tokens': tokens[None].astype(np.int64), 'frame_to_token': frame_to_token,
        })[0]

    def _vocode(self, mel: np.ndarray) -> np.ndarray:
        v = self.config['vocoder']
        lookahead = int(v['block_lookaheads'][0])
        session = self.sessions['vocoder']
        state = {
            'embed_state': np.zeros((1, v['n_mels'], 6), dtype=np.float32),
            'conv_state': np.zeros((v['num_layers'], 1, v['dim'], 6), dtype=np.float32),
            'pending': np.zeros((v['num_layers'], 1, v['dim'], lookahead), dtype=np.float32),
        }
        first, embed, conv, pending = session.run(None, {
            'mel': mel[None].astype(np.float32), 'is_start': np.array(True), 'is_flush': np.array(False), **state,
        })
        tail = session.run(['istft_features'], {
            'mel': np.zeros((1, v['n_mels'], 0), dtype=np.float32), 'is_start': np.array(False),
            'is_flush': np.array(True), 'embed_state': embed, 'conv_state': conv, 'pending': pending,
        })[0]
        return dsp.istft(np.concatenate([first[0], tail[0]], axis=0))

    def _segment(self, text, reference, params, prompt_tokens, progress, cancelled):
        hop_ratio = self.manifest['hopRatio']
        generated = self._semantic(text, reference, params, prompt_tokens, progress, cancelled)
        if not len(generated):
            return np.zeros(0, dtype=np.float32), generated
        if cancelled and cancelled():
            raise Cancelled()
        p_mel = reference.mel_frames
        all_tokens = np.concatenate([reference.semantic_tokens, generated])
        frames = p_mel + len(generated) * hop_ratio
        mu = self._condition(all_tokens, frames)
        x0 = dsp.Random((params['seed'] ^ 0xa53c9e1d) & 0xFFFFFFFF).normals(100 * frames).reshape(1, 100, frames)
        cond_mel = np.zeros((1, 100, frames), dtype=np.float32)
        cond_mel[0, :, :p_mel] = reference.mel
        cond_mask = np.zeros((1, 1, frames), dtype=np.float32)
        cond_mask[..., :p_mel] = 1
        cos, sin = _rotary(frames, self.config['model']['acoustic_dit_dim_head'])
        if progress:
            progress(0.85)
        normalized = self.sessions['acoustic_offline_2'].run(['mel'], {
            'x_init': x0, 'mu': mu, 'cond_vec': reference.cond_vec, 'cond_mel': cond_mel,
            'cond_mask': cond_mask, 'cos': cos, 'sin': sin,
        })[0][0]
        if cancelled and cancelled():
            raise Cancelled()
        context = min(32, p_mel)
        mel = normalized[:, p_mel - context:] * self.mel_std + self.mel_mean
        if progress:
            progress(0.95)
        waveform = self._vocode(mel)
        start = context * self.config['vocoder']['hop_length']
        target = len(generated) * self.manifest['tokenSamples']
        return waveform[start:start + target], generated


def graph_path(model_dir: Path, profile: dict, name: str) -> Path:
    """Where a graph is on disk: the export's own layout, profile/ or common/."""
    graph = profile['graphs'][name]
    relative = (graph.get('url') or graph['file']).replace('../', '')
    if '/' not in relative:
        relative = f'{PROFILE}/{relative}'
    return Path(model_dir) / relative


def model_files(manifest: dict) -> list[dict]:
    """Each file the CPU profile reads, with its size and hash where pinned."""
    profile = manifest['profiles'][PROFILE]
    files = []
    for name in GRAPHS:
        graph = profile['graphs'][name]
        files.append({'path': graph_path(Path('.'), profile, name).as_posix(), 'bytes': graph['bytes'], 'sha256': graph['sha256']})
    for shard in profile['shards'].values():
        relative = (shard.get('url') or shard['file']).replace('../', '')
        if '/' not in relative:
            relative = f'{PROFILE}/{relative}'
        if not any(f['path'] == relative for f in files):
            files.append({'path': relative, 'bytes': shard['bytes'], 'sha256': shard['sha256']})
    return files


def default_threads() -> int:
    return min(8, max(1, (os.cpu_count() or 4) // 2))
