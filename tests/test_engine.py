"""Real synthesis with the pinned model, from source.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The model (336 MB) downloads into $SUBTITLD_MODELS_DIR, or
~/.cache/subtitld/models, on the first run, checked against pins.json.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sopro_addon import __main__ as addon, audio, download, dsp, engine, voices  # noqa: E402

pytestmark = pytest.mark.model


@pytest.fixture(scope='module')
def model():
    download.ensure(addon.pins(), addon._pins['repo'], lambda *_: None, lambda: False)
    return engine.Sopro(addon.model_dir(), threads=engine.default_threads())


def _preset_reference(model, index=0):
    samples, rate = audio.read_wav(voices.clip_path(voices.presets()[index]))
    return model.prepare_reference(dsp.resample(samples, rate, engine.SAMPLE_RATE))


def test_every_preset_prepares(model):
    for index, voice in enumerate(voices.presets()):
        ref = _preset_reference(model, index)
        assert ref.cond_vec.shape == (1, 512), voice.id
        assert len(ref.semantic_tokens) > 50 and ref.mel.shape[0] == 100, voice.id
        assert np.isfinite(ref.mel).all() and np.isfinite(ref.cond_vec).all(), voice.id


@pytest.mark.parametrize('language,text', [
    ('en', 'We have to leave before sunrise.'),
    ('pt', 'Temos de sair antes do nascer do sol.'),
    ('fr', 'Nous devons partir avant le lever du soleil.'),
    ('de', 'Wir müssen vor Sonnenaufgang aufbrechen.'),
])
def test_speaks_each_language(model, language, text):
    ref = _preset_reference(model)
    wav = model.synthesize(text, ref, language=language, seed=3)
    seconds = len(wav) / engine.SAMPLE_RATE
    assert 0.8 < seconds < 8, seconds
    assert wav.dtype == np.float32 and np.isfinite(wav).all()
    assert 0.01 < float(np.sqrt(np.mean(wav ** 2))) < 0.5
    assert float(np.max(np.abs(wav))) <= 1.0


def test_a_seed_gives_the_same_line(model):
    ref = _preset_reference(model)
    a = model.synthesize('Same seed, same line.', ref, language='en', seed=11)
    b = model.synthesize('Same seed, same line.', ref, language='en', seed=11)
    np.testing.assert_array_equal(a, b)


def test_long_text_is_split_and_joined(model):
    ref = _preset_reference(model)
    text = ('This is a long line, long enough that Sopro splits it into segments before speaking. ' * 5).strip()
    wav = model.synthesize(text, ref, language='en', seed=2)
    assert len(wav) / engine.SAMPLE_RATE > 15


def test_cancel_stops_between_steps(model):
    ref = _preset_reference(model)
    calls = []
    flag = threading.Event()

    def progress(value):
        calls.append(value)
        flag.set()

    with pytest.raises(engine.Cancelled):
        model.synthesize('A line that is cancelled as soon as it starts to be spoken.', ref,
                         language='en', progress=progress, cancelled=flag.is_set)
    assert calls


def test_threads_option_is_used(model):
    assert engine.default_threads() >= 1
    assert os.cpu_count() is None or engine.default_threads() <= max(1, os.cpu_count())
