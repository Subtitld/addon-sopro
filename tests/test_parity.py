"""The Python port gives the numbers upstream's JavaScript runtime gives.

tests/fixtures/js_parity.json comes from upstream's web/src/tokenizer.js and
dsp.js, run by tools/js_fixtures.mjs on the same inputs as here. The ONNX
graphs were exported and validated with that runtime, so a mismatch in the
features they take, or in the iSTFT of what they give, would change the voice.
"""

import base64
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sopro_addon import dsp
from sopro_addon.engine import PROFILE
from sopro_addon.text import Tokenizer, normalize_text, split_text

FIXTURES = json.loads((Path(__file__).parent / 'fixtures' / 'js_parity.json').read_text(encoding='utf-8'))
TOKENIZER = Path(__file__).parent / 'fixtures' / 'tokenizer.json'


def f32(value: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(value), dtype=np.float32)


def signal(rate: int, seconds: float) -> np.ndarray:
    """tools/js_fixtures.mjs's test signal."""
    state = 12345
    n = round(rate * seconds)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        noise = state / 4294967296 - 0.5
        t = i / rate
        gate = 0.02 if math.floor(t * 2.5) % 3 == 2 else 1
        out[i] = gate * (0.3 * math.sin(2 * math.pi * (140 + 20 * math.sin(2 * math.pi * 3 * t)) * t)
                         + 0.1 * math.sin(2 * math.pi * 1100 * t)) + 0.01 * noise
    return out


@pytest.fixture(scope='module')
def wav24():
    return signal(24000, 4.3)


def test_text_normalization():
    for case in FIXTURES['tokens']:
        assert normalize_text(case['text'], case['language']) == case['normalized']


def test_tokenizer_ids():
    if not TOKENIZER.is_file():
        pytest.fail('tests/fixtures/tokenizer.json is missing')
    tokenizer = Tokenizer.from_file(TOKENIZER)
    for case in FIXTURES['tokens']:
        assert tokenizer.encode(case['text'], case['language']) == case['ids'], case['text']


def test_split_text():
    assert split_text(FIXTURES['split']['text'], 120) == FIXTURES['split']['segments']


def test_resample(wav24):
    expected = f32(FIXTURES['resample'])
    got = dsp.resample(wav24[:4800], 24000, 16000)
    np.testing.assert_allclose(got, expected, atol=2e-6)


def test_reference_features(wav24):
    f = FIXTURES['features']
    got = dsp.reference_features(wav24)
    assert got['n_tokens'] == f['n_tokens']
    assert got['interp_left'].tolist() == f['left']
    assert got['interp_right'].tolist() == f['right']
    np.testing.assert_allclose(got['interp_weight'], f32(f['weight']), atol=1e-6)
    for name in ('semantic', 'speaker', 'acoustic'):
        expected = f32(f[name]).reshape(-1, f[f'{name}_frames'])
        assert got[name].shape == expected.shape, name
        # Log spectra: compare where the signal is, not deep in the floor.
        np.testing.assert_allclose(got[name], expected, atol=2e-3, err_msg=name)


def test_crop_and_normalize_long():
    c = FIXTURES['crop']
    wav, level = dsp.crop_and_normalize_reference(signal(24000, 13.7), 24000, 10)
    assert len(wav) == c['length']
    assert level == pytest.approx(c['level_db'], abs=1e-4)
    np.testing.assert_allclose(wav[:2000], f32(c['head']), atol=1e-5)
    np.testing.assert_allclose(wav[-2000:], f32(c['tail']), atol=1e-5)


def test_crop_and_normalize_short():
    c = FIXTURES['crop_short']
    wav, level = dsp.crop_and_normalize_reference(signal(24000, 3.1), 24000, 10)
    assert len(wav) == c['length']
    assert level == pytest.approx(c['level_db'], abs=1e-4)
    np.testing.assert_allclose(wav[-2000:], f32(c['tail']), atol=1e-5)


def test_istft():
    features = f32(FIXTURES['istft']['input']).reshape(9, 1026)
    np.testing.assert_allclose(dsp.istft(features), f32(FIXTURES['istft']['output']), atol=1e-4)


def test_trims_and_limit():
    t = FIXTURES['trims']
    s = signal(24000, 2.2)
    g = np.where(np.arange(len(s)) < 9000, s * np.float32(0.001), s).astype(np.float32)
    assert len(dsp.trim_lead(g, 24000)) == t['lead']
    assert len(dsp.trim_lead(g, 24000, 0.3, 0.1)) == t['lead_seg']
    assert len(dsp.trim_trail(g, 24000)) == t['trail']
    np.testing.assert_allclose(dsp.soft_limit(np.array([0.5, 0.95, -1.4, 2.0], dtype=np.float32)),
                               f32(t['limit']), atol=1e-6)


def test_profile_is_the_cpu_one():
    assert PROFILE == 'wasm-uint8'
