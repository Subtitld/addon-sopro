"""manifest.json, pins.json, the presets and the code agree with each other.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sopro_addon import ADDON_ID, __version__, audio, voices  # noqa: E402
from sopro_addon import __main__ as addon  # noqa: E402
from sopro_addon.engine import GRAPHS, PROFILE, graph_path  # noqa: E402

MANIFEST = json.loads((REPO_ROOT / 'manifest.json').read_text(encoding='utf-8'))
_SHA256 = re.compile(r'^[0-9a-f]{64}$')


def test_versions_agree():
    pyproject = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    assert MANIFEST['version'] == __version__ == pyproject['project']['version']
    assert re.fullmatch(r'\d\d\.\d\d\.\d+', __version__), 'versions are YY.MM.N'
    assert MANIFEST['id'] == ADDON_ID


def test_manifest_lists_what_the_code_offers():
    assert MANIFEST['languages'] == list(voices.LANGUAGES)
    assert MANIFEST['voices'] == voices.manifest_voices()
    assert MANIFEST['tasks'] == ['tts.synthesize']
    assert MANIFEST['voice_clone'] is True
    assert MANIFEST['compute'] == 'cpu'
    assert MANIFEST['models'][0]['revision'] == json.loads(
        (REPO_ROOT / 'sopro_addon' / 'pins.json').read_text())['revision']
    keys = {f['key'] for f in MANIFEST['config_schema']['fields']}
    assert keys == {'threads', 'variation', 'voice_ref_audio'}


def test_pins_cover_every_graph_and_shard():
    pins = addon.pins()
    paths = {p['path'] for p in pins}
    assert {'manifest.json', 'tokenizer.json'} <= paths
    manifest = {'profiles': {PROFILE: {'graphs': {g: {'file': f'{g}.onnx'} for g in GRAPHS}}}}
    for pin in pins:
        assert _SHA256.match(pin['sha256']) and pin['size'] > 0, pin
        assert re.fullmatch(r'[0-9a-f]{40}', pin['revision']), pin
    assert sum(p['size'] for p in pins) / 1e6 == pytest_approx(336, 2)
    assert all(p['path'].startswith(('common/', f'{PROFILE}/')) or '/' not in p['path'] for p in pins)
    assert graph_path(Path('m'), manifest['profiles'][PROFILE], 'vocoder') == Path('m') / PROFILE / 'vocoder.onnx'


def pytest_approx(value, tolerance):
    import pytest
    return pytest.approx(value, abs=tolerance)


def test_every_preset_is_a_clip_of_one_voice():
    presets = voices.presets()
    assert len(presets) == 8 and len({p.id for p in presets}) == 8
    index = json.loads((voices.PRESETS_DIR / 'presets.json').read_text(encoding='utf-8'))['presets']
    for entry, voice in zip(index, presets):
        samples, rate = audio.read_wav(voices.clip_path(voice))
        assert rate == 24000 and 9 < len(samples) / rate < 16, voice.id
        assert entry['language'] in voices.LANGUAGES
        assert entry['source'] in ('LibriTTS-R', 'LibriVox via Multilingual LibriSpeech')
    credits = (voices.PRESETS_DIR / 'README.md').read_text(encoding='utf-8')
    assert all(f'`{v.clip}`' in credits for v in presets)
    assert (voices.PRESETS_DIR / 'CC-BY-4.0.txt').is_file()
