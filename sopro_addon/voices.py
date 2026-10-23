"""The voices the add-on offers, and the languages Sopro speaks.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Sopro has no voices of its own: every line clones a reference clip.

- `sopro-clone` clones the speaker from the video: the host hands over a clip
  of their speech (or Configure names one).
- The presets clone a clip that ships with the add-on: audiobook readers from
  LibriVox, whose recordings are in the public domain, taken from LibriTTS-R
  (CC-BY-4.0) for English and from the LibriVox originals for the rest.
  presets/README.md credits each one. A preset speaks every language; its own
  language only sets its accent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CLONE = 'sopro-clone'
LANGUAGES = ('en', 'pt', 'fr', 'de')
LANGUAGE_NAMES = {'en': 'English', 'pt': 'Portuguese', 'fr': 'French', 'de': 'German'}

PRESETS_DIR = Path(__file__).resolve().parent / 'presets'


@dataclass(frozen=True)
class Voice:
    id: str
    display_name: str
    clip: str = ''   # file name in presets/, empty for the clone


@lru_cache(maxsize=1)
def presets() -> tuple[Voice, ...]:
    index = json.loads((PRESETS_DIR / 'presets.json').read_text(encoding='utf-8'))
    return tuple(Voice(p['id'], p['display_name'], p['clip']) for p in index['presets'])


def all_voices() -> tuple[Voice, ...]:
    return (Voice(CLONE, "Speaker's own voice (cloned from the video)"), *presets())


def by_id(voice_id: str) -> Voice | None:
    return next((v for v in all_voices() if v.id == voice_id), None)


def clip_path(voice: Voice) -> Path:
    return PRESETS_DIR / voice.clip


def manifest_voices() -> list[dict]:
    """The voice list as manifest.json and the hello frame carry it."""
    out = []
    for voice in all_voices():
        entry = {'id': voice.id, 'language': '*', 'display_name': voice.display_name, 'sample_rate': 24000}
        if voice.id == CLONE:
            entry['requires'] = ['voice_ref_audio']
        out.append(entry)
    return out


def primary_language(tag) -> str:
    """`pt-BR` -> `pt`; '' when there is none."""
    return str(tag or '').strip().lower().replace('_', '-').split('-', 1)[0]
