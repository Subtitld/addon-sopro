"""Sopro's text front end: normalization, segment splitting and tokenizer.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

A port of upstream's web/src/tokenizer.js (Apache-2.0, Samuel Vitorino),
which the ONNX export ships with: tokenizer.json is the SentencePiece unigram
model as plain pieces, scores and types, so no sentencepiece library is
needed. Encoding is the same Viterbi search over a trie, with byte fallback.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path

LANGUAGE_TAGS = {'en': '<|lang_en|>', 'pt': '<|lang_pt|>', 'fr': '<|lang_fr|>', 'de': '<|lang_de|>'}
_TERMINALS = ('.', '!', '?', '-', ',', ';', ':')
_SENTENCE_END = re.compile(r'(?<=[.!?…])\s+')
_CLAUSE_END = re.compile(r'(?<=[,;:])\s+')


def language_tag(language: str | None) -> str | None:
    code = str(language or '').strip().lower()
    if not code or code == 'auto':
        return None
    # Subtitld sends BCP-47 codes; pt-BR, en-GB and so on use the base tag.
    code = code.replace('_', '-').split('-', 1)[0]
    tag = LANGUAGE_TAGS.get(code)
    if tag is None:
        raise ValueError(f'Unsupported language: {language}')
    return tag


def normalize_text(value: str, language: str | None = None) -> str:
    text = str(value or '').strip()
    tag = language_tag(language)
    if not text and tag:
        return tag
    if not text:
        text = 'You need to add some text for me to talk.'
    if text[0] == text[0].lower() and text[0] != text[0].upper():
        text = text[0].upper() + text[1:]
    text = text.replace('…', '...')
    text = re.sub('[“”]', '"', text)
    text = re.sub('[‘’]', "'", text)
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not text.endswith(_TERMINALS):
        text += '.'
    if tag:
        text = f'{tag} {text}'
    return text


def _pack(parts: list[str], max_chars: int) -> list[str]:
    out: list[str] = []
    current = ''
    for part in parts:
        if not current:
            current = part
        elif len(current) + 1 + len(part) <= max_chars:
            current = f'{current} {part}'
        else:
            out.append(current)
            current = part
    if current:
        out.append(current)
    return out


def split_text(value: str, max_chars: int) -> list[str]:
    """Sentences, else clauses, else words, packed up to max_chars each."""
    text = ' '.join(str(value).split())
    if len(text) <= max_chars:
        return [text] if text else []
    segments: list[str] = []
    for sentence in _pack(_SENTENCE_END.split(text), max_chars):
        if len(sentence) <= max_chars:
            segments.append(sentence)
            continue
        for clause in _pack(_CLAUSE_END.split(sentence), max_chars):
            if len(clause) <= max_chars:
                segments.append(clause)
            else:
                segments.extend(_pack(clause.split(' '), max_chars))
    return segments


class Tokenizer:
    def __init__(self, spec: dict):
        if spec.get('format') != 'sentencepiece-unigram-v1':
            raise ValueError(f"Unsupported tokenizer format: {spec.get('format')}")
        self.bos_id = int(spec['bosId'])
        self.eos_id = int(spec['eosId'])
        self.unk_id = int(spec['unkId'])
        self.max_length = int(spec['maxLength'])
        # A trie of dicts; the key '' holds (id, score) where a piece ends.
        self.root: dict = {}
        self.byte_ids: dict[int, int] = {}
        for piece_id, (piece, kind) in enumerate(zip(spec['pieces'], spec['types'])):
            if kind == 'byte':
                self.byte_ids[int(piece[3:5], 16)] = piece_id
                continue
            if kind != 'normal':
                continue
            node = self.root
            for char in piece:
                node = node.setdefault(char, {})
            node[''] = (piece_id, float(spec['scores'][piece_id]))

    @classmethod
    def from_file(cls, path: Path) -> 'Tokenizer':
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def encode(self, value: str, language: str | None = None) -> list[int]:
        normalized = unicodedata.normalize('NFKC', normalize_text(value, language))
        chars = '▁' + re.sub(r'\s+', '▁', normalized)
        n = len(chars)
        best = [-math.inf] * (n + 1)
        back: list[tuple[int, list[int]] | None] = [None] * (n + 1)
        best[0] = 0.0
        for start in range(n):
            if best[start] == -math.inf:
                continue
            node = self.root
            matched = False
            for end in range(start, n):
                node = node.get(chars[end])
                if node is None:
                    break
                hit = node.get('')
                if hit is not None:
                    matched = True
                    score = best[start] + hit[1]
                    if score > best[end + 1]:
                        best[end + 1] = score
                        back[end + 1] = (start, [hit[0]])
            if not matched or best[start + 1] == -math.inf:
                ids = [self.byte_ids.get(b, self.unk_id) for b in chars[start].encode('utf-8')]
                score = best[start] - 100 - len(ids)
                if score > best[start + 1]:
                    best[start + 1] = score
                    back[start + 1] = (start, ids)
        pieces: list[int] = []
        cursor = n
        while cursor > 0:
            item = back[cursor]
            if item is None:
                pieces.insert(0, self.unk_id)
                cursor -= 1
                continue
            pieces[0:0] = item[1]
            cursor = item[0]
        return [self.bos_id, *pieces, self.eos_id][:self.max_length]
