# Sopro add-on for Subtitld

Offline **voice-cloning text-to-speech** for
[Subtitld](https://github.com/Subtitld/subtitld), powered by
[Sopro V2 Turbo](https://github.com/samuel-vitorino/sopro), a 120M-parameter
model by Samuel Vitorino, run on [ONNX Runtime](https://onnxruntime.ai).

- Serves the `tts.synthesize` task over Subtitld's add-on protocol.
- **Clones a voice from a few seconds of speech.** Subtitld sends a clip of
  the speaker from the video, and the dub speaks in their voice. Preset voices
  are there for lines with no clip, or to pick a different voice.
- **Four languages.** English, European Portuguese, French and German, at
  24 kHz. Every voice speaks every language.
- **CPU only, and fast.** About twice as fast as real time on a laptop CPU,
  with about 1 GB of memory. No GPU and no torch.
- **Offline after the first line.** The model, 336 MB, downloads on the first
  line, then nothing leaves the machine: no telemetry, no account.

## Voices

| Voice | What it clones |
|-------|----------------|
| Speaker's own voice | The clip of the speaker Subtitld sends with each line, or the fallback clip in Configure |
| English woman, English man | A LibriVox reader of a English book |
| Portuguese woman, Portuguese man | A LibriVox reader of a Portuguese book |
| French woman, French man | A LibriVox reader of a French book |
| German woman, German man | A LibriVox reader of a German book |

A preset speaks every language; its own language only gives it an accent, so
pick one from the project's language for the most natural dub. The presets are
LibriVox audiobook readers, whose recordings are in the public domain; they
are credited in [sopro_addon/presets/README.md](sopro_addon/presets/README.md).

A good clip to clone is 5 to 20 seconds of one person speaking, with no music
or other voices. Sopro uses the first 10 seconds or so, cut at a pause.

## Options

Set these in **Settings ▸ Add-ons ▸ Sopro ▸ Configure**.

| Option | Default | Env var | Notes |
|--------|---------|---------|-------|
| CPU threads | automatic | `SOPRO_THREADS` | 0 picks half the logical processors, at most 8. |
| Variation | 0.8 | `SOPRO_VARIATION` | The sampling temperature: lower is steadier, higher livelier. |
| Fallback reference clip | none | `SOPRO_VOICE_REF_AUDIO` | Cloned when Subtitld has no clip of the speaker to send. |

Each line is spoken with the seed Subtitld sends, so the same line in the same
voice comes out the same every time. Subtitld stretches a dub to fit its
subtitle itself, so the add-on ignores the rate.

## How it runs

Upstream ships an ONNX export of the model
([samuel-vitorino/sopro-v2-turbo-onnx](https://huggingface.co/samuel-vitorino/sopro-v2-turbo-onnx))
for its in-browser demo, split into graphs, with the loop between them in
JavaScript. The add-on is a Python port of that runtime's offline path:

- `sopro_addon/text.py`: text normalization, segment splitting and the
  SentencePiece tokenizer (`web/src/tokenizer.js`).
- `sopro_addon/dsp.py`: the reference's mel spectrograms, crops and levels,
  and the iSTFT after the vocoder (`web/src/dsp.js`).
- `sopro_addon/engine.py`: the graphs and the generation loop
  (`web/src/index.js`), on the CPU profile (float32 graphs, 8-bit weights).

`tests/test_parity.py` checks the port against upstream's JavaScript, run on
the same inputs by `tools/js_fixtures.mjs`. Against the official PyTorch
package on 12 lines (4 languages, 2 cloned voices, Whisper and Resemblyzer
scoring), the add-on had a word error rate of 0.028 against 0.036 and a
speaker similarity of 0.825 against 0.839, at the same speed.

## What is bundled, what is downloaded

| What | Size | Licence | Source |
|------|------|---------|--------|
| Add-on, ONNX Runtime, numpy (bundled) | see the release | Apache-2.0; MIT; BSD-3-Clause | this repository, PyPI |
| Preset voice clips (bundled) | about 4 MB | public domain; CC-BY-4.0 | LibriVox, LibriTTS-R |
| Model (downloaded) | 336 MB | Apache-2.0 | [samuel-vitorino/sopro-v2-turbo-onnx](https://huggingface.co/samuel-vitorino/sopro-v2-turbo-onnx) |

The model goes to Subtitld's model folder (`$SUBTITLD_MODELS_DIR/sopro`), at
the revision in `sopro_addon/pins.json`, each file checked by its sha256. An
interrupted download resumes. `$HF_ENDPOINT` points it at a mirror.

Each release archive carries `THIRD_PARTY_LICENSES/README.md`, which names
the notice for every native file in it; the build fails if one has none. The
Linux archive carries libquadmath (LGPL-2.1), from numpy's wheel; its source
is attached to the same release.

Upstream asks that Sopro not be used to impersonate people. Clone a voice
only with its owner's consent.

## Limits

- **Four languages.** A project in any other language is refused with a
  message that says so. Portuguese is European Portuguese; Brazilian text is
  read, with a European accent.
- **Numbers and symbols.** Sopro's text front end is minimal: "1 + 2" may be
  misread, "one plus two" is not. Words from another language in a sentence
  may be mispronounced.
- **Platforms.** Releases ship Linux x86_64, macOS arm64 and Windows x86_64.
- **One line at a time.** A cancel stops the line between model steps, or
  stops the download, which resumes on the next line.

## Build

```bash
pip install -c constraints.txt -e '.[build,dev]'
pytest
pyinstaller pyinstaller.spec --noconfirm
dist/sopro-addon/sopro-addon --self-test
SOPRO_TEST_BINARY=$PWD/dist/sopro-addon/sopro-addon pytest tests/test_installed.py
```

To move to another revision of the model, run `tools/pin_model.py REVISION`,
then `node tools/js_fixtures.mjs <sopro checkout> <tokenizer.json>` against
upstream's matching runtime, and check that the tests still pass.

A tag `vYY.MM.N` builds, tests and publishes every platform
(`.github/workflows/release.yml`).
