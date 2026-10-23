# Preset voice clips

Each preset is a clip of a LibriVox volunteer reading a public-domain book.
LibriVox releases its recordings into the public domain. The clips are cut at a
pause to 10 to 14 seconds, resampled to 24 kHz mono and peak-normalized.

| Clip | Voice | Source |
|------|-------|--------|
| `en-woman.wav` | English woman | LibriTTS-R `test-clean`, utterance `4507_16021_000033_000000` (speaker 4507) |
| `en-man.wav` | English man | LibriTTS-R `test-clean`, utterance `7127_75946_000037_000001` (speaker 7127) |
| `pt-woman.wav` | Portuguese woman | [canaa_1901_librivox](https://www.archive.org/download/canaa_1901_librivox/canaa_06_aranha_64kb.mp3), 757.14 s to 774.12 s (MLS speaker 12114) |
| `pt-man.wav` | Portuguese man | [cartas_de_inglaterra_1805_librivox](https://www.archive.org/download/cartas_de_inglaterra_1805_librivox/cartasdeinglaterra_08_queiros_64kb.mp3), 598.53 s to 608.91 s (MLS speaker 12287) |
| `fr-woman.wav` | French woman | [fables_lafontaine_04_0712_librivox](https://www.archive.org/download/fables_lafontaine_04_0712_librivox/fables_04_07_lafontaine_64kb.mp3), 14.41 s to 32.02 s (MLS speaker 2154) |
| `fr-man.wav` | French man | [comte_monte_cristo_jg_librivox](https://www.archive.org/download/comte_monte_cristo_jg_librivox/montecristo_049_dumas_64kb.mp3), 256.23 s to 272.70 s (MLS speaker 2216) |
| `de-woman.wav` | German woman | [20000_meilen_1001_librivox](https://www.archive.org/download/20000_meilen_1001_librivox/20000_meilen_2_23_verne_64kb.mp3), 219.75 s to 233.40 s (MLS speaker 3588) |
| `de-man.wav` | German man | [fliegende_hollaender_0903_librivox](https://www.archive.org/download/fliegende_hollaender_0903_librivox/fliegende-hollaender_08_wolff_64kb.mp3), 248.95 s to 265.60 s (MLS speaker 3040) |

## English: LibriTTS-R

The English clips come from LibriTTS-R, a restored version of LibriTTS (itself
made from LibriSpeech's LibriVox recordings), licensed under the Creative
Commons Attribution 4.0 International licence (`CC-BY-4.0.txt` here):

> Y. Koizumi, H. Zen, S. Karita, Y. Ding, K. Yatabe, N. Morioka, M. Bacchiani,
> Y. Zhang, W. Han and A. Bapna, "LibriTTS-R: A Restored Multi-Speaker
> Text-to-Speech Corpus", Interspeech 2023. https://www.openslr.org/141/

The clips were changed: cut, resampled and peak-normalized.

## Portuguese, French and German: LibriVox

These clips were found through Multilingual LibriSpeech (Pratap et al., 2020,
CC-BY-4.0), whose segment times they use, and cut from the original LibriVox
MP3 recordings above, which are in the public domain. No MLS audio is
included.

## How they were chosen

Candidates were one utterance per speaker from LibriTTS-R `test-clean` and the
MLS test sets. Each was cloned on a line in its own language and in English,
and scored with Whisper (word error rate) and Resemblyzer (similarity to the
clip); the best-scoring clip in each language and pitch range was kept. Woman
and man are guesses from the median pitch, not checked by ear.
