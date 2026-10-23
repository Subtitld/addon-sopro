// Write tests/fixtures/js_parity.json from upstream's own JavaScript runtime.
//
//   node tools/js_fixtures.mjs <sopro checkout> <tokenizer.json>
//
// The Python port (sopro_addon/text.py, dsp.py) must give the same numbers;
// tests/test_parity.py checks it. Run again when upstream's runtime changes.
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const [checkout, tokenizerPath] = process.argv.slice(2);
const src = (name) => pathToFileURL(path.join(checkout, 'web', 'src', name)).href;
const { SentencePieceTokenizer, normalizeText, splitText } = await import(src('tokenizer.js'));
const dsp = await import(src('dsp.js'));

const b64 = (array) => Buffer.from(new Float32Array(array).buffer).toString('base64');

// A deterministic test signal (tests/test_parity.py makes the same one): a voiced tone with vibrato, pauses and noise.
function signal(rate, seconds) {
  let state = 12345;
  const rnd = () => { state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0; return state / 4294967296 - 0.5; };
  const n = Math.round(rate * seconds), out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / rate, gate = (Math.floor(t * 2.5) % 3 === 2) ? 0.02 : 1;
    out[i] = gate * (0.3 * Math.sin(2 * Math.PI * (140 + 20 * Math.sin(2 * Math.PI * 3 * t)) * t) + 0.1 * Math.sin(2 * Math.PI * 1100 * t)) + 0.01 * rnd();
  }
  return out;
}

const tokenizer = new SentencePieceTokenizer(JSON.parse(readFileSync(tokenizerPath, 'utf8')));
const texts = [
  ['hello there, General Kenobi', null], ['“Quotes” and ‘apostrophes’… ok', 'en'],
  ['olá, tudo bem? Vamos começar.', 'pt'], ['Où est la gare ?', 'fr'],
  ['Straße, Größe und Übermut!', 'de'], ['emoji 🎬 and ½ fractions', 'en'], ['', 'pt'],
];
const wav24 = signal(24000, 4.3), long24 = signal(24000, 13.7);
const features = dsp.referenceFeatures(wav24);
const cropped = dsp.cropAndNormalizeReference(long24, 24000, 10);
const short = dsp.cropAndNormalizeReference(signal(24000, 3.1), 24000, 10);
let state = 7;
const rnd = () => { state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0; return state / 4294967296; };
const istftIn = Float32Array.from({ length: 9 * 1026 }, (_, i) => (i % 1026) < 513 ? rnd() * 3 - 2 : rnd() * 6 - 3);
const longText = 'One sentence here. Another one, with a clause; and more! ' .repeat(12);
const out = {
  tokens: texts.map(([t, l]) => ({ text: t, language: l, normalized: normalizeText(t, l), ids: tokenizer.encode(t, l) })),
  split: { text: longText, segments: splitText(longText, 120) },
  features: {
    semantic: b64(features.semantic.data), semantic_frames: features.semantic.frames,
    speaker: b64(features.speaker.data), speaker_frames: features.speaker.frames,
    acoustic: b64(features.acoustic.data), acoustic_frames: features.acoustic.frames,
    left: Array.from(features.interp.left, Number), right: Array.from(features.interp.right, Number),
    weight: b64(features.interp.weight), n_tokens: features.nTokens,
  },
  resample: b64(dsp.resample(wav24.subarray(0, 4800), 24000, 16000)),
  crop: { length: cropped.wav.length, level_db: cropped.levelDb, head: b64(cropped.wav.subarray(0, 2000)), tail: b64(cropped.wav.subarray(cropped.wav.length - 2000)) },
  crop_short: { length: short.wav.length, level_db: short.levelDb, tail: b64(short.wav.subarray(short.wav.length - 2000)) },
  istft: { input: b64(istftIn), output: b64(dsp.istft(istftIn, 9)) },
  trims: (() => { const s = signal(24000, 2.2); const g = Float32Array.from(s, (v, i) => i < 9000 ? v * 0.001 : v);
    return { lead: dsp.trimLead(g, 24000).length, lead_seg: dsp.trimLead(g, 24000, 0.3, 0.1).length, trail: dsp.trimTrail(g, 24000).length,
             limit: b64(dsp.softLimit(Float32Array.from([0.5, 0.95, -1.4, 2.0]))) }; })(),
};
writeFileSync('tests/fixtures/js_parity.json', JSON.stringify(out));
console.log('wrote tests/fixtures/js_parity.json');
