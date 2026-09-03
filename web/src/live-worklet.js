// AudioWorkletProcessor for the sentence-live path.
//
// Takes the mic graph's float32 input, converts to little-endian int16 mono at
// 16 kHz and posts ~100 ms chunks to the main thread, which forwards them over
// the WebSocket.  We ask the AudioContext for 16 kHz up front (Chrome and
// Firefox honour it and resample the mic properly); if the context runs at
// another rate we fall back to linear resampling here so the server always
// sees 16 kHz.

const TARGET_RATE = 16000;
const CHUNK_SAMPLES = 1600; // 100 ms at 16 kHz

class Pcm16CaptureProcessor extends AudioWorkletProcessor {
	constructor() {
		super();
		this._step = sampleRate / TARGET_RATE; // input samples per output sample
		this._pos = 0; // fractional read position relative to the current block
		this._prev = 0; // last sample of the previous block (for interpolation at pos<1)
		this._buf = new Int16Array(CHUNK_SAMPLES);
		this._n = 0;
		this._running = true;
		this.port.onmessage = (e) => {
			if (e.data === "stop") this._running = false;
		};
	}

	_push(v) {
		const s = v < -1 ? -1 : v > 1 ? 1 : v;
		this._buf[this._n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
		if (this._n === CHUNK_SAMPLES) {
			const out = this._buf;
			this.port.postMessage(out.buffer, [out.buffer]);
			this._buf = new Int16Array(CHUNK_SAMPLES);
			this._n = 0;
		}
	}

	process(inputs) {
		if (!this._running) return false;
		const ch = inputs[0]?.[0];
		if (!ch || ch.length === 0) return true;
		if (this._step === 1) {
			for (let i = 0; i < ch.length; i++) this._push(ch[i]);
			return true;
		}
		// Linear interpolation between sample (i-1) and sample i, where a
		// position in [0,1) interpolates between the previous block's last
		// sample and this block's first.
		let pos = this._pos;
		while (pos < ch.length) {
			const i = Math.floor(pos);
			const frac = pos - i;
			const a = i === 0 ? this._prev : ch[i - 1];
			const b = ch[i];
			this._push(a + (b - a) * frac);
			pos += this._step;
		}
		this._pos = pos - ch.length;
		this._prev = ch[ch.length - 1];
		return true;
	}
}

registerProcessor("pcm16-capture", Pcm16CaptureProcessor);
