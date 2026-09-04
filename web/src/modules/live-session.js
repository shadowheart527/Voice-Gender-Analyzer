// live-session.js — browser side of sentence-live mode.
//
// Attaches an AudioWorklet to the recorder's MediaStream, streams 16 kHz
// int16 PCM over a WebSocket to the live worker (/api/live, proxied by Vite
// to voiceya.live), and surfaces the worker's events.  The summary events
// carry a batch-shaped result, so main.js renders them through the same
// code path as a finished analysis.

const WORKLET_URL = new URL("../live-worklet.js", import.meta.url);

function _wsUrl() {
	const proto = location.protocol === "https:" ? "wss:" : "ws:";
	return `${proto}//${location.host}/api/live`;
}

export async function probeLive() {
	try {
		const r = await fetch("/api/live/healthz", { cache: "no-store" });
		if (!r.ok) return null;
		return await r.json();
	} catch {
		return null;
	}
}

/**
 * @param {{
 *   stream: MediaStream,
 *   language: string,
 *   mode: "free"|"script",
 *   script: string|null,
 *   onEvent: (ev: object) => void,
 *   onError: (msg: string) => void,
 * }} opts
 * @returns {Promise<{ stop: () => Promise<void>, sampleRate: number }>}
 */
export async function startLiveSession({
	stream,
	language,
	mode,
	script,
	onEvent,
	onError,
	realtime = false,
	chunkMs = 100,
}) {
	// Ask for 16 kHz so the browser does the resampling; fall back to the
	// device rate and let the worklet resample.
	let ctx;
	try {
		ctx = new AudioContext({ sampleRate: 16000 });
	} catch {
		ctx = new AudioContext();
	}
	await ctx.audioWorklet.addModule(WORKLET_URL);
	const source = ctx.createMediaStreamSource(stream);
	const node = new AudioWorkletNode(ctx, "pcm16-capture", {
		numberOfInputs: 1,
		numberOfOutputs: 0,
		channelCount: 1,
		channelCountMode: "explicit",
		processorOptions: { chunkSamples: Math.max(160, Math.round((16000 * chunkMs) / 1000)) },
	});

	const ws = new WebSocket(_wsUrl());
	ws.binaryType = "arraybuffer";
	let doneResolve;
	const donePromise = new Promise((res) => (doneResolve = res));
	let opened = false;
	let stopping = false;

	await new Promise((resolve, reject) => {
		ws.addEventListener("open", () => {
			opened = true;
			ws.send(JSON.stringify({ type: "start", language, mode, script: script || null, realtime }));
			resolve();
		});
		ws.addEventListener("error", () => reject(new Error("live socket failed")));
		ws.addEventListener("close", () => {
			if (!opened) reject(new Error("live socket closed"));
		});
	});

	ws.addEventListener("message", (e) => {
		if (typeof e.data !== "string") return;
		let ev;
		try {
			ev = JSON.parse(e.data);
		} catch {
			return;
		}
		if (ev.type === "done") doneResolve();
		if (ev.type === "error" && !stopping) onError?.(ev.msg || "live error");
		onEvent?.(ev);
	});
	ws.addEventListener("close", () => doneResolve());
	ws.addEventListener("error", () => {
		if (!stopping) onError?.("live socket error");
		doneResolve();
	});

	node.port.onmessage = (e) => {
		if (ws.readyState === WebSocket.OPEN && !stopping) ws.send(e.data);
	};
	source.connect(node);
	if (ctx.state === "suspended") await ctx.resume().catch(() => {});

	async function stop() {
		if (stopping) return donePromise;
		stopping = true;
		try {
			node.port.postMessage("stop");
			source.disconnect();
			node.disconnect();
		} catch {}
		try {
			await ctx.close();
		} catch {}
		if (ws.readyState === WebSocket.OPEN) {
			ws.send(JSON.stringify({ type: "stop" }));
			// Wait for the worker to drain queued sentences (bounded).
			await Promise.race([donePromise, new Promise((r) => setTimeout(r, 30000))]);
			try {
				ws.close();
			} catch {}
		}
	}

	return { stop, sampleRate: ctx.sampleRate };
}
