"""Stream a WAV file to the live worker at real-time pace and print the events.

Usage:
    python scripts/live_e2e_client.py path/to/16k_mono.wav [--url ws://127.0.0.1:8091/api/live]
        [--language en-US] [--mode script] [--script-file rainbow.txt] [--speed 1.0] [--out result.json]

Exercises the full sentence-live pipeline without a browser: VAD → Whisper →
script cursor → sidecar → aggregated summary.  ``--speed 0`` streams as fast
as possible (useful for throughput checks, not latency).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave

import websockets


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--url", default="ws://127.0.0.1:8091/api/live")
    ap.add_argument("--language", default="en-US")
    ap.add_argument("--mode", default="free", choices=["free", "script"])
    ap.add_argument("--script-file", default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with wave.open(args.wav, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == 16000, (
            "need 16 kHz mono s16"
        )
        pcm = w.readframes(w.getnframes())
    script = open(args.script_file, encoding="utf-8").read().strip() if args.script_file else None

    chunk_bytes = 2 * 16 * args.chunk_ms
    t0 = time.perf_counter()
    last = None
    n_summary = 0

    async with websockets.connect(args.url, max_size=1 << 22) as ws:

        async def reader():
            nonlocal last, n_summary
            async for raw in ws:
                ev = json.loads(raw)
                dt = time.perf_counter() - t0
                kind = ev.get("type")
                if kind == "summary":
                    n_summary += 1
                    last = ev["result"]
                    s = ev["sentence"]
                    ec = ev["result"]["summary"].get("engine_c") or {}
                    print(
                        f"[{dt:6.2f}s] summary #{ev['index']}  sent {s['start']:.1f}-{s['end']:.1f}s"
                        f"  latency={s['latency_ms']}ms  matched={s.get('matched')}"
                        f"  phones_total={ec.get('phone_count')}  zone={ec.get('resonance_zone_key')}"
                        f"  text={s['transcript'][:60]!r}"
                    )
                elif kind == "vad":
                    print(f"[{dt:6.2f}s] vad speaking={ev['speaking']} t={ev['t']}")
                elif kind == "done":
                    print(f"[{dt:6.2f}s] done sentences={ev.get('sentences')}")
                    return
                else:
                    print(f"[{dt:6.2f}s] {json.dumps(ev)[:160]}")

        reader_task = asyncio.create_task(reader())
        await ws.send(
            json.dumps(
                {"type": "start", "language": args.language, "mode": args.mode, "script": script}
            )
        )
        pos = 0
        while pos < len(pcm):
            await ws.send(pcm[pos : pos + chunk_bytes])
            pos += chunk_bytes
            if args.speed > 0:
                target = t0 + (pos / 32000.0) / args.speed
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.wait_for(reader_task, timeout=300)

    if args.out and last:
        json.dump(last, open(args.out, "w"), indent=1, ensure_ascii=False)
        print("saved", args.out)
    return 0 if n_summary else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
