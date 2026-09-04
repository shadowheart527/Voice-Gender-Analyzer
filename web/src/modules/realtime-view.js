// realtime-view.js — scrolling strip chart for the experimental frame-level
// live mode.  Consumes rt_frames / rt_phones / rt_stats events from
// voiceya.live.realtime and draws the last WINDOW_S seconds:
//
//   ┌ F0 lane: one dot per 10 ms frame, coloured by pitch, zone bands behind
//   ├ phone lane: one block per phone, filled by its resonance score, label on top
//   └ resonance lane: per-phone bars against the language's p25/p75 zone lines
//
// A header row shows rolling numbers (F0 over 3 s, resonance over 4 s, zone,
// Engine A tendency).  The right edge is "now"; the unlabelled strip between
// the last committed phone and now shows how far the phone model lags.

import { divergingPitch, divergingResonance } from "./diverging.js";
import { getLang, t } from "./i18n.js";
import { PITCH_ZONES_HZ, RESONANCE_ZONES } from "./zones.js";

const WINDOW_S = 8;
const F0_MIN = 70;
const F0_MAX = 420;
const KEEP_S = WINDOW_S + 1;

function _cssVar(name, fallback) {
	return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

function _zoneLabel(key) {
	if (!key) return "—";
	const k = `advice.resonance.summary.${key}`;
	const s = t(k);
	return s === k ? key.replace(/_/g, " ") : s;
}

export class RealtimeView {
	constructor(root) {
		this.root = root;
		this.frames = []; // [t, f0, f1, f2, f3]
		this.phones = []; // {start,end,phone,resonance,...}
		this.stats = null;
		this.lastT = 0; // latest audio time known to the server
		this.lastWall = performance.now();
		this.committedT = 0; // end of the last committed phone (or silence)
		this.phoneLagMs = null;
		this._raf = null;
		this._dpr = Math.min(window.devicePixelRatio || 1, 2);
		this._build();
	}

	_build() {
		this.root.innerHTML = `
			<div class="rt-view">
				<div class="rt-head">
					<div class="rt-stat"><span class="rt-stat__label">F0 · 3 s</span><span class="rt-stat__value" data-rt="f0">—</span></div>
					<div class="rt-stat"><span class="rt-stat__label">${t("rt.resonance4s")}</span><span class="rt-stat__value" data-rt="res">—</span></div>
					<div class="rt-stat"><span class="rt-stat__label">${t("rt.zone")}</span><span class="rt-stat__value rt-stat__value--text" data-rt="zone">—</span></div>
					<div class="rt-stat"><span class="rt-stat__label">${t("rt.nn")}</span><span class="rt-stat__value rt-stat__value--text" data-rt="nn">—</span></div>
					<div class="rt-stat"><span class="rt-stat__label">${t("rt.lag")}</span><span class="rt-stat__value" data-rt="lag">—</span></div>
				</div>
				<canvas class="rt-canvas" aria-label="${t("rt.canvasAria")}"></canvas>
				<div class="rt-foot" data-rt="foot"></div>
			</div>`;
		this.canvas = this.root.querySelector(".rt-canvas");
		this.ctx = this.canvas.getContext("2d");
		this.els = {};
		for (const el of this.root.querySelectorAll("[data-rt]")) this.els[el.dataset.rt] = el;
	}

	start() {
		if (this._raf) return;
		const loop = () => {
			this._raf = requestAnimationFrame(loop);
			this._draw();
		};
		this._raf = requestAnimationFrame(loop);
	}

	stop() {
		if (this._raf) cancelAnimationFrame(this._raf);
		this._raf = null;
	}

	destroy() {
		this.stop();
		this.root.innerHTML = "";
	}

	push(ev) {
		switch (ev.type) {
			case "rt_frames": {
				for (const f of ev.frames) this.frames.push(f);
				this.lastT = Math.max(this.lastT, ev.t);
				this.lastWall = performance.now();
				const cut = this.lastT - KEEP_S;
				while (this.frames.length && this.frames[0][0] < cut) this.frames.shift();
				break;
			}
			case "rt_phones": {
				for (const p of ev.phones) {
					this.phones.push(p);
					this.committedT = Math.max(this.committedT, p.end);
				}
				if (ev.lag_ms != null) this.phoneLagMs = ev.lag_ms;
				const cut = this.lastT - KEEP_S;
				while (this.phones.length && this.phones[0].end < cut) this.phones.shift();
				break;
			}
			case "rt_stats":
				this.stats = ev;
				this._renderStats();
				break;
			case "ready":
				if (ev.phones_enabled === false) this.els.foot.textContent = t("rt.noPhones");
				break;
			default:
				break;
		}
	}

	_renderStats() {
		const s = this.stats;
		if (!s) return;
		this.els.f0.textContent = s.f0_med_3s != null ? `${Math.round(s.f0_med_3s)} Hz` : "—";
		this.els.res.textContent = s.res_med_4s != null ? `${Math.round(s.res_med_4s * 100)}%` : "—";
		this.els.zone.textContent = _zoneLabel(s.zone_key);
		const nn = s.nn;
		this.els.nn.textContent = nn?.label
			? `${t(nn.label === "female" ? "rt.fem" : "rt.masc")} ${Math.round((nn.confidence || 0) * 100)}%`
			: "—";
		this.els.lag.textContent = this.phoneLagMs != null ? `${this.phoneLagMs} ms` : "—";
		if (s.phones_enabled === false) this.els.foot.textContent = t("rt.noPhones");
		else this.els.foot.textContent = t("rt.foot", { n: s.vowel_count ?? 0 });
	}

	_now() {
		// Extrapolate between server frames so the strip scrolls smoothly.
		return this.lastT + Math.min(0.5, (performance.now() - this.lastWall) / 1000);
	}

	_draw() {
		const c = this.canvas;
		const cssW = c.clientWidth;
		const cssH = c.clientHeight;
		if (!cssW || !cssH) return;
		const dpr = this._dpr;
		if (c.width !== Math.round(cssW * dpr) || c.height !== Math.round(cssH * dpr)) {
			c.width = Math.round(cssW * dpr);
			c.height = Math.round(cssH * dpr);
		}
		const ctx = this.ctx;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		const W = cssW;
		const H = cssH;
		ctx.clearRect(0, 0, W, H);

		const now = this._now();
		const t0 = now - WINDOW_S;
		const x = (tt) => ((tt - t0) / WINDOW_S) * W;
		const laneF0 = { y: 4, h: H * 0.46 };
		const lanePh = { y: H * 0.5, h: H * 0.16 };
		const laneRes = { y: H * 0.69, h: H * 0.29 };
		const textMuted = _cssVar("--text-muted", "#9b9b93");
		const textSec = _cssVar("--text-secondary", "#6b6b63");
		const border = _cssVar("--border-strong", "rgba(0,0,0,0.14)");
		const lang = getLang();

		// ── F0 lane: zone bands + dots ──────────────────────────
		const fy = (hz) =>
			laneF0.y + laneF0.h - ((Math.log(hz) - Math.log(F0_MIN)) / (Math.log(F0_MAX) - Math.log(F0_MIN))) * laneF0.h;
		ctx.fillStyle = "rgba(90, 120, 200, 0.07)";
		ctx.fillRect(0, fy(PITCH_ZONES_HZ.male), W, laneF0.y + laneF0.h - fy(PITCH_ZONES_HZ.male));
		ctx.fillStyle = "rgba(194, 94, 143, 0.07)";
		ctx.fillRect(0, laneF0.y, W, fy(PITCH_ZONES_HZ.female) - laneF0.y);
		ctx.strokeStyle = border;
		ctx.lineWidth = 1;
		ctx.font = "10px system-ui, sans-serif";
		ctx.fillStyle = textMuted;
		for (const hz of [100, PITCH_ZONES_HZ.male, PITCH_ZONES_HZ.female, 250, 350]) {
			const yy = fy(hz);
			ctx.beginPath();
			ctx.moveTo(0, yy);
			ctx.lineTo(W, yy);
			ctx.stroke();
			ctx.fillText(`${hz}`, 3, yy - 2);
		}
		for (const [tt, f0] of this.frames) {
			if (f0 == null || tt < t0) continue;
			const col = divergingPitch(f0, { stretch: true });
			if (!col) continue;
			ctx.fillStyle = col;
			ctx.beginPath();
			ctx.arc(x(tt), fy(Math.max(F0_MIN, Math.min(F0_MAX, f0))), 2.2, 0, Math.PI * 2);
			ctx.fill();
		}

		// ── phone lane ─────────────────────────────────────────
		ctx.font = "600 11px system-ui, sans-serif";
		ctx.textAlign = "center";
		ctx.textBaseline = "middle";
		for (const p of this.phones) {
			if (p.end < t0) continue;
			const x0 = Math.max(0, x(p.start));
			const x1 = Math.min(W, x(p.end));
			const w = x1 - x0;
			if (w <= 0) continue;
			const col = p.resonance != null ? divergingResonance(p.resonance) : null;
			ctx.fillStyle = col || "rgba(128,128,128,0.18)";
			ctx.fillRect(x0, lanePh.y, w, lanePh.h);
			if (w >= 14) {
				ctx.fillStyle = col ? "#1a1a18" : textSec;
				ctx.fillText(p.phone, x0 + w / 2, lanePh.y + lanePh.h / 2);
			}
		}
		// Unlabelled leading edge (phone model lookahead).
		const xc = x(Math.max(this.committedT, t0));
		if (xc < W) {
			ctx.fillStyle = "rgba(128,128,128,0.08)";
			ctx.fillRect(xc, lanePh.y, W - xc, lanePh.h);
		}

		// ── resonance lane ─────────────────────────────────────
		const zones = RESONANCE_ZONES[lang] || RESONANCE_ZONES["en-US"];
		const ry = (v) => laneRes.y + laneRes.h - v * laneRes.h;
		ctx.textAlign = "left";
		ctx.font = "10px system-ui, sans-serif";
		ctx.strokeStyle = border;
		for (const [v, label] of [
			[zones.p25, "p25"],
			[zones.p75, "p75"],
		]) {
			ctx.beginPath();
			ctx.moveTo(0, ry(v));
			ctx.lineTo(W, ry(v));
			ctx.stroke();
			ctx.fillStyle = textMuted;
			ctx.fillText(label, 3, ry(v) - 2);
		}
		for (const p of this.phones) {
			if (p.end < t0 || p.resonance == null) continue;
			const x0 = Math.max(0, x(p.start));
			const x1 = Math.min(W, x(p.end));
			if (x1 - x0 <= 0) continue;
			ctx.fillStyle = divergingResonance(p.resonance, { stretch: true }) || textSec;
			ctx.globalAlpha = p.is_vowel ? 0.95 : 0.4;
			ctx.fillRect(x0, ry(p.resonance), x1 - x0, laneRes.y + laneRes.h - ry(p.resonance));
			ctx.globalAlpha = 1;
		}

		// "now" line
		ctx.strokeStyle = textSec;
		ctx.beginPath();
		ctx.moveTo(W - 1, 0);
		ctx.lineTo(W - 1, H);
		ctx.stroke();
	}
}
