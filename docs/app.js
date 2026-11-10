/* Scramble Drill Score — static site logic. No framework, no build step. */

const COLORS = {
  text: "#e8e8ea", muted: "#9aa0aa", accent: "#8b7cf6",
  green: "#34d399", red: "#ef4444", blue: "#3b82f6",
  border: "#2c3038", card: "#1f2229",
};
const FIELD_LEN_VIS = 40; // yards of field length shown around the action

let SITE = null;

init();

async function init() {
  try {
    SITE = await (await fetch("data/site.json")).json();
  } catch (e) {
    document.getElementById("hero-explain").textContent =
      "Could not load data/site.json. Run `python main.py` to generate it.";
    return;
  }
  renderHero();
  renderKeyNumbers();
  renderPayoff();
  renderPlays();
  renderLeaderboard();
  renderByPosition();
  renderTakeaways();
}

/* ---------------- 1. HERO ---------------- */
function renderHero() {
  const h = SITE.headline;
  document.getElementById("hero-gain").textContent = h.gainers_caught_pct + "%";
  document.getElementById("hero-gain-detail").textContent =
    `caught when the receiver gained 1+ yd of separation (${h.gainers_detail})`;
  document.getElementById("hero-nongain").textContent =
    h.nongainers_caught_pct + "%";
  document.getElementById("hero-nongain-detail").textContent =
    `caught when he held steady, within ±1 yd (${h.nongainers_detail})`;

  document.getElementById("hero-explain").textContent =
    `On a scramble, the play breaks down and the quarterback leaves the pocket. ` +
    `We measure how much separation each receiver gains between that moment and the throw. ` +
    `The gap above is statistically significant (two-proportion z-test p = ${h.z_test_p}). ` +
    `Even after controlling for how open the receiver already was and his position, each extra ` +
    `yard of separation at the throw multiplies the odds of a completion by ${h.logistic_odds_ratio_per_yd}× ` +
    `(logistic regression p = ${h.logistic_p_controlled}).`;
}

/* ---------------- 2. KEY NUMBERS ---------------- */
function renderKeyNumbers() {
  const row = document.getElementById("key-numbers-row");
  const kn = SITE.key_numbers;
  const cards = [
    kn.scramble_dropbacks,
    kn.scramble_throws_with_start,
    kn.median_scramble_duration_s,
    kn.targeted_throws,
    kn.z_test_p,
    // Controlled odds ratio from the headline model (adjusted for sep_start + position).
    {
      value: SITE.headline.logistic_odds_ratio_per_yd + "×",
      label: "Odds of completion per +1 yd (adjusted for starting separation and position)",
    },
  ];
  row.innerHTML = cards.map((n) => `<div class="card kn-card">
      <div class="kn-value">${n.value}</div>
      <div class="kn-label">${n.label}</div>
    </div>`).join("");
}

/* ---------------- 3. THE PAYOFF ---------------- */
function renderPayoff() {
  const b = SITE.payoff_buckets;
  const labels = b.map((x) => x.bucket);
  const comp = b.map((x) => x.completion_pct);
  const ypa = b.map((x) => x.yards_per_attempt);
  const ns = b.map((x) => x.n);

  const overallComp = weightedMean(comp, ns);
  const overallYpa = weightedMean(ypa, ns);

  barChart("chart-comp", labels, comp, ns, COLORS.green, "%", overallComp);
  barChart("chart-ypa", labels, ypa, ns, COLORS.blue, " yd", overallYpa);

  const lost = b.find((x) => x.bucket.startsWith("Lost"));
  const held = b.find((x) => x.bucket.startsWith("Held"));
  document.getElementById("payoff-note").textContent =
    `Why does "Lost 1+" beat "Held"? Those receivers (often running backs) started wide open ` +
    `(${lost.mean_sep_start} yds from a defender) and were still at ${lost.mean_sep_throw} yds at the throw, ` +
    `more open than the "Held" group (${held.mean_sep_throw} yds).`;
}

function weightedMean(vals, weights) {
  const s = weights.reduce((a, b) => a + b, 0);
  return vals.reduce((a, v, i) => a + v * weights[i], 0) / s;
}

function barChart(canvasId, labels, data, ns, color, unit, overall) {
  const overallLine = {
    id: "overallLine",
    afterDraw(chart) {
      const y = chart.scales.y.getPixelForValue(overall);
      const { left, right } = chart.chartArea;
      const ctx = chart.ctx;
      ctx.save();
      ctx.strokeStyle = COLORS.accent;
      ctx.setLineDash([6, 5]);
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
      ctx.setLineDash([]);
      // label pinned to the LEFT edge, BELOW the line, so it clears the
      // bar-top value labels (which always sit above the bars)
      const label = `overall ${overall.toFixed(1)}${unit}`;
      ctx.font = "bold 12px sans-serif";
      const lw = ctx.measureText(label).width + 10;
      ctx.fillStyle = "rgba(22,24,29,0.85)";
      ctx.fillRect(left + 2, y + 4, lw, 16);
      ctx.fillStyle = COLORS.accent;
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(label, left + 6, y + 6);
      ctx.textBaseline = "alphabetic";
      ctx.restore();
    },
  };
  // value on top of each bar, with n below it
  const barLabels = {
    id: "barLabels",
    afterDatasetsDraw(chart) {
      const ctx = chart.ctx;
      const meta = chart.getDatasetMeta(0);
      ctx.save();
      ctx.textAlign = "center";
      meta.data.forEach((bar, i) => {
        ctx.fillStyle = COLORS.text;
        ctx.font = "bold 13px sans-serif";
        ctx.fillText(`${data[i]}${unit}`, bar.x, bar.y - 20);
        ctx.fillStyle = COLORS.muted;
        ctx.font = "11px sans-serif";
        ctx.fillText(`n=${ns[i]}`, bar.x, bar.y - 6);
      });
      ctx.restore();
    },
  };
  new Chart(document.getElementById(canvasId), {
    type: "bar",
    data: { labels, datasets: [{ data, backgroundColor: color, borderRadius: 6 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false },
        tooltip: { callbacks: { label: (c) => `${c.raw}${unit} (n=${ns[c.dataIndex]})` } } },
      layout: { padding: { top: 28 } },
      scales: {
        x: { ticks: { color: COLORS.muted }, grid: { color: COLORS.border } },
        y: {
          beginAtZero: true,
          suggestedMax: Math.max(...data, overall) * 1.18,
          ticks: { color: COLORS.muted }, grid: { color: COLORS.border },
        },
      },
    },
    plugins: [overallLine, barLabels],
  });
}

/* ---------------- 4. WATCH THE PLAYS ---------------- */
const PV = {
  data: null,      // loaded play json
  meta: null,      // play row from site.json
  frame: 0,
  playing: false,
  speed: 1,
  tracked: null,   // nflId being tracked (defaults to featured)
  raf: null,
  lastTs: 0,
  acc: 0,          // accumulated ms toward next data frame
  view: null,      // cached field crop {xLo,xHi}
};

function resultWord(r) {
  return r === "C" ? "Complete" : r === "IN" ? "Intercepted" : "Incomplete";
}

function renderPlays() {
  const list = document.getElementById("play-list");

  // Order: Lee Smith first (if present), then the rest in site.json order.
  const order = SITE.plays.map((_, i) => i);
  const lee = SITE.plays.findIndex((p) => /lee smith/i.test(p.receiver));
  if (lee > 0) { order.splice(lee, 1); order.unshift(lee); }

  list.innerHTML = order.map((i) => {
    const p = SITE.plays[i];
    const sepCls = p.sep_gained >= 0 ? "pos" : "neg";
    const sign = p.sep_gained >= 0 ? "+" : "";
    const rl = p.result === "C" ? "C" : p.result === "IN" ? "INT" : "I";
    const pillCls = p.result === "C" ? "c" : "i";
    const pillTxt = p.result === "C" ? "Complete" : p.result === "IN" ? "INT" : "Incomplete";
    return `<div class="play-item" data-idx="${i}">
      <div class="pi-top">
        <span class="pi-sep ${sepCls}">${sign}${p.sep_gained} yd</span>
        <span class="pi-pill ${pillCls}">${pillTxt}</span>
      </div>
      <div class="pi-name">${p.receiver} <span style="color:var(--muted)">(${p.position})</span></div>
      <div class="pi-sub">${p.team} vs ${p.opponent}, Wk ${p.week}</div>
    </div>`;
  }).join("");

  list.querySelectorAll(".play-item").forEach((el) => {
    el.addEventListener("click", () => selectPlay(+el.dataset.idx));
  });

  // controls
  document.getElementById("play-btn").addEventListener("click", togglePlay);
  const scr = document.getElementById("scrubber");
  scr.addEventListener("input", (e) => { pause(); PV.frame = +e.target.value; drawFrame(); updateControls(); });

  document.querySelectorAll("#speed-toggle button").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#speed-toggle button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      PV.speed = parseFloat(b.dataset.sp);
    });
  });

  setupFieldInteraction();
  setupKeys();

  // default to Lee Smith if present, else top play
  let idx = SITE.plays.findIndex((p) => /lee smith/i.test(p.receiver));
  if (idx < 0) idx = 0;
  selectPlay(idx);
}

async function selectPlay(idx) {
  pause();
  document.querySelectorAll(".play-item").forEach((x) =>
    x.classList.toggle("active", +x.dataset.idx === idx));
  const meta = SITE.plays[idx];
  PV.meta = meta;

  // header
  document.getElementById("ph-title").textContent =
    `${meta.receiver} (${meta.position}), ${meta.team} vs ${meta.opponent}, Wk ${meta.week}`;
  document.getElementById("ph-sub").textContent =
    `Q${meta.quarter} · ${meta.down_distance} · ${meta.playDescription}`;
  const pill = document.getElementById("result-pill");
  pill.textContent = resultWord(meta.result);
  pill.className = "result-pill " + (meta.result === "C" ? "c" : "i");

  try {
    PV.data = await (await fetch(`data/plays/${meta.gameId}_${meta.playId}.json`)).json();
  } catch (e) {
    document.getElementById("stat-strip").textContent = "Play file not available.";
    return;
  }
  PV.frame = 0;
  PV.tracked = PV.data.featured_nflId;
  PV.view = null;
  const scr = document.getElementById("scrubber");
  scr.max = PV.data.frames.length - 1;
  scr.value = 0;
  buildTicks();
  renderStatStrip();
  drawFrame();
  updateControls();
}

function buildTicks() {
  const d = PV.data, n = d.frames.length - 1;
  const fid = (f) => d.frames.findIndex((x) => x.frameId === f);
  const marks = [
    [fid(d.snap_frame), "snap"],
    [d.scramble_start_frame != null ? fid(d.scramble_start_frame) : -1, "scramble"],
    [fid(d.throw_frame), "throw"],
  ];
  document.getElementById("ticks").innerHTML = marks
    .filter(([i]) => i >= 0)
    .map(([i, label]) => `<span class="tick" style="left:${(i / n) * 100}%">${label}</span>`)
    .join("");
}

/* --- phase label --- */
function phaseLabel(fr, d) {
  const t = ((fr.frameId - d.snap_frame) / 10).toFixed(1);
  let phase;
  if (fr.frameId < d.snap_frame) phase = "Pre-snap";
  else if (fr.frameId >= d.throw_frame) phase = "Throw";
  else if (d.scramble_start_frame != null && fr.frameId >= d.scramble_start_frame) phase = "SCRAMBLE";
  else phase = "Dropback";
  return `${phase} | ${t >= 0 ? "" : ""}${t} s`;
}

/* --- playback --- */
function togglePlay() { PV.playing ? pause() : play(); }
function play() {
  if (!PV.data) return;
  PV.playing = true;
  document.getElementById("play-btn").textContent = "❚❚ Pause";
  PV.lastTs = performance.now(); PV.acc = 0;
  PV.raf = requestAnimationFrame(tick);
}
function pause() {
  PV.playing = false;
  document.getElementById("play-btn").textContent = "▶ Play";
  if (PV.raf) { cancelAnimationFrame(PV.raf); PV.raf = null; }
}
function tick(ts) {
  if (!PV.playing) return;
  const dt = ts - PV.lastTs; PV.lastTs = ts;
  PV.acc += dt * PV.speed;
  const stepMs = 100; // 10 fps data rate
  while (PV.acc >= stepMs) {
    PV.acc -= stepMs;
    PV.frame++;
    if (PV.frame > PV.data.frames.length - 1) { PV.frame = 0; }
  }
  document.getElementById("scrubber").value = PV.frame;
  drawFrame(); updateControls();
  PV.raf = requestAnimationFrame(tick);
}
function updateControls() {
  const d = PV.data; if (!d) return;
  const fr = d.frames[PV.frame];
  document.getElementById("play-clock").textContent =
    ((fr.frameId - d.snap_frame) / 10).toFixed(1) + "s";
}

/* --- nearest defender to the tracked receiver --- */
function nearestDef(fr, who) {
  if (!who) return null;
  let nd = null, best = Infinity;
  fr.players.filter((p) => p.team === "def").forEach((p) => {
    const dd = Math.hypot(p.x - who.x, p.y - who.y);
    if (dd < best) { best = dd; nd = p; }
  });
  return nd ? { def: nd, dist: best } : null;
}

/* --- draw one frame --- */
function drawFrame() {
  const d = PV.data; if (!d) return;
  const canvas = document.getElementById("field");
  const fr = d.frames[PV.frame];
  const dpr = window.devicePixelRatio || 1;

  // cache a stable crop across the whole play (so it doesn't jitter)
  if (!PV.view) {
    const xs = [], ys = [];
    d.frames.forEach((f) => f.players.forEach((p) => { xs.push(p.x); ys.push(p.y); }));
    let xLo = Math.min(d.los_x - 10, ...xs) - 3;
    let xHi = Math.max(d.los_x + 20, ...xs) + 3;
    if (xHi - xLo < d.field_width + 6) {
      const pad = (d.field_width + 6 - (xHi - xLo)) / 2; xLo -= pad; xHi += pad;
    }
    PV.view = { xLo, xHi };
  }
  const { xLo, xHi } = PV.view;
  const W = canvas.clientWidth;
  const H = Math.round(W * (d.field_width / (xHi - xLo)));
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.height = H + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  PV.sx = (x) => ((x - xLo) / (xHi - xLo)) * W;
  PV.sy = (y) => H - (y / d.field_width) * H;
  const sx = PV.sx, sy = PV.sy;
  PV.W = W; PV.H = H;

  // field
  ctx.fillStyle = "#2f6b3a"; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "rgba(255,255,255,0.5)"; ctx.lineWidth = 1;
  ctx.fillStyle = "rgba(255,255,255,0.55)"; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
  for (let yd = Math.ceil(xLo / 10) * 10; yd < xHi; yd += 10) {
    ctx.beginPath(); ctx.moveTo(sx(yd), 0); ctx.lineTo(sx(yd), H); ctx.stroke();
    const num = Math.round(yd) % 100;
    ctx.fillText(String(num).padStart(2, "0"), sx(yd), H - 6);
  }
  // line of scrimmage (dashed blue)
  ctx.strokeStyle = "#3b82f6"; ctx.lineWidth = 2.5; ctx.setLineDash([8, 6]);
  ctx.beginPath(); ctx.moveTo(sx(d.los_x), 0); ctx.lineTo(sx(d.los_x), H); ctx.stroke();
  ctx.setLineDash([]);

  // pocket rectangle at snap (shaded) — around the QB's snap spot
  const snapFr = d.frames.find((f) => f.frameId === d.snap_frame);
  const qbSnap = snapFr && snapFr.players.find((p) => p.role === "Pass");
  if (qbSnap) {
    ctx.fillStyle = "rgba(139,124,246,0.12)";
    const px = sx(d.los_x - 7), pw = sx(d.los_x) - sx(d.los_x - 7);
    const py = sy(qbSnap.y + 4), ph = sy(qbSnap.y - 4) - sy(qbSnap.y + 4);
    ctx.fillRect(px, py, pw, ph);
  }

  // fading trails for QB and tracked receiver
  drawTrail(ctx, d, fr, (p) => p.role === "Pass", "rgba(250,204,21,0.5)");
  drawTrail(ctx, d, fr, (p) => p.nflId === PV.tracked, "rgba(52,211,153,0.6)");

  // tracked receiver -> nearest defender line + offset label
  const who = fr.players.find((p) => p.nflId === PV.tracked);
  if (who) {
    const nd = nearestDef(fr, who);
    if (nd) {
      ctx.strokeStyle = "#34d399"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(sx(who.x), sy(who.y));
      ctx.lineTo(sx(nd.def.x), sy(nd.def.y)); ctx.stroke();
      // label offset 1.5 yd from the receiver along the line, boxed
      const vx = nd.def.x - who.x, vy = nd.def.y - who.y;
      const norm = Math.hypot(vx, vy) || 1;
      const lx = sx(who.x + (vx / norm) * 1.5);
      const ly = sy(who.y + (vy / norm) * 1.5) - 10;
      labelBox(ctx, lx, ly, nd.dist.toFixed(1) + " yd");
    }
  }

  // players
  fr.players.forEach((p) => {
    if (p.team === "ball") {
      ctx.fillStyle = "#8B5A2B";
      ctx.beginPath(); ctx.ellipse(sx(p.x), sy(p.y), 5, 3.2, 0, 0, 7); ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.7)"; ctx.lineWidth = 0.8; ctx.stroke();
      return;
    }
    const isTracked = p.nflId === PV.tracked;
    let fill = p.team === "off" ? "#3b82f6" : "#ef4444";
    if (p.role === "Pass") fill = "#facc15";        // QB yellow
    if (isTracked) fill = "#34d399";                // tracked receiver green
    const r = isTracked ? 11 : 9;
    ctx.beginPath(); ctx.arc(sx(p.x), sy(p.y), r, 0, 7);
    ctx.fillStyle = fill; ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = isTracked ? 2.5 : 1.2; ctx.stroke();
    if (p.jersey != null) {
      ctx.fillStyle = p.role === "Pass" ? "#000" : "#fff";
      ctx.font = "bold 9px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(p.jersey, sx(p.x), sy(p.y)); ctx.textBaseline = "alphabetic";
    }
  });

  // phase badge
  labelBox(ctx, 58, 18, phaseLabel(fr, d), "#8b7cf6", "left");
}

function drawTrail(ctx, d, fr, pred, color) {
  const pts = [];
  for (let i = 0; i <= PV.frame; i++) {
    const p = d.frames[i].players.find(pred);
    if (p) pts.push(p);
  }
  if (pts.length < 2) return;
  ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.beginPath();
  pts.forEach((p, i) => {
    const X = PV.sx(p.x), Y = PV.sy(p.y);
    i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
  });
  ctx.stroke();
}

function labelBox(ctx, x, y, text, color = "#facc15", align = "center") {
  ctx.font = "bold 12px sans-serif";
  const w = ctx.measureText(text).width + 12;
  const bx = align === "left" ? x - 6 : x - w / 2;
  ctx.fillStyle = "rgba(5,16,28,0.9)";
  roundRect(ctx, bx, y - 11, w, 20, 5); ctx.fill();
  ctx.strokeStyle = color; ctx.lineWidth = 1; roundRect(ctx, bx, y - 11, w, 20, 5); ctx.stroke();
  ctx.fillStyle = color; ctx.textAlign = align === "left" ? "left" : "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, align === "left" ? x : x, y - 1);
  ctx.textBaseline = "alphabetic";
}
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function renderStatStrip() {
  const m = PV.meta;
  const d = PV.data;
  // sep at scramble start & throw for the featured receiver (from frames)
  const at = (fid) => {
    const f = d.frames.find((x) => x.frameId === fid);
    if (!f) return null;
    const who = f.players.find((p) => p.nflId === d.featured_nflId);
    if (!who) return null;
    const nd = nearestDef(f, who);
    return nd ? nd.dist : null;
  };
  const ss = d.scramble_start_frame != null ? at(d.scramble_start_frame) : null;
  const st = at(d.throw_frame);
  const cls = m.sep_gained >= 0 ? "pos" : "neg";
  const sign = m.sep_gained >= 0 ? "+" : "";
  document.getElementById("stat-strip").innerHTML = `
    <div class="ss-item">sep at scramble start<b>${ss != null ? ss.toFixed(1) + " yd" : "—"}</b></div>
    <div class="ss-item">sep at throw<b>${st != null ? st.toFixed(1) + " yd" : "—"}</b></div>
    <div class="ss-item">gained<b class="${cls}">${sign}${m.sep_gained} yd</b></div>
    <div class="ss-item">result<b>${resultWord(m.result)}${m.yards != null ? ", " + m.yards + " yd" : ""}</b></div>`;
}

/* --- interaction: drag to scrub, click to track a receiver --- */
function setupFieldInteraction() {
  const canvas = document.getElementById("field");
  let dragging = false;
  let moved = false;

  const scrubAt = (clientX) => {
    const rect = canvas.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    PV.frame = Math.round(frac * (PV.data.frames.length - 1));
    document.getElementById("scrubber").value = PV.frame;
    drawFrame(); updateControls();
  };

  canvas.addEventListener("mousedown", (e) => { dragging = true; moved = false; });
  window.addEventListener("mousemove", (e) => {
    if (!dragging || !PV.data) return;
    moved = true; pause(); scrubAt(e.clientX);
  });
  window.addEventListener("mouseup", (e) => {
    if (dragging && !moved && PV.data) trackNearestOffense(e);
    dragging = false;
  });
}

function trackNearestOffense(e) {
  const canvas = document.getElementById("field");
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const fr = PV.data.frames[PV.frame];
  let best = Infinity, pick = null;
  fr.players.filter((p) => p.team === "off").forEach((p) => {
    const dd = Math.hypot(PV.sx(p.x) - mx, PV.sy(p.y) - my);
    if (dd < best && dd < 24) { best = dd; pick = p; }
  });
  if (pick && pick.nflId != null) { PV.tracked = pick.nflId; drawFrame(); }
}

function setupKeys() {
  const canvas = document.getElementById("field");
  canvas.addEventListener("keydown", (e) => {
    if (!PV.data) return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    else if (e.code === "ArrowRight") {
      e.preventDefault(); pause();
      PV.frame = Math.min(PV.data.frames.length - 1, PV.frame + 1);
      document.getElementById("scrubber").value = PV.frame; drawFrame(); updateControls();
    } else if (e.code === "ArrowLeft") {
      e.preventDefault(); pause();
      PV.frame = Math.max(0, PV.frame - 1);
      document.getElementById("scrubber").value = PV.frame; drawFrame(); updateControls();
    }
  });
}

/* ---------------- 5. LEADERBOARD ---------------- */
let lbState = { pos: "All", search: "", sort: "mean_sep_gained", dir: -1, showAll: false };

function renderLeaderboard() {
  document.querySelectorAll("#lb-chips .chip").forEach((c) => {
    c.addEventListener("click", () => {
      document.querySelectorAll("#lb-chips .chip").forEach((x) => x.classList.remove("active"));
      c.classList.add("active");
      lbState.pos = c.dataset.pos;
      drawLeaderboard();
    });
  });
  document.getElementById("lb-search").addEventListener("input", (e) => {
    lbState.search = e.target.value.toLowerCase();
    drawLeaderboard();
  });
  document.getElementById("lb-showall").addEventListener("click", (e) => {
    lbState.showAll = !lbState.showAll;
    e.target.classList.toggle("active", lbState.showAll);
    e.target.textContent = lbState.showAll ? "Show top 10 / bottom 5" : "Show all players";
    drawLeaderboard();
  });
  document.querySelectorAll("#lb-table th").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      if (lbState.sort === key) lbState.dir *= -1;
      else { lbState.sort = key; lbState.dir = typeof SITE.leaderboard[0][key] === "number" ? -1 : 1; }
      drawLeaderboard();
    });
  });
  drawLeaderboard();
}

function lbRowHtml(r, maxAbs, divider) {
  const w = (Math.abs(r.mean_sep_gained) / maxAbs) * 90;
  const cls = r.mean_sep_gained >= 0 ? "pos" : "neg";
  const sign = r.mean_sep_gained >= 0 ? "+" : "";
  const divRow = divider
    ? `<tr class="lb-divider"><td colspan="7">⋯ bottom 5 ⋯</td></tr>` : "";
  return divRow + `<tr>
    <td>${r.name}</td><td>${r.position}</td><td>${r.team}</td>
    <td class="num">${r.snaps}</td>
    <td class="num"><span class="sep-bar-wrap"><span>${sign}${r.mean_sep_gained}</span>
      <span class="sep-bar ${cls}" style="width:${w}px"></span></span></td>
    <td class="num">${r.targets}</td><td class="num">${r.completions}</td>
  </tr>`;
}

function drawLeaderboard() {
  let rows = SITE.leaderboard.slice();
  if (lbState.pos !== "All") rows = rows.filter((r) => r.position === lbState.pos);
  if (lbState.search) rows = rows.filter((r) => r.name.toLowerCase().includes(lbState.search));
  const k = lbState.sort, dir = lbState.dir;
  rows.sort((a, b) => {
    if (typeof a[k] === "number") return (a[k] - b[k]) * dir;
    return String(a[k]).localeCompare(String(b[k])) * dir;
  });

  const maxAbs = Math.max(...SITE.leaderboard.map((r) => Math.abs(r.mean_sep_gained)), 0.1);
  const filtered = lbState.pos !== "All" || lbState.search;

  let html;
  // Default view (no filter/search, not showAll): top 10 + bottom 5 by sep_gained.
  if (!lbState.showAll && !filtered && rows.length > 15 && k === "mean_sep_gained") {
    const byGain = SITE.leaderboard.slice().sort((a, b) => b.mean_sep_gained - a.mean_sep_gained);
    const top = byGain.slice(0, 10);
    const bottom = byGain.slice(-5);
    html = top.map((r) => lbRowHtml(r, maxAbs, false)).join("") +
      bottom.map((r, i) => lbRowHtml(r, maxAbs, i === 0)).join("");
  } else {
    html = rows.map((r) => lbRowHtml(r, maxAbs, false)).join("");
  }
  document.getElementById("lb-body").innerHTML = html;

  document.querySelectorAll("#lb-table th").forEach((th) =>
    th.classList.toggle("sorted", th.dataset.sort === k));
}

/* ---------------- 6. BY POSITION ---------------- */
function renderByPosition() {
  const p = SITE.by_position;
  new Chart(document.getElementById("chart-pos"), {
    type: "bar",
    data: {
      labels: p.map((x) => `${x.position} (n=${x.n})`),
      datasets: [
        { label: "Separation at scramble start", data: p.map((x) => x.mean_sep_start), backgroundColor: COLORS.muted, borderRadius: 6 },
        { label: "Separation at throw", data: p.map((x) => x.mean_sep_throw), backgroundColor: COLORS.green, borderRadius: 6 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: COLORS.text } } },
      scales: {
        x: { ticks: { color: COLORS.muted }, grid: { color: COLORS.border } },
        y: { beginAtZero: true, ticks: { color: COLORS.muted }, grid: { color: COLORS.border },
          title: { display: true, text: "yards of separation", color: COLORS.muted } },
      },
    },
  });
}

/* ---------------- 7. TAKEAWAYS ---------------- */
function renderTakeaways() {
  const h = SITE.headline, q = SITE.qb_most_open;
  const lift = (h.gainers_caught_pct - h.nongainers_caught_pct).toFixed(1);
  document.getElementById("takeaway-rec").textContent =
    `Keep working after the quarterback breaks the pocket. Gaining 1+ yard of separation during the ` +
    `scramble is worth about +${lift} completion percentage points over holding your ground ` +
    `(${h.gainers_caught_pct}% vs ${h.nongainers_caught_pct}%).`;
  document.getElementById("takeaway-qb").textContent =
    `Keep your eyes up. The most open receiver gets the ball on only ${q.pct_to_most_open}% of scramble throws — ` +
    `yet when he does, those throws complete at ${q.completion_pct_most_open}% versus ` +
    `${q.completion_pct_not_most_open}% when the quarterback goes elsewhere.`;
}
