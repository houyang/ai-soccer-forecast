// soccer-agent dashboard — vanilla JS, no build, no framework.
// Single responsibility: fetch /api/dashboard and render.
// All endpoints called: /api/dashboard, POST /predictions, POST /predictions/:id/result.

const $ = (sel) => document.querySelector(sel);
const fmt = (x, digits = 3) => (x === null || x === undefined) ? "—" :
  (typeof x === "number" ? x.toFixed(digits) : String(x));
const fmtPct = (x) => (x === null || x === undefined) ? "—" :
  (typeof x === "number" ? `${(x * 100).toFixed(1)}%` : String(x));

// ---------- summary tiles ----------

function renderSummary(s) {
  $("#t_n_predictions").textContent = fmt(s.n_predictions, 0);
  $("#t_n_resolved").textContent    = fmt(s.n_resolved, 0);
  $("#t_accuracy").textContent      = fmtPct(s.accuracy);
  $("#t_brier").textContent         = fmt(s.brier, 3);
  $("#t_log_loss").textContent      = fmt(s.log_loss, 3);
  $("#t_ece").textContent           = fmt(s.calibration_ece, 3);

  // Color the accuracy tile vs 50% baseline (random) — not great as
  // a hard rule but useful as a quick visual cue.
  const a = $("#t_accuracy");
  a.closest(".tile").classList.remove("good", "bad");
  if (typeof s.accuracy === "number") {
    a.closest(".tile").classList.add(s.accuracy >= 0.5 ? "good" : "bad");
  }
  // Brier: lower is better. 0 = perfect, 0.25 = naive (uniform 1/3).
  const b = $("#t_brier");
  b.closest(".tile").classList.remove("good", "bad");
  if (typeof s.brier === "number") {
    b.closest(".tile").classList.add(s.brier <= 0.2 ? "good" : "bad");
  }
  // ECE: lower is better. < 0.05 is the target from docs/calibration.md.
  const e = $("#t_ece");
  e.closest(".tile").classList.remove("good", "bad");
  if (typeof s.calibration_ece === "number") {
    e.closest(".tile").classList.add(s.calibration_ece <= 0.05 ? "good" : "bad");
  }
}

// ---------- calibration monitor (Task 33) ----------
//
// Shows the *live* effect of the calibrator: how many recent
// predictions had a calibrator applied, and the mean raw→final
// delta. Different from the reliability chart, which is a
// longer-window aggregate of bucket counts.

function renderCalibrationMonitor(cm) {
  const n = cm?.n_with_raw ?? 0;
  $("#calmon_n").textContent = fmt(n, 0);
  $("#calmon_n_calibrated").textContent = fmt(cm?.n_calibrated ?? 0, 0);
  // Mean delta is signed — show "+0.05" / "-0.10". Null-safe.
  const md = cm?.mean_delta;
  const ad = cm?.abs_mean_delta;
  $("#calmon_mean_delta").textContent =
    (md === null || md === undefined) ? "—" : `${md >= 0 ? "+" : ""}${md.toFixed(3)}`;
  $("#calmon_abs_mean").textContent = ad === null || ad === undefined ? "—" : ad.toFixed(3);
  // Calibrator usage breakdown. If only one is in use we show its
  // label; if multiple we summarize.
  const cal = cm?.calibrators || {};
  const labels = Object.keys(cal);
  const summary = labels.length === 0
    ? "—"
    : labels.length === 1
      ? `${labels[0]} (${cal[labels[0]]})`
      : labels
          .map(k => `${k.split("@")[0]}=${cal[k]}`)
          .join(", ");
  $("#calmon_breakdown").textContent = summary;
  // Color the abs-mean tile: < 0.05 tight, > 0.15 moving a lot.
  const a = $("#calmon_abs_mean");
  a.closest(".tile").classList.remove("good", "bad");
  if (typeof ad === "number") {
    a.closest(".tile").classList.add(ad <= 0.05 ? "good" : ad >= 0.15 ? "bad" : "");
  }
}

// ---------- reliability chart (inline SVG) ----------

function renderReliability(cal) {
  const svg = $("#reliability");
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const bins = (cal.raw && cal.raw.reliability) || [];
  $("#cal_n").textContent = cal.n_samples ?? 0;
  if (bins.length === 0) {
    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", 300); t.setAttribute("y", 160);
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("fill", "#8a93a3");
    t.textContent = "no data";
    svg.appendChild(t);
    return;
  }

  const W = 600, H = 320;
  const ML = 50, MR = 20, MT = 20, MB = 40;
  const PW = W - ML - MR, PH = H - MT - MB;

  // axes
  const axes = (() => {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("stroke", "#2a2f3a");
    // x axis
    g.appendChild(line(ML, MT + PH, ML + PW, MT + PH));
    // y axis
    g.appendChild(line(ML, MT, ML, MT + PH));
    // gridlines + labels (y: 0, 0.5, 1)
    [0, 0.5, 1].forEach((v) => {
      const y = MT + PH - v * PH;
      g.appendChild(line(ML, y, ML + PW, y, "#22262f", "1"));
      const lab = document.createElementNS("http://www.w3.org/2000/svg", "text");
      lab.setAttribute("x", ML - 6); lab.setAttribute("y", y + 3);
      lab.setAttribute("text-anchor", "end");
      lab.setAttribute("fill", "#8a93a3");
      lab.setAttribute("font-size", "10");
      lab.textContent = v.toFixed(1);
      g.appendChild(lab);
    });
    // x ticks (bin midpoints)
    bins.forEach((b, i) => {
      const x = ML + (i + 0.5) * (PW / bins.length);
      const lab = document.createElementNS("http://www.w3.org/2000/svg", "text");
      lab.setAttribute("x", x); lab.setAttribute("y", MT + PH + 14);
      lab.setAttribute("text-anchor", "middle");
      lab.setAttribute("fill", "#8a93a3");
      lab.setAttribute("font-size", "9");
      lab.textContent = b.bin_label || `[${(b.lo * 100).toFixed(0)}-${(b.hi * 100).toFixed(0)}]`;
      g.appendChild(lab);
    });
    return g;
  })();
  svg.appendChild(axes);

  // perfect-calibration diagonal
  svg.appendChild(line(ML, MT + PH, ML + PW, MT, "#5a6275", "2 4"));

  // bars: gap between stated confidence (bin midpoint) and actual win rate
  bins.forEach((b, i) => {
    if (b.count === 0) return;
    const x = ML + (i + 0.5) * (PW / bins.length);
    const w = Math.min(34, PW / bins.length * 0.6);
    const mid = b.bin_midpoint ?? ((b.lo + b.hi) / 2);
    const yMid = MT + PH - mid * PH;
    const yObs = MT + PH - b.avg_actual * PH;
    // vertical bar from stated to actual
    const top = Math.min(yMid, yObs);
    const h = Math.abs(yObs - yMid);
    const fill = (b.avg_actual >= mid) ? "#57c785" : "#e26a6a";
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", x - w / 2);
    rect.setAttribute("y", top);
    rect.setAttribute("width", w);
    rect.setAttribute("height", Math.max(1, h));
    rect.setAttribute("fill", fill);
    rect.setAttribute("opacity", "0.85");
    // tooltip via <title>
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `stated ${(mid * 100).toFixed(0)}%, actual ${(b.avg_actual * 100).toFixed(0)}%, n=${b.count}`;
    rect.appendChild(title);
    svg.appendChild(rect);
    // marker dot at the observed point
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("cx", x);
    dot.setAttribute("cy", yObs);
    dot.setAttribute("r", 3.5);
    dot.setAttribute("fill", "#e6e8ec");
    dot.setAttribute("stroke", "#0f1115");
    dot.setAttribute("stroke-width", "1");
    svg.appendChild(dot);
  });
}

function line(x1, y1, x2, y2, stroke = "#2a2f3a", dash = null) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", "line");
  el.setAttribute("x1", x1); el.setAttribute("y1", y1);
  el.setAttribute("x2", x2); el.setAttribute("y2", y2);
  el.setAttribute("stroke", stroke);
  if (dash) el.setAttribute("stroke-dasharray", dash);
  return el;
}

// ---------- predictions table ----------

function renderPredictions(preds) {
  const tbody = $("#predictions tbody");
  tbody.innerHTML = "";
  if (preds.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 6;
    td.className = "muted";
    td.textContent = "no predictions yet — use the form above to create one";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }
  for (const p of preds) {
    const tr = document.createElement("tr");
    tr.appendChild(td(p.match_id));
    const pickTd = document.createElement("td");
    const pickSpan = document.createElement("span");
    pickSpan.className = `pick ${(p.pick || "").toLowerCase()}`;
    pickSpan.textContent = p.pick || "—";
    if (p.result && typeof p.result.was_correct === "boolean") {
      pickSpan.classList.add(p.result.was_correct ? "correct" : "wrong");
    }
    pickTd.appendChild(pickSpan);
    tr.appendChild(pickTd);
    tr.appendChild(td(fmt(p.confidence, 2), "num"));
    // Cal. column — show raw→final delta and the calibrator label
    // so the user can see when the calibrator actually moved a
    // prediction. Falls back to a dash when no calibrator fired.
    const calTd = document.createElement("td");
    calTd.className = "num";
    const raw = (p.model_versions || {}).raw_confidence
      ?? (typeof p.raw_confidence === "number" ? p.raw_confidence : null);
    const cal = (p.model_versions || {}).calibrator
      ?? p.calibrator
      ?? null;
    if (raw != null && cal) {
      const delta = p.confidence - raw;
      const sign = delta >= 0 ? "+" : "";
      calTd.textContent = `${raw.toFixed(2)}→${p.confidence.toFixed(2)} (${sign}${delta.toFixed(2)})`;
      calTd.title = `calibrator: ${cal}`;
      calTd.classList.add("cal-fired");
    } else if (raw != null) {
      calTd.textContent = `${raw.toFixed(2)}→${p.confidence.toFixed(2)}`;
      calTd.title = "no calibrator fitted";
      calTd.classList.add("cal-passthrough");
    } else {
      calTd.textContent = "—";
      calTd.title = "pre-31 prediction (no raw recorded)";
    }
    tr.appendChild(calTd);
    const resultTd = document.createElement("td");
    if (p.result && p.result.home_goals !== null && p.result.home_goals !== undefined) {
      resultTd.textContent = `${p.result.home_goals}-${p.result.away_goals}` +
        (typeof p.result.was_correct === "boolean" ? (p.result.was_correct ? " ✓" : " ✗") : "");
      if (typeof p.result.was_correct === "boolean") {
        resultTd.style.color = p.result.was_correct ? "var(--good)" : "var(--bad)";
      }
    } else {
      resultTd.className = "muted";
      resultTd.textContent = "—";
    }
    tr.appendChild(resultTd);
    tr.appendChild(td(p.result ? fmt(p.result.brier, 3) : "—", "num"));
    tr.appendChild(td((p.created_at || "").replace("T", " ").slice(0, 16), "muted"));
    tbody.appendChild(tr);
  }
}
function td(text, cls) {
  const el = document.createElement("td");
  el.textContent = text;
  if (cls) el.className = cls;
  return el;
}

// ---------- fetch + orchestration ----------

async function refresh() {
  const btn = $("#refresh");
  btn.disabled = true;
  try {
    const r = await fetch("/api/dashboard");
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    const body = await r.json();
    renderSummary(body.summary);
    renderReliability(body.calibration);
    renderCalibrationMonitor(body.calibration_monitor || {});
    renderPredictions(body.predictions);
    $("#generated_at").textContent = `updated ${(body.generated_at || "").replace("T", " ").slice(0, 19)} UTC`;
  } catch (e) {
    $("#generated_at").textContent = `error: ${e.message}`;
  } finally {
    btn.disabled = false;
  }
}

// ---------- forms ----------

function setStatus(el, msg, kind) {
  el.textContent = msg;
  el.classList.remove("error", "ok");
  if (kind) el.classList.add(kind);
}

async function postJson(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = "";
    try { detail = (await r.json()).detail || ""; } catch { detail = await r.text(); }
    throw new Error(`HTTP ${r.status}: ${detail || r.statusText}`);
  }
  return r.json();
}

document.getElementById("predict").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const status = $("#predict_status");
  const fd = new FormData(ev.target);
  const payload = {
    home_id: fd.get("home_id"),
    away_id: fd.get("away_id"),
    venue_id: fd.get("venue_id"),
    kickoff: fd.get("kickoff"),
    competition: fd.get("competition") || "UCL",
    round: fd.get("round") || null,
  };
  setStatus(status, "running agent…");
  try {
    const r = await postJson("/predictions", payload);
    setStatus(status, `saved: pick=${r.pick}, confidence=${r.confidence}`, "ok");
    ev.target.reset();
    await refresh();
  } catch (e) {
    setStatus(status, e.message, "error");
  }
});

document.getElementById("result").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const status = $("#result_status");
  const fd = new FormData(ev.target);
  const match_id = fd.get("match_id");
  const body = {
    home_goals: parseInt(fd.get("home_goals"), 10),
    away_goals: parseInt(fd.get("away_goals"), 10),
  };
  setStatus(status, "saving result…");
  try {
    const r = await postJson(`/predictions/${encodeURIComponent(match_id)}/result`, body);
    setStatus(status, `recorded ${r.result.home_goals}-${r.result.away_goals}; correct=${r.result.was_correct}`, "ok");
    await refresh();
  } catch (e) {
    setStatus(status, e.message, "error");
  }
});

$("#refresh").addEventListener("click", refresh);

// initial load + auto-refresh every 30s
refresh();
setInterval(refresh, 30000);
