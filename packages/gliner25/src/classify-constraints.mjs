/**
 * Constrained classification decode over per-label sigmoid scores.
 * implies[src] = [dst, ...]  keys are "task:label" or "label" (unique across tasks).
 * excludes same. Single-label tasks are mutex within the task.
 */
function keyOf(task, label) {
  return `${task}:${label}`;
}

function violates(selected, implies, excludes) {
  const set = new Set(selected);
  for (const [src, dsts] of Object.entries(implies || {})) {
    if (!set.has(src)) continue;
    for (const d of dsts) if (!set.has(d)) return true;
  }
  for (const [src, dsts] of Object.entries(excludes || {})) {
    if (!set.has(src)) continue;
    for (const d of dsts) if (set.has(d)) return true;
  }
  return false;
}

function choicesForTask(name, spec) {
  const labels = spec.labels || [];
  const scores = spec.scores || {};
  const multi = Boolean(spec.multi_label ?? spec.multiLabel);
  const threshold = spec.threshold ?? 0.5;
  if (multi) {
    const on = labels.filter((l) => (scores[l] ?? 0) >= threshold);
    const off = [[]];
    const combos = [on];
    if (on.length === 0) return [[]];
    return combos;
  }
  const ranked = labels.slice().sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0));
  return ranked.slice(0, 4).map((l) => [l]);
}

export function decodeConstrained(tasks, { implies = {}, excludes = {}, beamSize = 16 } = {}) {
  const names = Object.keys(tasks);
  let beam = [{ selected: [], score: 0 }];
  for (const name of names) {
    const spec = tasks[name];
    const options = choicesForTask(name, spec);
    const expanded = [];
    for (const st of beam) {
      for (const pick of options) {
        const selected = st.selected.concat(pick.map((l) => keyOf(name, l)));
        if (violates(selected, implies, excludes)) continue;
        let gain = 0;
        for (const l of pick) gain += spec.scores?.[l] ?? 0;
        expanded.push({ selected, score: st.score + gain, pickByTask: { ...st.pickByTask, [name]: pick } });
      }
    }
    expanded.sort((a, b) => b.score - a.score);
    beam = expanded.slice(0, beamSize);
    if (!beam.length) {
      beam = [{ selected: [], score: 0, pickByTask: {} }];
      break;
    }
  }
  const best = beam[0] || { pickByTask: {}, score: 0 };
  const out = {};
  for (const name of names) {
    const spec = tasks[name];
    const pick = best.pickByTask?.[name] || [];
    const multi = Boolean(spec.multi_label ?? spec.multiLabel);
    if (multi) {
      out[name] = pick.map((label) => ({ label, score: spec.scores?.[label] ?? 0 }));
    } else {
      const label = pick[0] ?? spec.labels?.[0];
      out[name] = {
        label,
        score: spec.scores?.[label] ?? 0,
        scores: spec.scores,
      };
    }
  }
  out._constrained = true;
  out._score = best.score;
  return out;
}
