// Core math for the COMPAS fairness explorer.
// Model: each person gets a risk score s in (0,1). Scores are perfectly
// calibrated by construction: P(re-arrested | score = s) = s, in BOTH groups.
// A group's score distribution is Beta(p*k, (1-p)*k), so its mean (= base rate) is p.

const GRID = 1000;

function lgamma(x) {
  // Lanczos approximation
  const g = 7;
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

// Discretised score distribution: arrays s[i], w[i] (weights sum to 1)
function scoreDist(baseRate, concentration) {
  const a = baseRate * concentration, b = (1 - baseRate) * concentration;
  const lnB = lgamma(a) + lgamma(b) - lgamma(a + b);
  const s = new Array(GRID), w = new Array(GRID);
  let tot = 0;
  for (let i = 0; i < GRID; i++) {
    s[i] = (i + 0.5) / GRID;
    w[i] = Math.exp((a - 1) * Math.log(s[i]) + (b - 1) * Math.log(1 - s[i]) - lnB);
    tot += w[i];
  }
  for (let i = 0; i < GRID; i++) w[i] /= tot;
  return { s, w };
}

// Confusion matrix as expected counts out of n people, flagging score >= t as "high risk"
function confusion(dist, t, n = 1000) {
  let TP = 0, FP = 0, FN = 0, TN = 0;
  for (let i = 0; i < GRID; i++) {
    const s = dist.s[i], w = dist.w[i] * n;
    if (s >= t) { TP += w * s; FP += w * (1 - s); }
    else { FN += w * s; TN += w * (1 - s); }
  }
  return { TP, FP, FN, TN };
}

function metrics(c) {
  const safe = (a, b) => (b > 0 ? a / b : NaN);
  return {
    fpr: safe(c.FP, c.FP + c.TN),   // not re-arrested but labeled high risk
    fnr: safe(c.FN, c.FN + c.TP),   // re-arrested but labeled low risk
    ppv: safe(c.TP, c.TP + c.FP),   // of those labeled high risk, share re-arrested
    flagged: safe(c.TP + c.FP, c.TP + c.FP + c.FN + c.TN),
  };
}

// Find threshold for dist so that metric(key) matches target. Bisection.
// fpr falls as t rises; ppv rises as t rises.
function thresholdFor(dist, key, target) {
  let lo = 0, hi = 1;
  const dir = key === "fpr" ? -1 : 1;
  for (let i = 0; i < 50; i++) {
    const mid = (lo + hi) / 2;
    const v = metrics(confusion(dist, mid))[key];
    if (Number.isNaN(v)) { hi = mid; continue; }
    if ((v - target) * dir < 0) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

if (typeof module !== "undefined") {
  module.exports = { scoreDist, confusion, metrics, thresholdFor, GRID };
}
