"""Single-trial verification UI. RapidMeta-style: human confirms or
disagrees with algorithmic flag decisions. State persists in
localStorage; export JSON for offline reconciliation.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd

# Repo-slug-prefixed to avoid collision on the multi-tenant github.io origin.
LOCALSTORAGE_KEY = "tta/trial-truthfulness-atlas/verification-v0.1.0"

_CSS = """
body { font-family: -apple-system, system-ui, sans-serif; max-width: 900px;
       margin: 1em auto; padding: 0 1em; color: #1a1a1a; }
header { display: flex; justify-content: space-between; align-items: baseline; }
.trial-card { border: 1px solid #ccc; border-radius: 8px; padding: 1em;
              margin-top: 1em; background: #fcfcfc; }
.trial-card h2 { margin-top: 0; }
.flag-row { display: grid; grid-template-columns: 200px 1fr; gap: 0.4em;
            padding: 0.4em 0; border-bottom: 1px dashed #eee; }
.flag-label { font-weight: bold; color: #333; }
.flag-value { font-family: monospace; }
.flag-value-flagged, .flag-value-required_not_posted,
.flag-value-substantively_different, .flag-value-flipped,
.flag-value-unbridgeable { color: #b00020; }
.flag-value-identical, .flag-value-concordant,
.flag-value-not_flagged, .flag-value-posted { color: #006d32; }
.flag-value-unscoreable, .flag-value-refinement,
.flag-value-not_required { color: #886000; }
.flag-symbol { display: inline-block; width: 1.2em; font-weight: bold; }
.status-msg { display: block; margin-top: 0.6em; min-height: 1.4em;
              font-size: 0.9em; color: #666; }
.status-msg.error { color: #b00020; }
.trial-card[tabindex] { outline: none; }
.trial-card[tabindex]:focus { box-shadow: 0 0 0 3px #4a90e2; }
@media (max-width: 480px) {
  .flag-row { grid-template-columns: 1fr; }
  .flag-label { padding-top: 0.2em; }
}
.controls { margin-top: 1em; }
/* min-height 44px meets Apple HIG / WCAG 2.5.5 minimum tap target. */
.controls button { margin-right: 0.5em; padding: 0.7em 1.2em; font-size: 1em;
                   min-height: 44px; min-width: 44px;
                   border-radius: 4px; border: 1px solid #888; cursor: pointer; }
.nav button { padding: 0.7em 1.2em; font-size: 1em;
              min-height: 44px; min-width: 44px;
              border-radius: 4px; border: 1px solid #888; cursor: pointer;
              margin-right: 0.5em; }
.controls button.confirm { background: #d9f0d9; }
.controls button.disagree { background: #ffd9d9; }
.controls button.skip { background: #f0e6c0; }
.nav { margin-top: 1em; }
.progress { color: #666; }
""".strip()


def _trial_to_dict(row: pd.Series) -> dict:
    keys = [
        "nct_id", "review_id", "review_doi", "Study", "bridge_method",
        "registered_outcome", "ma_extracted_outcome", "outcome_drift",
        "registered_n", "ma_n", "n_drift",
        "registered_effect_log", "ma_effect_log", "direction_concordance",
        "results_posting",
    ]
    out = {}
    for k in keys:
        v = row.get(k)
        if pd.isna(v):
            out[k] = None
        elif isinstance(v, (int, float, str, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


# NOTE: The </script> tag below closes the ONE <script> block in the page.
# Inside the JS body, JSON data is escaped via .replace("</", "<\\/") so
# no trial text field can inject a stray </script> tag.
# Python source uses \\ to write a single backslash into the template string.
_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta property="og:title" content="__TITLE__">
<meta property="og:type" content="website">
<meta property="og:description" content="Trial Truthfulness Atlas verification UI — confirm or disagree with algorithmic flag decisions, one trial at a time.">
<title>__TITLE__</title>
<style>__CSS__</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <span class="progress" aria-live="polite" aria-atomic="true">Trial <span id="cur">1</span> of <span id="total"></span></span>
</header>
<div id="trial-card-host"></div>
<div class="nav">
  <button id="prev">Previous</button>
  <button id="next">Next</button>
  <button id="export">Export verifications</button>
</div>
<script>
var TRIALS = __TRIALS_JSON__;
var STORAGE_KEY = "__STORAGE_KEY__";
var idx = 0;

function load() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); }
  catch (e) { return {}; }
}
function setStatus(msg, isError) {
  var el = document.getElementById("status-msg");
  if (!el) return;
  el.textContent = msg || "";
  el.classList.toggle("error", !!isError);
}
function save(state) {
  // Wrap in try/catch — Safari private browsing throws SecurityError;
  // any browser can throw QuotaExceededError. Without the catch, the
  // verification decision would silently fail to persist with no feedback.
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    return true;
  } catch (e) {
    setStatus("Saving failed (" + (e.name || "Error") + "). Export now to "
              + "preserve your work; private browsing or quota may block "
              + "storage.", true);
    return false;
  }
}
function record(decision) {
  var state = load();
  var t = TRIALS[idx];
  state[t.nct_id || ("__row__" + idx)] = {
    decision: decision,
    at: new Date().toISOString(),
    snapshot: t
  };
  if (save(state)) {
    setStatus("Recorded: " + decision);
  }
  next();
}
// Color-blind-safe symbol prefix per flag value. Pairs with the existing
// red/green/yellow color so the meaning survives without color perception.
var FLAG_SYMBOL = {
  flagged: "✗", required_not_posted: "✗",
  substantively_different: "✗", flipped: "✗",
  unbridgeable: "✗",
  identical: "✓", concordant: "✓",
  not_flagged: "✓", posted: "✓",
  unscoreable: "~", refinement: "~", not_required: "~",
  dossiergap_direct: "✓"
};
function flagSymbol(v) {
  return FLAG_SYMBOL[v] || "";
}
// HTML-entity-escape — required because innerHTML is used below for
// performance. Without this, a study label with <img onerror=...> XSSes.
function esc(v) {
  if (v === null || v === undefined) return "—";
  return String(v).replace(/&/g, "&amp;").replace(/</g, "&lt;")
                  .replace(/>/g, "&gt;").replace(/"/g, "&quot;")
                  .replace(/'/g, "&#39;");
}
// CSS class fragments must be alphanumeric/underscore — guard against
// injection of attribute-breaking characters via flag values.
var SAFE_CLASS_RE = /^[a-z_]+$/;
function safeClass(v) {
  return SAFE_CLASS_RE.test(String(v)) ? String(v) : "unknown";
}
// Backward-compat alias (pre-XSS-fix code referenced fmt()).
function fmt(v) { return esc(v); }
function render() {
  if (!TRIALS.length) {
    document.getElementById("trial-card-host").innerHTML =
      '<div class="trial-card"><h2>No trials to review</h2>'
      + '<p>The atlas is empty. Run <code>tta build --fixture-mode</code> '
      + 'to populate it.</p></div>';
    document.getElementById("prev").disabled = true;
    document.getElementById("next").disabled = true;
    return;
  }
  var t = TRIALS[idx];
  var flags = ["bridge_method", "outcome_drift", "n_drift",
               "direction_concordance", "results_posting"];
  var rows = "";
  rows += '<div class="flag-row"><span class="flag-label">NCT</span>'
        + '<span class="flag-value">' + fmt(t.nct_id) + '</span></div>';
  rows += '<div class="flag-row"><span class="flag-label">Study</span>'
        + '<span class="flag-value">' + fmt(t.Study) + '</span></div>';
  rows += '<div class="flag-row"><span class="flag-label">Review DOI</span>'
        + '<span class="flag-value">' + fmt(t.review_doi) + '</span></div>';
  rows += '<div class="flag-row"><span class="flag-label">Registered outcome</span>'
        + '<span class="flag-value">' + fmt(t.registered_outcome) + '</span></div>';
  rows += '<div class="flag-row"><span class="flag-label">MA outcome</span>'
        + '<span class="flag-value">' + fmt(t.ma_extracted_outcome) + '</span></div>';
  rows += '<div class="flag-row"><span class="flag-label">Registered N</span>'
        + '<span class="flag-value">' + fmt(t.registered_n) + '</span></div>';
  rows += '<div class="flag-row"><span class="flag-label">MA N</span>'
        + '<span class="flag-value">' + fmt(t.ma_n) + '</span></div>';
  for (var i = 0; i < flags.length; i++) {
    var f = flags[i];
    var raw = t[f];
    var v = esc(raw);
    var cls = safeClass(raw);
    var symbol = flagSymbol(raw);
    var symHtml = symbol
      ? '<span class="flag-symbol" aria-hidden="true">' + symbol + '</span>'
      : '';
    rows += '<div class="flag-row"><span class="flag-label">' + esc(f) + '</span>'
          + '<span class="flag-value flag-value-' + cls + '">'
          + symHtml + v + '</span></div>';
  }
  document.getElementById("trial-card-host").innerHTML =
    '<div class="trial-card" tabindex="-1"><h2>' + fmt(t.Study) + '</h2>'
    + rows
    + '<div class="controls">'
    + '<button class="confirm" data-trial-index="' + idx + '" onclick="record(\'confirm\')">Confirm</button>'
    + '<button class="disagree" onclick="record(\'disagree\')">Disagree</button>'
    + '<button class="skip" onclick="record(\'skip\')">Skip</button>'
    + '<span class="status-msg" id="status-msg" role="status" aria-live="polite"></span>'
    + '</div></div>';
  document.getElementById("cur").textContent = (idx + 1);
  // Move focus to the new card so keyboard users land on the new content
  // instead of the now-stale Next/Previous button. tabindex=-1 makes the
  // card focusable without inserting it into the natural Tab order.
  var card = document.querySelector(".trial-card");
  if (card && typeof card.focus === "function") { card.focus(); }
}
function next() {
  if (idx < TRIALS.length - 1) { idx += 1; }
  render();
}
function prev() {
  if (idx > 0) { idx -= 1; }
  render();
}
function downloadJson() {
  var blob = new Blob([JSON.stringify(load(), null, 2)],
    {type: "application/json"});
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  document.body.appendChild(a);
  a.href = url;
  a.download = "tta-verifications.json";
  a.click();
  setTimeout(function(){ URL.revokeObjectURL(url); a.remove(); }, 1000);
  setStatus("Downloaded tta-verifications.json");
}
document.getElementById("total").textContent = TRIALS.length;
document.getElementById("prev").onclick = prev;
document.getElementById("next").onclick = next;
document.getElementById("export").onclick = downloadJson;
render();
</script>
</body>
</html>
"""


def render(atlas: pd.DataFrame, title: str) -> str:
    trials = [_trial_to_dict(r) for _, r in atlas.iterrows()]
    # ensure_ascii=True forces non-ASCII to \uXXXX so multi-byte sequences
    # cannot resolve to </script>; the .replace then covers the literal-ASCII
    # </ case. Both layers are needed against Unicode-bypass attacks.
    trials_json = json.dumps(trials, ensure_ascii=True).replace("</", "<\\/")
    return (_TEMPLATE
            .replace("__TITLE__", escape(title))
            .replace("__CSS__", _CSS)
            .replace("__TRIALS_JSON__", trials_json)
            .replace("__STORAGE_KEY__", LOCALSTORAGE_KEY))


def write(atlas: pd.DataFrame, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(atlas, title=title), encoding="utf-8")
