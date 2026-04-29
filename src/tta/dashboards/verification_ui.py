"""Single-trial verification UI. RapidMeta-style: human confirms or
disagrees with algorithmic flag decisions. State persists in
localStorage; export JSON for offline reconciliation.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd

LOCALSTORAGE_KEY = "tta-verification-v0.1.0"

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
.controls { margin-top: 1em; }
.controls button { margin-right: 0.5em; padding: 0.5em 1em; font-size: 1em;
                   border-radius: 4px; border: 1px solid #888; cursor: pointer; }
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
<title>__TITLE__</title>
<style>__CSS__</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <span class="progress">Trial <span id="cur">1</span> of <span id="total"></span></span>
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
function save(state) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}
function record(decision) {
  var state = load();
  var t = TRIALS[idx];
  state[t.nct_id || ("__row__" + idx)] = {
    decision: decision,
    at: new Date().toISOString(),
    snapshot: t
  };
  save(state);
  next();
}
function fmt(v) {
  if (v === null || v === undefined) return "—";
  return String(v);
}
function render() {
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
    var v = fmt(t[f]);
    rows += '<div class="flag-row"><span class="flag-label">' + f + '</span>'
          + '<span class="flag-value flag-value-' + v + '">' + v + '</span></div>';
  }
  document.getElementById("trial-card-host").innerHTML =
    '<div class="trial-card"><h2>' + fmt(t.Study) + '</h2>' + rows
    + '<div class="controls">'
    + '<button class="confirm" data-trial-index="' + idx + '" onclick="record(\'confirm\')">Confirm</button>'
    + '<button class="disagree" onclick="record(\'disagree\')">Disagree</button>'
    + '<button class="skip" onclick="record(\'skip\')">Skip</button>'
    + '</div></div>';
  document.getElementById("cur").textContent = (idx + 1);
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
    # Escape any </script> sequences in the JSON data so they can't break the
    # HTML parser — replaces </ with <\/ which is valid JSON and safe in HTML.
    trials_json = json.dumps(trials, ensure_ascii=False).replace("</", "<\\/")
    return (_TEMPLATE
            .replace("__TITLE__", escape(title))
            .replace("__CSS__", _CSS)
            .replace("__TRIALS_JSON__", trials_json)
            .replace("__STORAGE_KEY__", LOCALSTORAGE_KEY))


def write(atlas: pd.DataFrame, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(atlas, title=title), encoding="utf-8")
