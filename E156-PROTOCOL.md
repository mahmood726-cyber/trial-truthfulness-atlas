# E156-PROTOCOL — Trial Truthfulness Atlas v0.1.1

Project: `trial-truthfulness-atlas`
Repo: github.com/mahmood726-cyber/trial-truthfulness-atlas
Pages: github.io/trial-truthfulness-atlas/
Date drafted: 2026-04-29
Date submitted: pending
Submitted to: Synthēsis Methods Note + E156 micro-paper
DOI: pending (Crossref at publication)

## Body (CURRENT BODY — AI version, freely updated until SUBMITTED [x])

Cardiology Cochrane meta-analyses pool effect estimates from trials whose
CT.gov registration may differ from what Cochrane extracted. We
quantified five integrity flags — NCT-bridge, outcome drift, N drift,
direction concordance, and FDAAA results-posting compliance — on a
v0.1.1 fixture of 5 cardiology trials (PARADIGM-HF, VICTORIA, GRIPHON,
plus 2 synthetic cases); the full sweep across ~6,386 MAs in 595 Pairwise70
reviews is deferred to v0.2.0. Bridge resolution used DossierGap direct
match in v0.1.1; outcome-drift used a frozen `gemma2:9b` local-LLM judge
with sha256-pinned prompts and disk-cached judgments. The v0.1.1 fixture
returned 80 % bridge resolution (4/5), 1/4 substantively different
outcomes, 1/4 N drift, 0/4 direction flips, and 1/4 FDAAA results-posting
violations among the four FDAAA-applicable bridged trials. The FDAAA
12-month deadline is anchored to AACT `primary_completion_date` per
42 CFR 11.64(b)(1)(ii). Production v0.2.0 will widen to all ~6,386 MAs,
add three additional bridge methods, and extend direction-concordance to
RR/OR/MD effects. The headline integrity rate (and its association with
crosses-null status) is reported on `dashboard.html`; reproducibility is
gated by the pinned numerical baseline at `baseline/cardio_v0.1.0.json`.
v0.1.1 treats the unbridgeable rate as a publishable finding in itself,
since silent CT.gov-to-MA non-resolution is the upstream confounder of
every prior reproducibility atlas.

## Author block

Mahmood Ahmad — middle-author on all E156 papers (per portfolio
convention; see global feedback memory).

## Dashboard

Pages link: github.io/trial-truthfulness-atlas/  (live after Pages
enable)

YOUR REWRITE:

SUBMITTED: [ ]
