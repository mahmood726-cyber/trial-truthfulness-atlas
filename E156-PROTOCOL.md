# E156-PROTOCOL — Trial Truthfulness Atlas v0.1.0

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
direction concordance, and FDAAA results-posting compliance — across the
v0.1.0 cardiology subset of Pairwise70 (595 reviews; ~6,386 MAs in the
full dataset). Bridge resolution used DossierGap direct match in v0.1.0;
outcome-drift used a frozen `gemma2:9b` local-LLM judge with sha256-pinned
prompts and disk-cached judgments. The v0.1.0 fixture (5 trials) returned
80 % bridge resolution, 1/4 substantively different outcome, 1/4 N drift,
0/4 direction flips, and 1/3 FDAAA results-posting violations. Production
v0.2.0 will widen to all ~6,386 MAs and add three additional bridge
methods. The headline integrity rate (and its association with crosses-
null status) is reported on `dashboard.html`; reproducibility is gated by
the pinned numerical baseline at `baseline/cardio_v0.1.0.json`. v0.1.0
treats the unbridgeable rate as a publishable finding in itself, since
silent CT.gov-to-MA non-resolution is the upstream confounder of every
prior reproducibility atlas.

## Author block

Mahmood Ahmad — middle-author on all E156 papers (per portfolio
convention; see global feedback memory).

## Dashboard

Pages link: github.io/trial-truthfulness-atlas/  (live after Pages
enable)

YOUR REWRITE:

SUBMITTED: [ ]
