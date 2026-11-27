# Minting the archival DOI (Zenodo) — last mile

The repository is fully DOI-ready. Only **one action requires the maintainer's Zenodo account**
(it cannot be automated from here): flipping the GitHub<->Zenodo switch ON. After that, the DOI
is issued automatically.

## Status checklist
- [x] Public GitHub repo, MIT license, CITATION.cff
- [x] Pinned environment, run_all.sh, fetch_data.py, result ledgers
- [x] Release notes prepared (RELEASE_NOTES_v1.0.0.md)
- [ ] **Zenodo toggle ON for this repo**  <- maintainer, ~30 s
- [ ] GitHub Release v1.0.0 published  <- automatable once toggle is ON
- [ ] DOI issued by Zenodo -> badge added to README + CITATION.cff

## Steps
1. zenodo.org -> your name (top-right) -> GitHub.
2. Find pijar-id-indonesian-ipi-detector; if absent click Sync now.
3. Flip its toggle to ON (installs the webhook).
4. Publish the v1.0.0 release (notes are ready) -> this triggers Zenodo.
5. Zenodo mints two DOIs: a version DOI (v1.0.0) and a concept DOI (always latest).
   Cite the concept DOI in the paper's Data & Code Availability statement.

> Order matters: Zenodo only archives releases created AFTER the toggle is ON.
