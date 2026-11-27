# PIJAR-ID v1.0.0 — first public release

Reproducibility artifact for the paper *"PIJAR-ID: Deteksi Indirect Prompt Injection Bahasa
Indonesia dan Campur-Kode di CPU melalui Pemodelan Leksikal yang Diperkuat Augmentasi
Terjemahan"* (Wicaksono & Suryani, Telkom University).

## What's inside
- **Method & code** — word TF-IDF (1-2 gram) + L2 logistic regression hardened by MarianMT
  en->id augmentation and NotInject-style bilingual hard-negatives.
- **Released Indonesian IPI corpus** — id_test (500), id_train_aug (600), id_spotcheck (40),
  hardneg (720).
- **5-seed result ledgers** (std < 0.001) reproducing Tables II-IV.
- **Serving backend** (demo/) — 0.8 MB model + FastAPI /scan with n-gram explainability.
- **One-command reproduction** (run_all.sh), pinned environment, fetch_data.py.

## Headline (held-out, zero-shot)
Beats a production DeBERTa-v3 guard on every out-of-domain axis at ~900x lower CPU latency:
cross-benchmark EN AUROC 0.968 vs 0.741 | cross-lingual ID AUROC 0.998 vs 0.673 |
over-defense FPR 0.136 vs 0.242 | attack recall 0.947 vs 0.880 | 0.53 ms/sample.

## License
MIT. Third-party public datasets are fetched, not redistributed (see expD/data/README.md).
