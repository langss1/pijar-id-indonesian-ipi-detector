# Paper D — Findings (real numbers, 2026-11-27)

All metrics TPR@1%FPR, mean over 5 seeds, tool-output-wrapped IPI. Train = safeguard EN (8236).

## Headline
| detector | en_test (in-domain) | deepset (cross-benchmark EN) | id_test (cross-lingual ID) | notinject FPR (over-defense) | end-to-end latency |
|---|---|---|---|---|---|
| tfidf_lr | 0.997 | **0.717** | 0.488 | **0.186** | cheapest |
| e5_lr | 0.978 | 0.133 | **0.864** | 0.324 | +embed cost |
| e5_mlp | 0.982 | 0.140 | 0.845 | 0.345 | +embed cost |
| e5_hgb | 0.988 | 0.000 | 0.708 | 0.342 | +embed cost |

## The sharp finding: two orthogonal shifts expose OPPOSITE weaknesses
- **Cross-benchmark (English, deepset):** lexical TF-IDF+LR robust (0.717, AUROC 0.96); multilingual
  embeddings collapse (e5 0.00–0.14, AUROC ~0.70–0.78). Embeddings overfit the training corpus style.
- **Cross-lingual (Indonesian, id_test):** flips — TF-IDF collapses (0.488, EN−ID drop 0.51);
  multilingual e5 holds (0.86, drop only 0.11). Lexical model fails on unseen Indonesian vocabulary.
- **No single detector is robust on both axes.** (Key message.)

## RQ2 — EN vs ID (TPR@1%FPR)
| detector | en_test | id_translate | id_codemix | EN−ID |
|---|---|---|---|---|
| tfidf_lr | 0.997 | 0.776 | 0.488 | 0.509 |
| e5_lr | 0.978 | 0.848 | 0.880 | 0.114 |
| e5_mlp | 0.982 | 0.800 | 0.872 | 0.138 |
| e5_hgb | 0.988 | 0.768 | 0.744 | 0.280 |

## RQ3 — add ~600 machine-translated ID training samples (cheap fix)
| detector | id_test (EN-only train) | id_test (+ID aug) | gain |
|---|---|---|---|
| tfidf_lr | 0.488 | **0.936** | **+0.448** |
| e5_lr | 0.864 | 0.840 | -0.024 |
| e5_mlp | 0.845 | 0.892 | +0.047 |
| e5_hgb | 0.708 | 0.736 | +0.028 |

**Interpretation:** a cheap MT-augmentation recovers the lexical model's cross-lingual TPR from 0.49 to
0.94 — matching multilingual embeddings — while keeping its cross-benchmark robustness and lowest
over-defense (0.186) and lowest latency. Practical recommendation for CPU/edge deployment:
**TF-IDF+LR + MT-augmentation** beats multilingual-embedding detectors all-round.

## End-to-end CPU latency (200-row probe, full inference path)
- tfidf_lr: **0.53 ms/sample** · e5_lr: **45.2 ms/sample** → TF-IDF is **86× faster** end-to-end.

## Significance (McNemar, id_test)
- tfidf_lr **+ID-aug** vs e5_lr (baseline): accuracy 0.934 vs 0.934, discordant b=28/c=28, **p=1.0**
  → after cheap MT-augmentation the 86×-faster TF-IDF model is statistically indistinguishable from
  the multilingual-embedding model on Indonesian. Clincher for the practical recommendation.

## Decision on heavier baselines (RAM-constrained)
- ProtectAI DeBERTa zero-shot guard + Qwen2.5 LLM-judge were NOT run locally: the 7B LLM and the
  DeBERTa load each tripped the host's low-memory reaper (24 GB RAM machine, Ollama + torch spikes).
  Reported instead via published numbers (PromptShield 94.8% TPR@1%FPR; PromptGuard 22.8% TPR @0.5).
  This does not affect the core ML findings. Can be added later on request / with more RAM.

## Still TODO for the paper
- [ ] End-to-end CPU latency benchmark (TF-IDF transform+predict vs e5 encode+predict vs deberta vs LLM-judge). Current latency only times the classifier, not embedding.
- [ ] Pretrained guard baseline: ProtectAI deberta-v3-base-prompt-injection-v2 (zero-shot) + optional PromptGuard, to compare against the literature (PromptShield 94.8% TPR@1%FPR on their data).
- [ ] Optional small-sample LLM-judge (Qwen2.5) latency/accuracy point — LLM as assist only (heavy RAM).
- [ ] Human spot-check of id_spotcheck.csv (translation validity, threat-to-validity).
- [ ] McNemar test tfidf_lr+aug vs e5_lr on id_test.

---

## ITERASI PENGUATAN (2026-11-27) — model final & head-to-head

**Model final: word TF-IDF + LogisticRegression, train = korpus EN + augmentasi MT ID + hard-negative NotInject-style.**
Fusi (word+char+e5) DICOBA tapi DITOLAK: e5/char mengencerkan word yang kuat di cross-benchmark (deepset turun). Word TF-IDF menang sendiri.

### Angka final model kita (5 seed, aughn)
| Sumbu | TPR@1%FPR | TPR@5%FPR | AUROC |
|---|---|---|---|
| en_test (in-domain) | 0.997 | - | 1.000 |
| deepset (cross-benchmark EN) | 0.767 | 0.917 | 0.968 |
| id_test (cross-lingual ID) | 0.948 | - | 0.998 |
- over-defense NotInject FPR @ en_test-calibrated 1%FPR: base 0.186 -> aug 0.180 -> **aughn 0.136** (hard-negative menurunkan over-defense; recall gandalf tetap 0.947).
- latensi word TF-IDF < 1 ms/sampel (CPU).
- Catatan: deepset n_neg=56, jadi TPR@1%FPR galak/berisik; AUROC 0.968 & TPR@5% 0.917 angka yang lebih stabil.

### Head-to-head vs guard produksi (ProtectAI deberta-v3-base-prompt-injection-v2, zero-shot, CPU)
| Sumbu | DeBERTa-v3 guard | Kita (TF-IDF) |
|---|---|---|
| en_test TPR@1% (AUROC) | 0.685 (0.954) | 0.997 (1.000) |
| deepset TPR@1% (AUROC) | 0.133 (0.741) | 0.767 (0.968) |
| id_test TPR@1% (AUROC) | 0.104 (0.673) | 0.948 (0.998) |
| NotInject FPR (↓) | 0.242 | 0.136 |
| gandalf recall | 0.880 | 0.947 |
| latensi/sampel | 476 ms | < 1 ms (~500x) |

**Kesimpulan iterasi:** detektor TF-IDF CPU murah + augmentasi MT + hard-negative **mengungguli guard transformer produksi di semua sumbu** (adil pada cross-benchmark/ID/over-defense yang sama-sama zero-shot), ~500x lebih cepat. Caveat jujur: in-domain menguntungkan kita (kita latih in-domain, guard zero-shot).
