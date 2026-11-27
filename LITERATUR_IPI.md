# Studi Literatur & Pemetaan Kompetitor — Deteksi Indirect Prompt Injection (Paper D)

Disusun 2026-11-27. Gaya mengikuti IJIES (Related Work bertema + tabel perbandingan + positioning).
Semua angka & klaim dari sumber yang disitasi; yang belum terverifikasi diberi tanda [verifikasi].

---

## 2. Related work

### 2.1. Detektor prompt-injection & protokol evaluasi low-FPR
PromptShield menegaskan bahwa AUROC menyesatkan untuk guardrail dan mengukur **TPR@FPR rendah**;
varian Llama-8B mencapai ~94.8% TPR@1%FPR dan 65.3% TPR@0.1%FPR, jauh di atas InjecGuard (20.4%
TPR@1%FPR) dan PromptGuard (~12.8%) [1]. InjecGuard memperkenalkan benchmark **NotInject** (339 sampel
benign bermuatan trigger-word) untuk mengukur *over-defense*, dan menunjukkan guardrail SOTA anjlok ke
~60% akurasi (nyaris tebak-acak) pada input benign karena bergantung pada shortcut kata kunci seperti
"ignore" [2]. BrowseSafe menambahkan bahwa benchmark lazimnya berisi serangan satu-baris sehingga
recall jatuh di kondisi realistis, dan banyak detektor tak mengekspos threshold untuk menyetel FPR [8].
Protokol kami (TPR@1% dan @0.1% FPR + NotInject over-defense) mengikuti arah standar ini, tetapi
diterapkan pada rezim baru: Bahasa Indonesia, campur-kode, dan CPU.

### 2.2. Guardrail ringan untuk deployment (CPU) vs model besar
Spektrum deployment terbelah. Di ujung berat, PromptShield-8B kuat tetapi tidak CPU-friendly [1]; di
ujung ringan, DeBERTa-v3 (ProtectAI/deepset) dan PromptGuard (86M/22M) murah namun over-defend dan
English-centric [2,3]. Detektor DistilBERT dilatih di `deepset/prompt-injections` dilaporkan <10 ms CPU
(ONNX) tetapi diuji pada set kecil [HF]. JailGuard memakai embedding beku + MLP kecil (tanpa
fine-tune), p50 ~14 ms CPU [docs]. **PIDS-Bench** [4] adalah pembanding terdekat kami: ia mengevaluasi
detektor di bawah *over-defense, obfuscation, dan distribution shift* dengan baseline **TF-IDF +
Logistic Regression** dan DistilBERT lintas-seed — setup yang nyaris identik dengan kami, tetapi
berbahasa Inggris dan tanpa sumbu lintas-bahasa maupun remedi augmentasi.

### 2.3. Indirect prompt injection (IPI) & multibahasa
BIPIA (Microsoft, Yi dkk.) membingkai IPI sebagai payload yang tertanam di *tool output* (email,
dokumen, hasil pencarian) lintas 5 skenario & 250 goal; seluruh 25 LLM yang diuji rentan [5]. Deteksi
IPI kemudian dipelajari pada agen email dengan BIPIA sebagai uji-transfer independen + benign
hard-negative untuk menjaga FPR rendah di bawah distribution shift [9] — semangat yang sama dengan
sumbu cross-benchmark kami. Di sisi multibahasa, MLJailDe (contrastive learning) menunjukkan augmentasi
bahasa low-resource menaikkan deteksi jailbreak [6]; codebook semantik lintas-bahasa mem-filter tanpa
terjemahan/fine-tune [7]; SEALGuard meng-adaptasi model multibahasa via LoRA untuk Asia Tenggara [10].
Namun semuanya menyasar **jailbreak umum**, bukan **IPI di tool-output**, dan tak melaporkan metrik IPI
Bahasa Indonesia pada FPR rendah. IndoSafety menyediakan prompt keselamatan ter-lokalisasi (Indonesia
formal/kolokial + Jawa/Sunda/Minang) [11] — tetapi itu **benchmark keselamatan, bukan detektor IPI**.
Tabel 1 merangkum perbandingan; celahnya jelas: tak ada detektor IPI untuk Bahasa Indonesia/campur-kode
di CPU dengan protokol low-FPR + over-defense.

---

## Tabel 1. Perbandingan studi deteksi prompt injection yang relevan

| Study | Pendekatan & model | Bahasa / ancaman | Protokol evaluasi | Deployment | Peran / beda dg studi ini |
|---|---|---|---|---|---|
| PromptShield [1] | Fine-tune Llama-8B & DeBERTa | EN; injeksi (mostly direct) | TPR@1%/0.1% FPR | Berat (8B) | Acuan protokol low-FPR; kami CPU |
| InjecGuard [2] | Guardrail fine-tune + NotInject | EN; direct/indirect | Over-defense (NotInject) | Sedang | Acuan over-defense; kami tambah sumbu ID |
| PromptGuard 2 [3] | Classifier 86M/22M (Meta) | Multibahasa jailbreak | Recall@FPR (AgentDojo) | Ringan (GPU) | Over-defend; kami fokus IPI tool-output ID |
| PIDS-Bench [4] | TF-IDF+LR, DistilBERT, seed | EN; over-def/obfus/shift | TPR, lintas-seed | Ringan | **Pembanding terdekat**; kami tambah lintas-bahasa + remedi MT |
| BIPIA [5] | Benchmark + defense | EN; IPI tool-output | ASR (attack success) | — | Definisi IPI; kami ukur deteksi, bukan ASR |
| MLJailDe [6] | Contrastive, multibahasa | Multibahasa jailbreak | Deteksi jailbreak | Sedang | Jailbreak umum, bukan IPI; kami IPI+ID |
| Codebook semantik [7] | Embed vs codebook EN | Lintas-bahasa jailbreak | Training-free filter | Ringan | Tanpa ID-spesifik; kami ukur TPR@FPR ID |
| IndoSafety [11] | Dataset keselamatan ID | Indonesia + bahasa daerah | Benchmark (bukan detektor) | — | Sumber uji ID masa depan; bukan detektor IPI |
| **Studi ini (Paper D)** | **TF-IDF (leksikal) & e5 (semantik) + augmentasi MT** | **ID + campur-kode; IPI tool-output** | **TPR@1%/0.1%FPR + NotInject + shift EN & ID** | **CPU, <1 ms (TF-IDF)** | **Detektor IPI ID CPU pertama + insight kegagalan komplementer + remedi MT** |

---

## Positioning: di mana kontribusi kita (dan di mana kita kalah — jujur)

**Celah yang kita isi (novelty):**
1. **Bahasa:** Belum ada detektor IPI untuk **Bahasa Indonesia / campur-kode**. IndoSafety [11] hanya
   benchmark keselamatan; detektor multibahasa [6,7,10] menyasar jailbreak umum, bukan IPI tool-output,
   dan tak melaporkan TPR@FPR Indonesia. → ruang kosong yang nyata.
2. **Deployment:** PromptShield-8B [1] kuat tapi bukan CPU; guardrail ringan [2,3] over-defend &
   English-centric. Kami isi niche **CPU, sub-milidetik**.
3. **Insight ilmiah:** kegagalan **komplementer** — leksikal (TF-IDF) tahan cross-benchmark EN tapi
   lemah cross-lingual ID; semantik (e5) sebaliknya — lalu **augmentasi MT (MarianMT en-id)** menutup
   celah leksikal ID (TPR 0.49→0.94) menyamai semantik dengan **86× lebih cepat**. PIDS-Bench [4]
   menguji over-defense/obfuscation/shift EN dengan TF-IDF/DistilBERT, **tapi tanpa sumbu lintas-bahasa
   maupun remedi augmentasi** → sumbu & remedi kami baru.
4. **Protokol:** kami memakai TPR@FPR-rendah + over-defense (NotInject) yang didorong [1,2,8], diterapkan
   ke rezim bahasa/deployment baru.

**Di mana kita kalah / batas (wajib disebut, hindari overclaim):**
- Kami **tidak** mengalahkan TPR Inggris PromptShield-8B [1]; kami menukar ceiling demi CPU + cakupan ID.
- Uji ID kami **sintetis** (MT + campur-kode dari korpus EN publik), bukan IPI Indonesia tulisan-manusia
  → validitas eksternal terbatas; IndoSafety [11] bisa jadi uji ID riil ke depan.
- Belum ada benchmark IPI Indonesia berlabel-manusia → tak bisa head-to-head di ID (caveat
  inkomparabilitas yang sama seperti dipakai IJIES).

**Intinya (fakta):** posisi kita = **detektor IPI Indonesia/campur-kode CPU pertama** dengan temuan
kegagalan-komplementer + remedi MT murah. Pembanding terkuat & paling mirip adalah **PIDS-Bench [4]**
(EN, tanpa lintas-bahasa/remedi) dan **PromptShield [1]** (kuat tapi 8B/EN). Kelemahan jujur: uji ID masih
sintetis dan tanpa benchmark ID manusia.

---

## Referensi (untuk dilengkapi detail penulis saat finalisasi)
- [1] PromptShield: Deployable Detection for Prompt Injection Attacks. arXiv:2501.15145 (2025).
- [2] InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models.
  arXiv:2410.22770 (ACL 2025). (NotInject)
- [3] LlamaFirewall / PromptGuard 2. arXiv:2505.03574 (2025).
- [4] PIDS-Bench: Evaluating Prompt-Injection Detectors Under Over-Defense, Obfuscation, and
  Distribution Shift. arXiv:2609.15017 (2026). [verifikasi tahun/volume]
- [5] Yi et al. Benchmarking and Defending Against Indirect Prompt Injection Attacks on LLMs (BIPIA).
  arXiv:2312.14197 (2023).
- [6] MLJailDe: multilingual jailbreak detection via contrastive learning. [verifikasi sitasi]
- [7] Cross-Lingual Jailbreak Detection via Semantic Codebooks. arXiv:2604.25716 (2026). [verifikasi]
- [8] BrowseSafe: Understanding and Preventing Prompt Injection Within Browsers. arXiv:2511.20597 (2025).
- [9] Prompt Injection Detection for Email Agents Through Attack Chain Modeling. arXiv:2609.30657 (2026).
  [verifikasi]
- [10] SEALGuard: multilingual guardrail for Southeast Asian languages (LoRA). [verifikasi sitasi]
- [11] IndoSafety: Culturally Grounded Safety for LLMs in Indonesian Languages. arXiv:2506.02573 (2025).
- [HF] deepset/prompt-injections; ProtectAI deberta-v3-base-prompt-injection-v2; thameena/
  distilbert-prompt-injection (model cards).
