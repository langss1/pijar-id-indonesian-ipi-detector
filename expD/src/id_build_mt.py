"""Build the Indonesian IPI data with a lightweight MarianMT model (Helsinki-NLP/opus-mt-en-id).
CPU, ~300MB, low RAM — replaces the heavy 7B LLM translator (which exhausted RAM).

Outputs:
  expD/data/id_test_raw.csv   id_translate + id_codemix of a balanced en_test sample (RQ2)
  expD/data/id_train_aug.csv  id_translate of a balanced train sample (RQ3 augmentation)
  expD/data/id_spotcheck.csv  40 rows for manual human verification (validity)

Run from project root: python expD/src/id_build_mt.py --test_n 250 --aug_n 600
"""
import os, re, argparse, random, torch, pandas as pd
from transformers import MarianTokenizer, MarianMTModel

MODEL = "Helsinki-NLP/opus-mt-en-id"
OUT = "expD/data"
_tok = _mod = None


def _load():
    global _tok, _mod
    if _tok is None:
        _tok = MarianTokenizer.from_pretrained(MODEL)
        _mod = MarianMTModel.from_pretrained(MODEL)
        _mod.eval()
    return _tok, _mod


def translate_batch(texts, bs=32, max_len=256):
    tok, mod = _load()
    out = []
    for i in range(0, len(texts), bs):
        chunk = [t[:1000] for t in texts[i:i + bs]]
        b = tok(chunk, return_tensors="pt", truncation=True, max_length=max_len, padding=True)
        with torch.no_grad():
            o = mod.generate(**b, max_length=max_len, num_beams=2)
        out.extend(tok.batch_decode(o, skip_special_tokens=True))
        print(f"  translated {min(i+bs,len(texts))}/{len(texts)}")
    return out


SENT = re.compile(r"(?<=[.!?])\s+")


def codemix(en_text, id_text, rng):
    """Sentence-level EN/ID code-mix: keep ~40% of sentences in English."""
    en_s = SENT.split(en_text)
    id_s = SENT.split(id_text)
    n = min(len(en_s), len(id_s))
    if n <= 1:
        # word-level fallback: keep a few English words
        return id_text
    mixed = [en_s[k] if rng.random() < 0.4 else id_s[k] for k in range(n)]
    return " ".join(mixed)


def balanced_sample(df, n, seed=0):
    pos = df[df.label == 1].sample(min(n // 2, int((df.label == 1).sum())), random_state=seed)
    neg = df[df.label == 0].sample(min(n // 2, int((df.label == 0).sum())), random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_n", type=int, default=250)
    ap.add_argument("--aug_n", type=int, default=600)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(0)

    # ---- ID test (RQ2) from en_test ----
    en_test = pd.read_csv(f"{OUT}/en_test.csv")
    samp = balanced_sample(en_test, args.test_n)
    src = samp["text"].astype(str).tolist()
    print(f"[id_test] translating {len(src)} rows ...")
    idt = translate_batch(src)
    rows = []
    for en, idd, lab in zip(src, idt, samp["label"]):
        rows.append({"text": idd, "label": int(lab), "source": "id_translate", "en_source": en})
        rows.append({"text": codemix(en, idd, rng), "label": int(lab), "source": "id_codemix", "en_source": en})
    test_df = pd.DataFrame(rows)
    test_df = test_df[test_df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    test_df.to_csv(f"{OUT}/id_test_raw.csv", index=False)
    test_df.sample(min(40, len(test_df)), random_state=1).to_csv(f"{OUT}/id_spotcheck.csv", index=False)

    # ---- ID train augmentation (RQ3) from train ----
    train = pd.read_csv(f"{OUT}/train.csv")
    aug = balanced_sample(train, args.aug_n)
    print(f"[id_aug] translating {len(aug)} rows ...")
    aug_id = translate_batch(aug["text"].astype(str).tolist())
    aug_df = pd.DataFrame({"text": aug_id, "label": aug["label"].astype(int).values,
                           "source": "id_translate"})
    aug_df = aug_df[aug_df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    aug_df.to_csv(f"{OUT}/id_train_aug.csv", index=False)

    print(f"id_test_raw={len(test_df)} (translate+codemix), id_train_aug={len(aug_df)}")


if __name__ == "__main__":
    main()
