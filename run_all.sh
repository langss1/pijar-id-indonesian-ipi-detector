#!/usr/bin/env bash
# One-command reproduction of PIJAR-ID (CPU only). See REPRODUCE.md for details.
# Usage:  bash run_all.sh
set -euo pipefail
export PYTHONIOENCODING=utf-8

echo "== [0/5] environment =="
python --version
python - <<'PY'
import sklearn, numpy, pandas, scipy
print("scikit-learn", sklearn.__version__, "| numpy", numpy.__version__, "| pandas", pandas.__version__)
PY

echo "== [1/5] fetch public datasets (HuggingFace) =="
python fetch_data.py

echo "== [2/5] build tool-output-wrapped corpora =="
python expD/src/build_corpus.py

echo "== [3/5] (optional) regenerate ID + hard-negative artifacts =="
echo "   committed artifacts are used by default; uncomment to regenerate:"
echo "   # python expD/src/id_build_mt.py --test_n 250 --aug_n 600"
echo "   # python expD/src/hardneg.py"

echo "== [4/5] reproduce tables =="
python expD/src/run_detectors.py     # Table II
python expD/src/iter2.py             # Table III (base / +aug / PIJAR-ID)
python expD/src/baseline_tf.py       # Table IV (DeBERTa-v3 guard; downloads HF model)
python expD/src/latency.py           # Table IV latency row
python expD/src/analyze.py           # over-defense, McNemar, OOV rates

echo "== [5/5] export serving model =="
python expD/src/export_model.py

echo "DONE. Compare expD/results/*.json against the table in REPRODUCE.md."
