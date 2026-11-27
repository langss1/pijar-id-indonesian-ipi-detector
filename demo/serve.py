"""Minimal localhost backend for the PIJAR-ID demo (BACKEND lane).

Exposes the `ScanResult` contract over HTTP so the frontend (Antigravity/Gemini lane)
can render an interactive injection scanner. CPU only, localhost only.

    pip install fastapi uvicorn joblib scikit-learn
    python demo/serve.py            # -> http://127.0.0.1:8000  (docs at /docs)

Endpoints:
    GET  /health            -> {"status":"ok","model":...}
    POST /scan  {"text": ...}  -> ScanResult  (see demo/predict.py)

Security: binds to 127.0.0.1 only. No auth, no external calls, no persistence.
CORS is open to localhost dev origins so the frontend dev server can call it.
"""
import os
from predict import load, scan

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    raise SystemExit("Install backend deps: pip install fastapi uvicorn")

app = FastAPI(title="PIJAR-ID Injection Scanner", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"], allow_headers=["*"],
)

_MODEL = load()


class ScanIn(BaseModel):
    text: str
    top_k: int = 8


@app.get("/health")
def health():
    return {"status": "ok", "model": _MODEL["meta"]}


@app.post("/scan")
def scan_endpoint(body: ScanIn):
    return scan(body.text, _MODEL, top_k=body.top_k)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
