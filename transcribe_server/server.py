#!/usr/bin/env python3
"""
transcribe_server/server.py

Parakeet transcription microservice.

Loads nvidia/parakeet-tdt-1.1b on startup and serves HTTP transcription
requests. Runs inside the official NeMo Docker container so no Python
version constraints apply.

── Docker usage ────────────────────────────────────────────────────────────────

    docker run \
      --gpus all \
      --rm \
      --shm-size=16g \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      -v $(pwd)/transcribe_server:/transcribe_server:ro \
      -p 8765:8765 \
      nvcr.io/nvidia/nemo:25.11.01 \
      python /transcribe_server/server.py

── API ─────────────────────────────────────────────────────────────────────────

    GET  /health
        Returns: {"status": "ok", "model": "nvidia/parakeet-tdt-1.1b"}

    POST /transcribe
        Body: multipart/form-data with field "file" (WAV audio, 16 kHz mono)
        Returns: {"transcript": "..."} or {"error": "..."}

── No extra Python packages required ───────────────────────────────────────────
Uses only stdlib + NeMo (already in the container).
"""

import json
import os
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.getenv("PARAKEET_PORT", "8765"))
MODEL_NAME = "nvidia/parakeet-tdt-1.1b"

# ── Load model at startup ────────────────────────────────────────────────────

print(f"Loading {MODEL_NAME} …", flush=True)
try:
    import nemo.collections.asr as nemo_asr
    _model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
    _model.eval()
    print(f"Model ready — listening on :{PORT}", flush=True)
except Exception as exc:
    print(f"FATAL: could not load model: {exc}", flush=True)
    sys.exit(1)


# ── HTTP handler ─────────────────────────────────────────────────────────────

def _json_response(handler: "Handler", body: dict, status: int = 200) -> None:
    data = json.dumps(body).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _parse_multipart(handler: "Handler") -> bytes | None:
    """
    Parse a multipart/form-data request and return the bytes of the first
    'file' field, or None if not found / wrong content-type.
    Uses stdlib cgi module (available in all CPython builds inside NeMo).
    """
    import cgi  # noqa: PLC0415  (inside function to keep top-level clean)

    ctype, pdict = cgi.parse_header(handler.headers.get("Content-Type", ""))
    if ctype != "multipart/form-data":
        return None

    pdict["boundary"] = pdict["boundary"].encode()
    pdict["CONTENT-LENGTH"] = int(handler.headers.get("Content-Length", 0))
    fields = cgi.parse_multipart(handler.rfile, pdict)
    raw = fields.get("file", [None])[0]
    if raw is None:
        return None
    return raw if isinstance(raw, bytes) else raw.encode()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            _json_response(self, {"status": "ok", "model": MODEL_NAME})
        else:
            _json_response(self, {"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/transcribe":
            _json_response(self, {"error": "not found"}, 404)
            return

        audio_bytes = _parse_multipart(self)
        if not audio_bytes:
            _json_response(self, {"error": "send multipart/form-data with a 'file' field"}, 400)
            return

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            results = _model.transcribe([tmp_path])
            text = results[0] if results else ""
            if hasattr(text, "text"):
                text = text.text
            _json_response(self, {"transcript": str(text).strip()})

        except Exception as exc:
            _json_response(self, {"error": str(exc)}, 500)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def log_message(self, fmt, *args):  # noqa: N802
        print(f"[{self.address_string()}] {fmt % args}", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    httpd = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Parakeet server up on 0.0.0.0:{PORT}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
