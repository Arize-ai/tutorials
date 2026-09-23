"""Arize remote evaluator using TypeSafe Jev's Noul yes/no question."""
from __future__ import annotations

import math
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

TYPESAFE_API_KEY = os.environ.get("TYPESAFE_API_KEY", "")
TYPESAFE_BASE_URL = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")
QUESTION_NAME = "resolves_request"
QUESTIONS = {
    QUESTION_NAME: {
        "type": "noul",
        "instructions": "Does the response resolve the user's request?",
        "criteria": {
            "true": "The response takes the action the user asked for, or clearly provides the requested solution.",
            "false": "The response does not resolve the request, only acknowledges it, or defers without a solution.",
        },
    }
}
app = FastAPI()


@app.get("/")
async def health():
    return {"ok": True, "jev_key_configured": bool(TYPESAFE_API_KEY)}


@app.post("/v1/evaluate")
async def evaluate(req: Request):
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"error": "Request body must be valid JSON"}, status_code=400)
    if not isinstance(body, dict) or not isinstance(body.get("input"), dict):
        return JSONResponse({"error": "Request must contain an input object"}, status_code=400)
    if not TYPESAFE_API_KEY:
        return JSONResponse({"error": "TYPESAFE_API_KEY is not set on the evaluator server"}, status_code=500)
    state = {key: value for key, value in body["input"].items() if value is not None}
    if not isinstance(state.get("input"), str) or not isinstance(state.get("output"), str):
        return JSONResponse({"error": "input.input and input.output must be strings"}, status_code=400)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            jev = await client.post(
                f"{TYPESAFE_BASE_URL}/v1/systemone",
                headers={"Authorization": f"Bearer {TYPESAFE_API_KEY}"},
                json={"model": JEV_MODEL, "state": state, "questions": QUESTIONS},
            )
        if jev.status_code >= 400:
            return JSONResponse({"error": f"Jev returned {jev.status_code}: {jev.text}"}, status_code=502)
        answer = (jev.json().get("answers") or {}).get(QUESTION_NAME)
        raw_probability = answer.get("noul") if isinstance(answer, dict) else None
        if raw_probability is None:
            return JSONResponse({"error": f"Jev response had no noul answer for {QUESTION_NAME}"}, status_code=502)
        probability = float(raw_probability)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("probability is outside [0, 1]")
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        return JSONResponse({"error": f"Could not read a valid Jev probability: {exc}"}, status_code=502)
    except Exception as exc:
        return JSONResponse({"error": f"Could not parse Jev response: {exc}"}, status_code=502)
    return JSONResponse({"label": "yes" if probability >= 0.5 else "no", "score": probability})
