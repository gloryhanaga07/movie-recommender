"""
app.py — FastAPI web server for the movie recommender.

Usage:
    OLLAMA_API_KEY=your_key uvicorn app:app --reload
"""

import os
import json
import traceback
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llm import get_recommendation, ALL_MOVIES

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

LOG_FILE = "/tmp/logs.json"

def append_log(entry: dict):
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            logs = json.load(f)
    logs.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)


class RecommendRequest(BaseModel):
    preferences: str
    history: list[str] = []
    history_ids: list[int] = []


@app.get("/", response_class=HTMLResponse)
def root():
    with open("static/index.html") as f:
        return f.read()


@app.get("/health")
def health():
    key_set = bool(os.environ.get("OLLAMA_API_KEY"))
    return {"status": "ok", "ollama_api_key_set": key_set}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        result = get_recommendation(req.preferences, req.history, req.history_ids)
    except KeyError as e:
        return JSONResponse(status_code=500, content={"error": f"Missing environment variable: {e}. Please set OLLAMA_API_KEY in your deployment settings."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "detail": traceback.format_exc()})

    row = ALL_MOVIES[ALL_MOVIES["tmdb_id"] == result["tmdb_id"]]
    if row.empty:
        return result

    movie = row.iloc[0]
    poster_path = str(movie.get("poster_path", ""))
    poster_url = (
        f"https://image.tmdb.org/t/p/w500{poster_path}"
        if poster_path and poster_path != "nan"
        else None
    )

    response = {
        "tmdb_id": int(result["tmdb_id"]),
        "description": result["description"],
        "title": str(movie["title"]),
        "year": int(movie["year"]),
        "genres": str(movie["genres"]),
        "vote_average": float(movie["vote_average"]),
        "tagline": str(movie.get("tagline", "")),
        "poster_url": poster_url,
    }

    append_log({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "preferences": req.preferences,
        "history": req.history,
        "recommended": response["title"],
        "tmdb_id": response["tmdb_id"],
        "description": response["description"],
    })

    return response


@app.get("/logs", response_class=HTMLResponse)
def view_logs():
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            logs = json.load(f)

    rows = ""
    for entry in reversed(logs):
        history = ", ".join(entry.get("history", [])) or "—"
        rows += f"""
        <tr>
          <td>{entry['time']}</td>
          <td>{entry['preferences']}</td>
          <td>{history}</td>
          <td><strong>{entry['recommended']}</strong></td>
          <td style="color:#aaa;font-size:12px">{entry['description']}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <title>Recommendation Logs</title>
  <style>
    body {{ font-family: sans-serif; background: #0f0f13; color: #f0f0f0; padding: 32px; }}
    h1 {{ margin-bottom: 24px; font-size: 20px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ text-align: left; padding: 10px; background: #1a1a24; color: #888; font-weight: 600; }}
    td {{ padding: 10px; border-bottom: 1px solid #2a2a38; vertical-align: top; max-width: 300px; }}
    tr:hover td {{ background: #1a1a24; }}
    .count {{ color: #888; font-size: 13px; margin-bottom: 16px; }}
  </style>
</head>
<body>
  <h1>Recommendation Logs</h1>
  <div class="count">{len(logs)} request{"s" if len(logs) != 1 else ""} total</div>
  <table>
    <thead>
      <tr>
        <th>Time</th><th>Preferences</th><th>History</th><th>Recommended</th><th>Description</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>"""
