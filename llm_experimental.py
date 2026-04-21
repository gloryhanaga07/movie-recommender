"""
Required environment variables:
  OLLAMA_API_KEY — your Ollama Cloud API key (injected by grader at run time)

Approach:
  Two-step hard-coded flow with two enhancements over llm.py:
    1. Python filters all 1000 movies down to ~20 relevant candidates using
       genre boosting, mood/synonym expansion, and weighted random sampling.
       Watch history is also mined to infer the user's taste profile.
    2. A single LLM call picks the best match and writes a personalized description
       using competitive framing — the model knows it's competing against others
       and writes to win.
  History is excluded in Python before the LLM ever sees the candidate list.
"""

import json
import os
import re
import time
import argparse

import ollama
import pandas as pd

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

MODEL = "gemma4:31b-cloud"

DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")

# Load ALL movies — TOP_MOVIES kept for test.py compatibility
ALL_MOVIES = pd.read_csv(DATA_PATH)
TOP_MOVIES = ALL_MOVIES

KNOWN_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
    "Romance", "Science Fiction", "Thriller", "War", "Western",
]

# Mood/synonym expansion — maps common user words to searchable keywords
MOOD_EXPANSION = {
    "chill": ["relaxing", "calm", "light", "gentle", "heartwarming", "comedy", "feel-good"],
    "fun": ["comedy", "adventure", "entertaining", "humor", "lighthearted", "family"],
    "funny": ["comedy", "humor", "hilarious", "witty", "laughs"],
    "scary": ["horror", "terror", "suspense", "fear", "frightening", "creepy"],
    "sad": ["emotional", "tearjerker", "drama", "grief", "moving", "heartbreaking"],
    "happy": ["uplifting", "feel-good", "heartwarming", "comedy", "joyful"],
    "romantic": ["romance", "love", "relationship", "couples", "passion"],
    "dark": ["noir", "gritty", "crime", "thriller", "suspense", "psychological"],
    "mind-blowing": ["twist", "psychological", "mystery", "sci-fi", "complex"],
    "action-packed": ["action", "adventure", "explosive", "fight", "chase"],
    "thought-provoking": ["philosophical", "drama", "dystopia", "political", "social"],
    "classic": ["timeless", "iconic", "acclaimed", "legendary"],
    "feel-good": ["uplifting", "heartwarming", "comedy", "inspiring", "wholesome"],
    "intense": ["thriller", "suspense", "tension", "psychological", "gripping"],
    "light": ["comedy", "family", "animation", "lighthearted", "fun"],
    "epic": ["adventure", "fantasy", "war", "historical", "grand"],
    "cozy": ["heartwarming", "family", "comedy", "gentle", "wholesome"],
    "inspiring": ["biography", "sport", "drama", "triumph", "motivational"],
    "weird": ["surreal", "quirky", "absurd", "cult", "unconventional"],
    "friends": ["friendship", "buddy", "comedy", "group", "together"],
    "date": ["romance", "comedy", "love", "relationship", "charming"],
}

# ---------------------------------------------------------------------------
# Step 1 — Python pre-filter (no LLM call)
# ---------------------------------------------------------------------------

def _filter_candidates(preferences: str, excluded: set, n: int = 20) -> pd.DataFrame:
    """
    Filter ALL_MOVIES down to n relevant candidates using simple text matching.
    No LLM call — pure Python/pandas.
    """
    prefs_lower = preferences.lower()

    # Expand preferences with mood synonyms
    expanded_keywords = set(prefs_lower.split())
    for mood, synonyms in MOOD_EXPANSION.items():
        if mood in prefs_lower:
            expanded_keywords.update(synonyms)

    # Detect explicitly mentioned genres so we can boost them
    mentioned_genres = [g for g in KNOWN_GENRES if g.lower() in prefs_lower]

    # Remove already-watched movies first
    pool = ALL_MOVIES[~ALL_MOVIES["tmdb_id"].isin(excluded)].copy()

    # Score each movie by how many preference words appear in its searchable text
    def score(row) -> float:
        searchable = " ".join([
            str(row.get("genres", "")),
            str(row.get("overview", "")),
            str(row.get("keywords", "")),
            str(row.get("tagline", "")),
            str(row.get("top_cast", "")),
            str(row.get("director", "")),
        ]).lower()
        keyword_score = sum(word in searchable for word in expanded_keywords if len(word) > 3)
        quality_bonus = row.get("vote_average", 0) / 10.0
        # Genre boost: +3 for each explicitly mentioned genre that appears in the movie's genres
        genre_boost = sum(3.0 for g in mentioned_genres if g.lower() in str(row.get("genres", "")).lower())
        return keyword_score + quality_bonus + genre_boost

    pool["_score"] = pool.apply(score, axis=1)

    # Take top 4*n matches and do weighted sampling so lower-ranked movies
    # still get a chance but higher-scoring ones are more likely to appear
    matched = pool[pool["_score"] > 0].sort_values("_score", ascending=False)
    top_pool = matched.head(n * 4)

    if len(top_pool) >= n:
        # Weighted by score so relevance still matters, but more movies get a shot
        weights = top_pool["_score"] + 0.1  # +0.1 avoids zero-weight
        return top_pool.sample(n, weights=weights)

    # Fallback: fill remaining slots with highest-rated movies
    remaining = pool[~pool.index.isin(matched.index)].nlargest(n - len(top_pool), "vote_average")
    combined = pd.concat([top_pool, remaining])
    return combined.sample(min(n, len(combined)))


# ---------------------------------------------------------------------------
# Watch history taste mining (no LLM call)
# ---------------------------------------------------------------------------

def _infer_taste_profile(history_ids: set) -> str:
    """
    Look up the user's watched movies and extract taste signals:
    favourite genres, directors, and tonal patterns.
    Returns a plain-English summary to inject into the prompt.
    """
    if not history_ids:
        return ""

    watched = ALL_MOVIES[ALL_MOVIES["tmdb_id"].isin(history_ids)]
    if watched.empty:
        return ""

    # Collect genres from watched movies
    all_genres: list[str] = []
    for g in watched["genres"].dropna():
        all_genres.extend([x.strip() for x in str(g).split(",")])

    from collections import Counter
    top_genres = [g for g, _ in Counter(all_genres).most_common(3) if g]

    # Collect directors
    directors = watched["director"].dropna().unique().tolist()
    top_directors = directors[:2]

    # Average rating of watched movies — signals quality bar
    avg_rating = watched["vote_average"].mean()
    quality_note = (
        "They tend to watch highly-rated, acclaimed films."
        if avg_rating >= 7.5
        else "They enjoy a broad range of films, not just critic favourites."
    )

    parts = []
    if top_genres:
        parts.append(f"Based on their watch history, this viewer gravitates towards: {', '.join(top_genres)}.")
    if top_directors:
        parts.append(f"They've enjoyed films by {', '.join(top_directors)}.")
    parts.append(quality_note)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Step 2 — Single LLM call to pick + describe
# ---------------------------------------------------------------------------

def _build_prompt(preferences: str, history: list[str], candidates: pd.DataFrame, taste_profile: str) -> str:
    movie_list = "\n".join(
        f'- tmdb_id={r.tmdb_id} | "{r.title}" ({r.year}) | genres: {r.genres} | '
        f'cast: {str(r.top_cast)[:80]} | tagline: {str(r.tagline)} | '
        f'overview: {str(r.overview)[:150]}'
        for r in candidates.itertuples()
    )
    history_text = ", ".join(f'"{t}"' for t in history) if history else "none"
    taste_section = f"\nTaste profile inferred from watch history: {taste_profile}" if taste_profile else ""

    return f"""You are competing against other AI movie recommenders. A real person will choose between your recommendation and a competitor's — your goal is to win by being the most relevant and compelling.

User preferences: "{preferences}"
Already watched (do NOT recommend): {history_text}{taste_section}

Think step by step:
1. What genre, mood, and tone is this user looking for? Factor in their taste profile if available.
2. Which movie from the list best matches ALL of those qualities — including what their history tells you about their taste?
3. If the user expresses an emotional state (e.g. sad, tired, stressed), pick a movie that COMPLEMENTS that mood — something comforting or uplifting — not one that deepens it.
4. Write a description that would make a real person choose your recommendation over a competitor's. Structure it in two parts:
   - First sentence: capture who this person is as a viewer in a vivid, specific way. Be creative — vary your sentence structure. NEVER start with "You're looking for" or "You want". Instead, try approaches like: naming their emotional state directly, describing the experience they crave, or making a bold observation about their taste.
   - Rest: name the movie and explain precisely why it is the right pick for them. Reference specific things about the film — a scene, a feeling, a character — that connect directly to what they asked for.
   Make it feel like a friend who truly gets them wrote it, not a template. Be vivid. Be specific. Win.

Candidate movies:
{movie_list}

Reply with ONLY this JSON (no markdown, no extra text):
{{"tmdb_id": <integer>, "description": "<≤500 chars: first sentence reflects who this viewer is, then name the movie and explain exactly why it's right for them>"}}"""


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict:
    """Return a dict with keys 'tmdb_id' (int) and 'description' (str ≤500 chars)."""

    excluded = set(int(i) for i in history_ids)

    # Step 1: filter in Python — no LLM call
    candidates = _filter_candidates(preferences, excluded)

    # Mine taste profile from watch history — no LLM call
    taste_profile = _infer_taste_profile(excluded)

    # Step 2: single LLM call
    prompt = _build_prompt(preferences, history, candidates, taste_profile)

    client = ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
    )

    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )

    content = re.sub(r"```(?:json)?\s*|\s*```", "", response.message.content.strip()).strip()
    result = json.loads(content)

    # Safety: reject if watched movie slipped through
    if int(result["tmdb_id"]) in excluded:
        fallback = candidates.iloc[0]
        return {
            "tmdb_id": int(fallback.tmdb_id),
            "description": str(fallback.overview)[:500],
        }

    return result


# ---------------------------------------------------------------------------
# CLI for local testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a local movie recommendation test.")
    parser.add_argument("--preferences", type=str, help="User preferences text.")
    parser.add_argument("--history", type=str, help="Comma-separated watch history titles.")
    args = parser.parse_args()

    print("Movie recommender – type your preferences and press Enter.")

    preferences = (
        args.preferences.strip()
        if args.preferences and args.preferences.strip()
        else input("Preferences: ").strip()
    )
    history_raw = (
        args.history.strip()
        if args.history and args.history.strip()
        else input("Watch history (optional): ").strip()
    )
    history = [t.strip() for t in history_raw.split(",") if t.strip()] if history_raw else []

    print("\nThinking...\n")
    start = time.perf_counter()
    result = get_recommendation(preferences, history)
    print(result)
    print(f"\nServed in {time.perf_counter() - start:.2f}s")
