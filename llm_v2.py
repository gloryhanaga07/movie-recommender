"""
Required environment variables:
  OLLAMA_API_KEY — your Ollama Cloud API key (injected by grader at run time)

Approach:
  Two-step hard-coded flow combining the best of two implementations:
    1. Python filters all movies down to a relevant candidate set using
       genre synonyms, stopword-cleaned keywords, intent rules (7 scenarios),
       emotional complement adjustments, year filtering, and weighted sampling.
       Watch history is also mined to infer the user's taste profile.
    2. A single LLM call picks the best match and writes a personalized,
       compelling description using competitive framing.
  History is excluded in Python before the LLM ever sees the candidate list.
"""

import json
import os
import re
import time
import argparse
from collections import Counter

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

# Maps common user phrases to canonical genre names
GENRE_SYNONYMS = {
    "sci-fi": "Science Fiction",
    "scifi": "Science Fiction",
    "science fiction": "Science Fiction",
    "romcom": "Romance",
    "rom-com": "Romance",
    "kids": "Family",
    "kid": "Family",
    "animated": "Animation",
    "superhero": "Action",
    "superheroes": "Action",
    "spooky": "Horror",
    "scary": "Horror",
    "creepy": "Horror",
    "thrilling": "Thriller",
    "murder": "Crime",
    "true story": "History",
}

# Rule-based intent detection for specific viewing scenarios
INTENT_RULES = [
    {
        "name": "superhero_action",
        "triggers": ["superhero", "superheroes", "marvel", "dc", "villain", "villains"],
        "genre_boosts": {"Action": 4.0, "Adventure": 2.5, "Science Fiction": 1.5},
        "keyword_boosts": ["hero", "villain", "comic", "superhero"],
    },
    {
        "name": "romantic_comedy",
        "triggers": ["romantic comedy", "romcom", "light and sweet", "butterflies", "cute romance"],
        "genre_boosts": {"Romance": 4.0, "Comedy": 3.0},
        "keyword_boosts": ["love", "romance", "relationship", "dating", "sweet"],
    },
    {
        "name": "family_animation",
        "triggers": ["younger siblings", "for all ages", "family", "kids", "children", "animated"],
        "genre_boosts": {"Animation": 4.0, "Family": 4.0, "Comedy": 1.5},
        "keyword_boosts": ["friendship", "family", "adventure", "fun"],
    },
    {
        "name": "true_story",
        "triggers": ["true story", "actually happened", "based on a true story", "real events"],
        "genre_boosts": {"History": 3.0, "Drama": 2.0, "Crime": 1.5},
        "keyword_boosts": ["true", "real", "biography", "based on", "historical"],
    },
    {
        "name": "horror",
        "triggers": ["genuinely scared", "creepy", "deeply unsettling", "horror", "terrifying", "disturbing"],
        "genre_boosts": {"Horror": 5.0, "Thriller": 1.5, "Mystery": 1.0},
        "keyword_boosts": ["haunted", "demon", "evil", "curse", "possession", "monster"],
    },
    {
        "name": "feel_good_music",
        "triggers": ["feel-good", "great music", "dancing", "lifts my mood", "uplifting", "warm hug"],
        "genre_boosts": {"Music": 4.0, "Comedy": 2.5, "Romance": 1.5, "Family": 1.0},
        "keyword_boosts": ["music", "dance", "sing", "joy", "uplifting", "hope"],
    },
    {
        "name": "mind_blowing_scifi",
        "triggers": ["mind-blowing", "big ideas", "space", "time", "ai", "artificial intelligence"],
        "genre_boosts": {"Science Fiction": 5.0, "Mystery": 2.0, "Thriller": 1.0},
        "keyword_boosts": ["space", "time", "future", "robot", "ai", "artificial intelligence", "alien"],
        "min_vote": 6.8,
    },
]

# Python-level emotional adjustments — applied before LLM sees candidates
EMOTIONAL_COMPLEMENT_RULES = [
    {
        "triggers": ["sad", "tired", "stressed", "drained", "exhausted", "down"],
        "prefer_genres": {"Comedy": 2.0, "Family": 2.0, "Music": 1.5, "Romance": 1.0},
        "avoid_genres": {"Horror": -2.0, "War": -1.5},
    }
]

STOPWORDS = {
    "i", "me", "my", "we", "our", "you", "your", "a", "an", "the", "and", "or", "but",
    "to", "of", "for", "with", "that", "this", "it", "is", "are", "be", "am", "was",
    "were", "something", "movie", "film", "watch", "want", "like", "love", "maybe",
    "just", "really", "good", "tonight", "nothing", "too", "very"
}

CURRENT_YEAR = pd.Timestamp.now().year

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _extract_preference_terms(preferences: str) -> list[str]:
    """Tokenize preferences, removing stopwords and short tokens."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-']+", preferences.lower())
    cleaned = []
    for token in tokens:
        token = token.strip("-'")
        if len(token) <= 2 or token in STOPWORDS:
            continue
        cleaned.append(token)
    return cleaned


def _detect_genres(preferences: str) -> set[str]:
    """Detect explicitly mentioned genres, including synonyms."""
    prefs_lower = _normalize_text(preferences)
    detected = set()
    for genre in KNOWN_GENRES:
        if genre.lower() in prefs_lower:
            detected.add(genre)
    for phrase, mapped_genre in GENRE_SYNONYMS.items():
        if phrase in prefs_lower:
            detected.add(mapped_genre)
    return detected


def _active_intent_rules(preferences: str) -> list[dict]:
    prefs_lower = _normalize_text(preferences)
    return [r for r in INTENT_RULES if any(t in prefs_lower for t in r["triggers"])]


def _active_emotion_rules(preferences: str) -> list[dict]:
    prefs_lower = _normalize_text(preferences)
    return [r for r in EMOTIONAL_COMPLEMENT_RULES if any(t in prefs_lower for t in r["triggers"])]


def _build_searchable_text(row) -> str:
    return _normalize_text(" ".join([
        str(row.get("title", "")),
        str(row.get("genres", "")),
        str(row.get("overview", "")),
        str(row.get("keywords", "")),
        str(row.get("tagline", "")),
        str(row.get("top_cast", "")),
        str(row.get("director", "")),
    ]))


# ---------------------------------------------------------------------------
# Watch history taste mining (no LLM call)
# ---------------------------------------------------------------------------

def _infer_taste_profile(history_ids: set) -> str:
    """
    Look up the user's watched movies and extract taste signals:
    favourite genres, directors, and quality bar.
    Returns a plain-English summary injected into the prompt.
    """
    if not history_ids:
        return ""

    watched = ALL_MOVIES[ALL_MOVIES["tmdb_id"].isin(history_ids)]
    if watched.empty:
        return ""

    all_genres: list[str] = []
    for g in watched["genres"].dropna():
        all_genres.extend([x.strip() for x in str(g).split(",")])

    top_genres = [g for g, _ in Counter(all_genres).most_common(3) if g]
    top_directors = watched["director"].dropna().unique().tolist()[:2]

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
# Step 1 — Python pre-filter (no LLM call)
# ---------------------------------------------------------------------------

def _filter_candidates(preferences: str, excluded: set, n: int = 20) -> pd.DataFrame:
    """
    Filter ALL_MOVIES down to n relevant candidates using:
    - Stopword-cleaned keyword matching
    - Genre synonym detection + explicit genre boosting
    - 7 intent rules for specific viewing scenarios
    - Emotional complement adjustments
    - Year filtering (no unreleased movies)
    - Semi-stable sampling: top 8 fixed + random diversity slice
    """
    prefs_lower = _normalize_text(preferences)
    preference_terms = _extract_preference_terms(preferences)
    detected_genres = _detect_genres(preferences)
    intent_rules = _active_intent_rules(preferences)
    emotion_rules = _active_emotion_rules(preferences)

    # Remove watched + unreleased movies
    pool = ALL_MOVIES[~ALL_MOVIES["tmdb_id"].isin(excluded)].copy()
    pool["year_num"] = pd.to_numeric(pool["year"], errors="coerce")
    pool = pool[pool["year_num"].notna() & (pool["year_num"] <= CURRENT_YEAR)].copy()

    unreleased_markers = ["coming soon", "upcoming", "to be announced", "tba"]
    combined_text = (
        pool["title"].fillna("").astype(str).str.lower() + " "
        + pool["tagline"].fillna("").astype(str).str.lower() + " "
        + pool["overview"].fillna("").astype(str).str.lower()
    )
    for marker in unreleased_markers:
        pool = pool[~combined_text.str.contains(marker, regex=False)].copy()

    def score(row) -> float:
        searchable = _build_searchable_text(row)
        genres_text = _normalize_text(row.get("genres", ""))
        vote_average = float(row.get("vote_average", 0) or 0)
        vote_count = float(row.get("vote_count", 0) or 0)

        keyword_score = sum(1.0 for term in preference_terms if term in searchable)

        phrase_bonus = 2.0 if (
            ("feel-good" in prefs_lower or "feel good" in prefs_lower)
            and any(t in searchable for t in ["feel good", "joy", "uplifting", "warm", "hope"])
        ) else 0.0

        explicit_genre_boost = sum(
            3.5 for genre in detected_genres if genre.lower() in genres_text
        )

        intent_boost = 0.0
        for rule in intent_rules:
            for genre, boost in rule.get("genre_boosts", {}).items():
                if genre.lower() in genres_text:
                    intent_boost += boost
            for kw in rule.get("keyword_boosts", []):
                if kw in searchable:
                    intent_boost += 0.8
            min_vote = rule.get("min_vote")
            if min_vote is not None:
                intent_boost += 0.8 if vote_average >= min_vote else -1.0

        emotion_adjustment = 0.0
        for rule in emotion_rules:
            for genre, boost in rule.get("prefer_genres", {}).items():
                if genre.lower() in genres_text:
                    emotion_adjustment += boost
            for genre, penalty in rule.get("avoid_genres", {}).items():
                if genre.lower() in genres_text:
                    emotion_adjustment += penalty

        quality_bonus = vote_average / 10.0
        popularity_bonus = min(vote_count / 5000.0, 1.2)

        return (
            keyword_score + phrase_bonus + explicit_genre_boost
            + intent_boost + emotion_adjustment
            + quality_bonus + popularity_bonus
        )

    pool["_score"] = pool.apply(score, axis=1)
    matched = pool.sort_values(["_score", "vote_average"], ascending=[False, False])

    # Top 8 fixed for stability + random slice for diversity
    top_fixed_count = min(8, len(matched))
    fixed_top = matched.head(top_fixed_count)
    remaining_pool = matched.iloc[top_fixed_count: max(top_fixed_count, n * 3)]

    if len(remaining_pool) > 0:
        random_needed = min(max(n - len(fixed_top), 0), len(remaining_pool))
        candidates = pd.concat([fixed_top, remaining_pool.sample(random_needed)])
    else:
        candidates = fixed_top

    if len(candidates) < n:
        already = set(candidates.index)
        filler = matched[~matched.index.isin(already)].head(n - len(candidates))
        candidates = pd.concat([candidates, filler])

    return (
        candidates.drop_duplicates(subset=["tmdb_id"])
        .head(n)
        .sort_values(["_score", "vote_average"], ascending=[False, False])
        .copy()
    )


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
1. Identify the user's genre, mood, tone, and viewing context. Factor in their taste profile if available.
2. Pick ONLY from the candidate movies below — do not invent a movie or tmdb_id.
3. If the user sounds sad, tired, stressed, or emotionally drained, prefer something comforting or uplifting — not something that deepens the mood.
4. Write a description that would make a real person choose your recommendation over a competitor's. Structure it in two parts:
   - First sentence: capture who this person is as a viewer in a vivid, specific way. NEVER start with "You're looking for" or "You want". Try naming their emotional state, describing the experience they crave, or making a bold observation about their taste.
   - Rest: name the movie and explain precisely why it is the right pick for them. Reference something specific — a feeling, a character, a scene — that connects directly to what they asked for.
   Make it feel like a friend who truly gets them wrote it, not a template. Be vivid. Be specific. Win.
5. Keep the description under 500 characters.

Candidate movies:
{movie_list}

Reply with ONLY this JSON (no markdown, no extra text):
{{"tmdb_id": <integer from the candidate list>, "description": "<≤500 chars>"}}"""


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict:
    """Return a dict with keys 'tmdb_id' (int) and 'description' (str ≤500 chars)."""

    excluded = set(int(i) for i in history_ids)

    # Step 1: Python filtering — no LLM call
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

    # Safety: validate tmdb_id is a real candidate and not already watched
    valid_ids = set(candidates["tmdb_id"].astype(int))
    try:
        chosen_id = int(result["tmdb_id"])
    except Exception:
        chosen_id = None

    if chosen_id not in valid_ids or chosen_id in excluded:
        fallback = candidates.iloc[0]
        fallback_desc = str(fallback.get("overview", ""))[:500].strip()
        if not fallback_desc:
            fallback_desc = f'{fallback["title"]} is a strong fit for your preferences based on genre, mood, and tone.'
        return {"tmdb_id": int(fallback.tmdb_id), "description": fallback_desc[:500]}

    result["tmdb_id"] = chosen_id
    result["description"] = str(result.get("description", "")).strip()[:500]

    if not result["description"]:
        fallback = candidates.iloc[0]
        fallback_desc = str(fallback.get("overview", ""))[:500].strip() or f'{fallback["title"]} is a strong fit for your preferences.'
        return {"tmdb_id": int(fallback.tmdb_id), "description": fallback_desc[:500]}

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
