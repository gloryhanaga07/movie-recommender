"""
ab_test.py — Round-robin A/B evaluation of three llm versions.

Versions compared:
  V1: llm_v1.py — original baseline
  V2: llm_v2.py — merged (best filtering + best prompting)
  V3: llm_v3.py — experimental (competitive framing + taste mining)

Judge: Google Gemini (gemini-1.5-flash) via GEMINI_API_KEY
Inputs: ab_test_inputs.csv

Usage:
  python ab_test.py
"""

import json
import os
import random
import time
from datetime import datetime

import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

judge_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
JUDGE_MODEL = "gemini-2.5-flash"

VERSIONS = {
    "V1 (llm_v1)": "llm_v1",
    "V2 (llm_v2)": "llm_v2",
    "V3 (llm_v3)": "llm_v3",
}

PAIRS = [
    ("V1 (llm_v1)", "V2 (llm_v2)"),
    ("V1 (llm_v1)", "V3 (llm_v3)"),
    ("V2 (llm_v2)", "V3 (llm_v3)"),
]

# ---------------------------------------------------------------------------
# Load recommenders lazily
# ---------------------------------------------------------------------------

_recommenders = {}

def get_recommender(version_key: str):
    if version_key not in _recommenders:
        module_name = VERSIONS[version_key]
        import importlib
        mod = importlib.import_module(module_name)
        _recommenders[version_key] = mod.get_recommendation
    return _recommenders[version_key]


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are an impartial judge evaluating two AI movie recommendation systems.

A user submitted this request:
"{preferences}"

Two AI agents each recommended a movie:

AGENT {a_label}:
Movie: {a_title}
Description: {a_desc}

AGENT {b_label}:
Movie: {b_title}
Description: {b_desc}

Score each agent on these dimensions (1-5):
1. RELEVANCE: Does the movie genuinely match what the user asked for in terms of genre, mood, and tone?
2. DESCRIPTION: Is the description compelling, specific, and personalized — not generic?
3. PERSUASIVENESS: If you were this user, how likely would you be to actually watch this movie based on the description?

Reply with ONLY this JSON (no markdown):
{{
  "agent_{a_key}": {{"relevance": <1-5>, "description": <1-5>, "persuasiveness": <1-5>}},
  "agent_{b_key}": {{"relevance": <1-5>, "description": <1-5>, "persuasiveness": <1-5>}},
  "winner": "<{a_key} or {b_key}>",
  "reason": "<1-2 sentences explaining why the winner beat the other>"
}}"""


def judge(preferences: str, a_label: str, a_title: str, a_desc: str,
          b_label: str, b_title: str, b_desc: str) -> dict:
    """Ask Gemini to judge two recommendations head-to-head."""
    a_key = a_label.replace(" ", "_")
    b_key = b_label.replace(" ", "_")

    prompt = JUDGE_PROMPT.format(
        preferences=preferences,
        a_label=a_label, a_title=a_title, a_desc=a_desc,
        b_label=b_label, b_title=b_title, b_desc=b_desc,
        a_key=a_key, b_key=b_key,
    )

    try:
        response = judge_client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"    ⚠️  Judge error: {e}")
        return None


# ---------------------------------------------------------------------------
# Run one test case for one pair
# ---------------------------------------------------------------------------

def run_pair(preferences: str, history: list, history_ids: list,
             ver_a: str, ver_b: str, all_movies: pd.DataFrame) -> dict:
    """
    Run both versions, judge twice (swapping A/B), return averaged result.
    """
    rec_a = get_recommender(ver_a)(preferences, history, history_ids)
    rec_b = get_recommender(ver_b)(preferences, history, history_ids)

    def get_title(rec):
        row = all_movies[all_movies["tmdb_id"] == rec["tmdb_id"]]
        return str(row.iloc[0]["title"]) if not row.empty else f"tmdb_id={rec['tmdb_id']}"

    title_a = get_title(rec_a)
    title_b = get_title(rec_b)
    desc_a = rec_a["description"]
    desc_b = rec_b["description"]

    # Judge round 1: A vs B
    time.sleep(1)  # avoid rate limit
    result1 = judge(preferences, "A", title_a, desc_a, "B", title_b, desc_b)

    # Judge round 2: swap B vs A
    time.sleep(1)
    result2 = judge(preferences, "A", title_b, desc_b, "B", title_a, desc_a)

    if result1 is None or result2 is None:
        return None

    # Extract scores for ver_a and ver_b from both rounds
    def extract_scores(result, a_is_vera: bool):
        a_key = "agent_A"
        b_key = "agent_B"
        if a_is_vera:
            vera_scores = result.get(a_key, {})
            verb_scores = result.get(b_key, {})
            winner_raw = result.get("winner", "")
            vera_wins = winner_raw == "A"
        else:
            vera_scores = result.get(b_key, {})
            verb_scores = result.get(a_key, {})
            winner_raw = result.get("winner", "")
            vera_wins = winner_raw == "B"
        return vera_scores, verb_scores, vera_wins, result.get("reason", "")

    s1_a, s1_b, w1, r1 = extract_scores(result1, a_is_vera=True)
    s2_a, s2_b, w2, r2 = extract_scores(result2, a_is_vera=False)

    def avg_scores(s1, s2):
        keys = ["relevance", "description", "persuasiveness"]
        return {k: round((s1.get(k, 3) + s2.get(k, 3)) / 2, 1) for k in keys}

    scores_a = avg_scores(s1_a, s2_a)
    scores_b = avg_scores(s1_b, s2_b)
    scores_a["total"] = sum(scores_a.values())
    scores_b["total"] = sum(scores_b.values())

    winner = ver_a if (w1 + w2) >= 1 else ver_b  # majority vote

    return {
        "version_a": ver_a,
        "version_b": ver_b,
        "title_a": title_a,
        "title_b": title_b,
        "desc_a": desc_a,
        "desc_b": desc_b,
        "scores_a": scores_a,
        "scores_b": scores_b,
        "winner": winner,
        "reasons": [r1, r2],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  A/B TEST — Movie Recommender")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    inputs = pd.read_csv("ab_test_inputs.csv")

    import llm_v1
    all_movies = llm_v1.ALL_MOVIES

    results = []
    win_counts = {v: 0 for v in VERSIONS}
    total_scores = {v: {"relevance": 0, "description": 0, "persuasiveness": 0, "n": 0} for v in VERSIONS}

    for _, row in inputs.iterrows():
        preferences = str(row["preferences"])
        history_raw = str(row.get("history", "") or "")
        history_ids_raw = str(row.get("history_ids", "") or "")

        # pandas reads empty CSV cells as NaN → str gives "nan", handle it
        if history_raw.strip().lower() == "nan":
            history_raw = ""
        if history_ids_raw.strip().lower() == "nan":
            history_ids_raw = ""

        history = [h.strip() for h in history_raw.split(",") if h.strip()] if history_raw else []
        history_ids = [int(i.strip()) for i in history_ids_raw.split(",") if i.strip()] if history_ids_raw else []

        print(f"\n[Test {int(row['id'])}] {row['category']}")
        print(f"  Preferences: {preferences[:80]}...")

        test_result = {
            "id": int(row["id"]),
            "category": row["category"],
            "preferences": preferences,
            "pairs": [],
        }

        for ver_a, ver_b in PAIRS:
            print(f"  ⚔️  {ver_a} vs {ver_b}", end=" ", flush=True)
            pair_result = run_pair(preferences, history, history_ids,
                                   ver_a, ver_b, all_movies)
            if pair_result is None:
                print("→ skipped (judge error)")
                continue

            winner = pair_result["winner"]
            win_counts[winner] += 1

            for ver, scores in [(ver_a, pair_result["scores_a"]), (ver_b, pair_result["scores_b"])]:
                for dim in ["relevance", "description", "persuasiveness"]:
                    total_scores[ver][dim] += scores[dim]
                total_scores[ver]["n"] += 1

            print(f"→ 🏆 {winner}")
            print(f"     {ver_a}: rel={pair_result['scores_a']['relevance']} desc={pair_result['scores_a']['description']} pers={pair_result['scores_a']['persuasiveness']} total={pair_result['scores_a']['total']}")
            print(f"     {ver_b}: rel={pair_result['scores_b']['relevance']} desc={pair_result['scores_b']['description']} pers={pair_result['scores_b']['persuasiveness']} total={pair_result['scores_b']['total']}")

            test_result["pairs"].append(pair_result)

        results.append(test_result)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    total_matchups = sum(win_counts.values())

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)

    ranked = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (ver, wins) in enumerate(ranked):
        n = total_scores[ver]["n"]
        if n > 0:
            avg_rel  = round(total_scores[ver]["relevance"] / n, 2)
            avg_desc = round(total_scores[ver]["description"] / n, 2)
            avg_pers = round(total_scores[ver]["persuasiveness"] / n, 2)
            avg_tot  = round((avg_rel + avg_desc + avg_pers), 2)
        else:
            avg_rel = avg_desc = avg_pers = avg_tot = 0

        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
        print(f"\n{medal} {ver}")
        print(f"   Wins         : {wins}/{total_matchups}")
        print(f"   Avg relevance     : {avg_rel}/5")
        print(f"   Avg description   : {avg_desc}/5")
        print(f"   Avg persuasiveness: {avg_pers}/5")
        print(f"   Avg total         : {avg_tot}/15")

    print("\n" + "=" * 60)
    overall_winner = ranked[0][0]
    print(f"  🏆 OVERALL WINNER: {overall_winner}")
    print("=" * 60)

    # Save full results
    output = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_winner": overall_winner,
        "win_counts": win_counts,
        "avg_scores": {
            ver: {
                "relevance": round(total_scores[ver]["relevance"] / max(total_scores[ver]["n"], 1), 2),
                "description": round(total_scores[ver]["description"] / max(total_scores[ver]["n"], 1), 2),
                "persuasiveness": round(total_scores[ver]["persuasiveness"] / max(total_scores[ver]["n"], 1), 2),
            }
            for ver in VERSIONS
        },
        "test_cases": results,
    }

    with open("ab_test_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Full results saved to ab_test_results.json")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
