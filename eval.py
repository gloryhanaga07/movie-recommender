"""
eval.py — LLM-as-a-judge evaluation for the movie recommender.

Runs get_recommendation() against 10 diverse test prompts, then asks
the LLM to score each result on relevance, description quality, and overall.

Usage:
    OLLAMA_API_KEY=your_key python eval.py
"""

import json
import os
import re
import time

import ollama

from llm import get_recommendation, MODEL

# ---------------------------------------------------------------------------
# 10 diverse test cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "label": "Warm & hopeful",
        "preferences": "I want something that makes me cry but also feel hopeful at the end, like a warm hug",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "Dark thriller",
        "preferences": "A psychological thriller that messes with my mind, dark and unpredictable",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "Friday fun",
        "preferences": "Something fun to watch with friends, lots of humor and action, nothing too serious",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "Romantic comedy",
        "preferences": "I love romantic comedies, something light and sweet that gives me butterflies",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "Sci-fi epic",
        "preferences": "A mind-blowing sci-fi movie with big ideas about space, time, or AI",
        "history": ["Inception"],
        "history_ids": [27205],
    },
    {
        "label": "Family animation",
        "preferences": "A great animated movie to watch with my younger siblings, fun for all ages",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "True story",
        "preferences": "A gripping movie based on a true story, something that actually happened",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "Horror",
        "preferences": "I want to be genuinely scared, something creepy and deeply unsettling",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "Superhero action",
        "preferences": "I love superheroes and big action sequences with memorable villains",
        "history": ["Avengers: Infinity War", "The Dark Knight"],
        "history_ids": [299536, 155],
    },
    {
        "label": "Feel-good music",
        "preferences": "A feel-good movie with great music or dancing that lifts my mood",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "Ambiguous — very vague",
        "preferences": "I just want to watch something good tonight",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "Ambiguous — contradictory",
        "preferences": "Something fun but also serious, maybe action or drama or comedy, I don't know",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "Incomplete — short adjectives only",
        "preferences": "good, sexy, lovely",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "Incomplete — casual slang, no punctuation",
        "preferences": "idk something chill maybe",
        "history": [],
        "history_ids": [],
    },
    {
        "label": "Incomplete — emotional fragments only",
        "preferences": "sad. tired.",
        "history": [],
        "history_ids": [],
    },
]

# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are a movie recommendation judge. Evaluate this recommendation objectively.

User preferences: "{preferences}"
Recommended movie tmdb_id: {tmdb_id}
Description shown to user: "{description}"

Score each dimension from 1 to 5:
- relevance          : Does this movie genuinely match what the user asked for?
- description        : Is the pitch compelling, specific, and personalized to their exact words?
- overall            : Would a real person with these preferences want to watch this?
- ambiguity_handling : If the preference was vague or contradictory, did the recommender still make a confident, specific, reasonable pick? (5 = handled well, 1 = ignored the vagueness or gave a generic answer)

Reply with ONLY this JSON:
{{"relevance": <1-5>, "description": <1-5>, "overall": <1-5>, "ambiguity_handling": <1-5>, "reason": "<one sentence explaining your scores>"}}"""


def _judge(preferences: str, tmdb_id: int, description: str) -> dict:
    client = ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
    )
    prompt = JUDGE_PROMPT.format(
        preferences=preferences,
        tmdb_id=tmdb_id,
        description=description,
    )
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    content = response.message.content.strip()
    if not content:
        return {"relevance": 3, "description": 3, "overall": 3, "ambiguity_handling": 3, "reason": "Judge returned empty response."}
    content = re.sub(r"```(?:json)?\s*|\s*```", "", content).strip()
    return json.loads(content)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_eval():
    print("=" * 60)
    print("Movie Recommender — LLM-as-a-Judge Evaluation")
    print("=" * 60)

    results = []

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {case['label']}")
        print(f"  Preferences : {case['preferences'][:70]}")

        # Step 1: get recommendation
        start = time.perf_counter()
        try:
            rec = get_recommendation(
                case["preferences"], case["history"], case["history_ids"]
            )
            elapsed = time.perf_counter() - start
        except Exception as e:
            print(f"  ERROR getting recommendation: {e}")
            continue

        print(f"  Movie       : tmdb_id={rec['tmdb_id']}  ({elapsed:.1f}s)")
        print(f"  Description : {rec['description'][:90]}...")

        # Step 2: judge the result
        try:
            scores = _judge(case["preferences"], rec["tmdb_id"], rec["description"])
            print(f"  Relevance   : {scores['relevance']}/5")
            print(f"  Description : {scores['description']}/5")
            print(f"  Overall     : {scores['overall']}/5")
            print(f"  Ambiguity   : {scores['ambiguity_handling']}/5")
            print(f"  Reason      : {scores['reason']}")
            results.append({**case, **rec, **scores, "elapsed": round(elapsed, 2)})
        except Exception as e:
            print(f"  ERROR in judge: {e}")

    # Summary
    if not results:
        print("\nNo results to summarize.")
        return

    avg = lambda key: sum(r[key] for r in results) / len(results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Tests run       : {len(results)}/{len(TEST_CASES)}")
    print(f"  Avg relevance   : {avg('relevance'):.1f} / 5")
    print(f"  Avg description : {avg('description'):.1f} / 5")
    print(f"  Avg overall     : {avg('overall'):.1f} / 5")
    print(f"  Avg ambiguity   : {avg('ambiguity_handling'):.1f} / 5")
    print(f"  Avg time        : {avg('elapsed'):.1f}s")

    # Save full results to JSON
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved → eval_results.json")


if __name__ == "__main__":
    if not os.environ.get("OLLAMA_API_KEY"):
        print("ERROR: OLLAMA_API_KEY is not set.")
        raise SystemExit(1)
    run_eval()
