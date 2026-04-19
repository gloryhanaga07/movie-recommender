# Movie Recommender — BAMS 521

A two-step agentic movie recommender that combines Python pre-filtering with a single LLM call to return personalized recommendations in under 10 seconds.

---

## Approach

The system uses a **two-step hard-coded flow** instead of a full agentic loop. This keeps latency low while still leveraging the LLM for the part that matters most: understanding the user's intent and writing a personalized pitch.

### Step 1 — Python pre-filter (no LLM)

`_filter_candidates()` scores all 1,000 movies in `tmdb_top1000_movies.csv` using three signals:

| Signal | Description |
|---|---|
| Keyword score | Count of preference words that appear in genres, overview, keywords, tagline, cast, or director |
| Genre boost | +3 per explicitly named genre (e.g. "horror" → boosts Horror-tagged movies) |
| Quality bonus | `vote_average / 10` to favour well-reviewed films |

The top 40 scored movies are randomly sampled down to 20 candidates. Random sampling prevents the same movie from appearing every time and keeps recommendations diverse across runs.

Movies in the user's watch history are excluded in Python before the LLM ever sees the candidate list — this is a hard guarantee, not just a prompt instruction.

### Step 2 — Single LLM call

`_build_prompt()` sends the 20 candidates to `gemma4:31b-cloud` via Ollama Cloud with:

- **Chain-of-thought instructions** — the model reasons about genre, mood, and tone before picking
- **Rich metadata** — genres, cast, tagline, and overview for each candidate
- **Mood-aware guidance** — if the user expresses an emotional state (sad, tired), the model is instructed to pick something complementary, not something equally draining
- **Forced movie title** — the description must name the film and connect it directly to the user's words

The LLM returns structured JSON: `{"tmdb_id": int, "description": str}`.

---

## Key Design Decisions

**Why Python pre-filter instead of a tool-calling loop?**
An agentic loop with SQL or vector search required 2+ LLM calls minimum, pushing latency to 25–30 seconds. The Python filter runs in milliseconds and gets relevant candidates reliably. The LLM's job is interpretation and writing, not retrieval.

**Why random sampling from the top pool?**
Taking the top N candidates deterministically caused the same movie to be recommended every run for similar inputs. Sampling from the top 2×N preserves quality while adding diversity.

**Why genre boosting?**
Pure keyword matching treated "horror" as any other word. Genre boosting gives a +3 multiplier per explicitly mentioned genre, ensuring Horror-tagged movies surface when the user asks for horror — not adjacent thrillers.

---

## Creative Improvements

Beyond the baseline implementation, four targeted improvements were made to increase recommendation quality:

**1. Explicit genre boosting**
The baseline treated "horror" as just another keyword. The improved filter detects any of 18 known genres mentioned in the user's preferences and applies a +3 score multiplier to movies tagged with that genre. This ensures genre-specific requests surface the right category of films — not thematically adjacent ones.

**2. Candidate diversity via random sampling**
The baseline always returned the top-N highest-scored movies, causing the same film to be recommended repeatedly for similar inputs. The improved system samples randomly from the top 2×N pool. Quality is preserved (only well-matched movies are in the pool) while each run produces a different, valid recommendation.

**3. Mood-aware prompt instruction**
A chain-of-thought step was added instructing the LLM to consider whether the user's emotional state calls for a *complementary* mood rather than a *matching* one. For example, "sad. tired." should surface a comforting film — not one that deepens the feeling. This improved handling of emotional and fragmented inputs.

**4. Expanded evaluation with ambiguity and edge-case testing**
The evaluation suite was extended beyond 10 genre prompts to include 5 additional edge cases: 2 ambiguous inputs (vague, contradictory) and 3 incomplete inputs (short adjectives, casual slang, emotional fragments). The ambiguity handling dimension was added to the judge scoring, and the system achieved a perfect 5.0/5 average across all ambiguity test cases.

---

## Evaluation Strategy

Evaluation uses an **LLM-as-a-judge** approach (`eval.py`). After `get_recommendation()` runs on each test case, the same model scores the output on four dimensions (1–5):

| Dimension | What it measures |
|---|---|
| Relevance | Does the movie genuinely match what the user asked for? |
| Description | Is the pitch compelling, specific, and personalized? |
| Overall | Would a real person with these preferences want to watch this? |
| Ambiguity handling | Did the system make a confident pick despite a vague or contradictory input? |

### Test Cases (15 total)

The test suite covers three categories:

- **Specific genres (10):** Warm & hopeful, Dark thriller, Friday fun, Romantic comedy, Sci-fi epic, Family animation, True story, Horror, Superhero action, Feel-good music
- **Ambiguous inputs (2):** Very vague ("I just want to watch something good tonight"), Contradictory ("fun but also serious, maybe action or drama or comedy")
- **Incomplete inputs (3):** Short adjectives only ("good, sexy, lovely"), Casual slang ("idk something chill maybe"), Emotional fragments ("sad. tired.")

### Eval Results

| Dimension | Score |
|---|---|
| Avg relevance | 4.5 / 5 |
| Avg description | 4.9 / 5 |
| Avg overall | 4.6 / 5 |
| Avg ambiguity handling | 5.0 / 5 |
| Avg response time | 2.8s |

Full results are in `eval_results.json`.

---

## Files

| File | Description |
|---|---|
| `llm.py` | Main implementation — `get_recommendation()` lives here |
| `eval.py` | LLM-as-a-judge evaluation script |
| `eval_results.json` | Output from the latest eval run |
| `test.py` | Provided grader tests |
| `tmdb_top1000_movies.csv` | Movie dataset (1,000 films with metadata) |
| `requirements.txt` | Python dependencies |

---

## Setup & Usage

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run a single recommendation:

```bash
OLLAMA_API_KEY=your_key python llm.py --preferences "I want a psychological thriller" --history "Inception"
```

Run the grader tests:

```bash
OLLAMA_API_KEY=your_key python test.py
```

Run the full evaluation:

```bash
OLLAMA_API_KEY=your_key python eval.py
```
