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

User preferences are also expanded using a **mood/synonym map** before scoring. For example, "chill" expands to ["relaxing", "calm", "gentle", "heartwarming", "comedy"] so vague inputs still find relevant candidates. This runs entirely in Python — no extra LLM call.

The top 80 scored movies are weighted-randomly sampled down to 20 candidates, where higher-scoring movies are more likely to be selected but lower-ranked ones still have a chance. This prevents the same movie from appearing every time while keeping quality high.

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

**Why weighted random sampling from a wider pool?**
Taking the top N candidates deterministically caused the same movie to be recommended every run for similar inputs. The system now samples from the top 4×N using score-weighted probabilities — higher-scoring movies are still favored, but the wider pool means genuinely different movies surface across runs.

**Why genre boosting?**
Pure keyword matching treated "horror" as just another word. Genre boosting gives a +3 multiplier per explicitly mentioned genre, ensuring Horror-tagged movies surface when the user asks for horror — not adjacent thrillers.

**Why mood/synonym expansion?**
Literal keyword matching fails on vague inputs like "something chill" or "scary movie" — those words rarely appear verbatim in movie metadata. A hand-built mood map expands user phrases into searchable terms before scoring, so natural language inputs find genuinely relevant candidates without any extra LLM call.

---

## Creative Improvements

Beyond the baseline implementation, six targeted improvements were made to increase recommendation quality:

**1. Explicit genre boosting**
The baseline treated "horror" as just another keyword. The improved filter detects any of 18 known genres mentioned in the user's preferences and applies a +3 score multiplier to movies tagged with that genre. This ensures genre-specific requests surface the right category of films — not thematically adjacent ones.

**2. Mood/synonym expansion**
Vague inputs like "something chill" or "scary movie" contain words that rarely appear verbatim in movie metadata. A hand-built mood map (22 moods, 100+ synonyms) expands the user's input before scoring — "chill" becomes ["relaxing", "calm", "gentle", "heartwarming", "comedy"]. This runs in pure Python with no extra LLM call, significantly improving candidate quality for natural language inputs.

**3. Candidate diversity via weighted sampling from a wider pool**
The baseline always returned the top-N highest-scored movies, causing the same film to be recommended repeatedly for similar inputs. The improved system draws from the top 4×N pool using score-weighted probabilities — higher-scoring movies are more likely to be selected, but lower-ranked ones still get a chance. This produces genuinely different recommendations across runs without sacrificing relevance.

**4. Mood-aware prompt instruction**
A chain-of-thought step was added instructing the LLM to consider whether the user's emotional state calls for a *complementary* mood rather than a *matching* one. For example, "sad. tired." should surface a comforting film — not one that deepens the feeling. This improved handling of emotional and fragmented inputs.

**5. Personalized two-part description**
The baseline prompt asked for a generic movie pitch. The improved prompt requires a two-part description: the first sentence vividly captures who the viewer is, and the rest names the movie and explains precisely why it fits them. Templates like "You're looking for..." are explicitly banned, forcing the model to vary its opening each time.

**6. Expanded evaluation with ambiguity and edge-case testing**
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
| `app.py` | FastAPI web server — serves the UI and `/recommend` endpoint |
| `static/index.html` | Dark-themed frontend with poster images and watch history |
| `eval.py` | LLM-as-a-judge evaluation script |
| `eval_results.json` | Output from the latest eval run |
| `test.py` | Provided grader tests |
| `tmdb_top1000_movies.csv` | Movie dataset (1,000 films with metadata) |
| `requirements.txt` | Python dependencies |

---

## Web App

A web interface is deployed at Leapcell for anyone to try:

- **App:** https://movie-recommender-gloryhanaga073409-2pynqouj.leapcell.dev
- **Logs:** https://movie-recommender-gloryhanaga073409-2pynqouj.leapcell.dev/logs

Features: movie poster images from TMDB, watch history tag input, animated loading state, and a request log showing all inputs and recommendations.

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
