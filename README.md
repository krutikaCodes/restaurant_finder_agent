# HW3 – Restaurant Recommendation Agent

AI agent that answers:

> "Find a Turkish restaurant in Downtown Baltimore, MD for two people to have dinner under $65 on
> Thursday night at 7:30 pm with a table for two near a window with a view of the garden or the street."

## Files

| File | What it is |
|---|---|
| `agent.py` | The whole agent (about 200 lines) |
| `restaurants.csv` | Small restaurant dataset (simulated, fictional names) |
| `requirements.txt` | Python packages to install |
| `README.md` | This file: setup, run, and architecture |

## How to build and run

```bash
pip install -r requirements.txt

# Optional: use Gemini for understanding the query (get a key at https://aistudio.google.com/apikey)
export GEMINI_API_KEY=your-key-here        # Windows PowerShell:  $env:GEMINI_API_KEY="your-key-here"

python agent.py
```

Without a key, the agent still works using its rule-based parser. The output is printed to the
screen, and a chart is saved as `results.png`.

## Agent architecture

The agent is **goal-based** (it plans actions to reach a goal) and **utility-based** (it ranks
restaurants by a score). It follows a simple loop: **perceive → plan → act**.

```
User query → [1. NLP: Gemini or rules] → [2. Plan with A*] → [3. Load CSV]
          → [4. Check hard constraints] → [5. Rank by utility] → [6. Explain + chart]
```

### 1. States, actions and goal (Deterministic Planning, Ch 6)

A **state** is a set of facts that are true, e.g. `{have_query, gemini_available}`.

| Action | Preconditions | Effect | Cost |
|---|---|---|---|
| parse_with_gemini | have_query, gemini_available | query_understood | 1 |
| parse_with_rules | have_query | query_understood | 3 |
| load_restaurants | query_understood | data_loaded | 1 |
| check_constraints | data_loaded | filtered | 1 |
| rank_results | filtered | ranked | 1 |
| explain_results | ranked | explained | 1 |

**Goal:** `{query_understood, data_loaded, filtered, ranked, explained}`

### 2. Searching for a plan (Searching for Solutions, Ch 3)

The agent does not follow a fixed script. It uses **A\* search** to find the cheapest sequence of
actions from the start state to the goal:

* `g` = cost of the actions so far
* `h` = number of goal facts still missing (never overestimates, so A\* finds the cheapest plan)
* `f = g + h` decides which partial plan to expand next

If a Gemini key is available, A\* picks `parse_with_gemini` (total cost 5). If not, it picks
`parse_with_rules` (total cost 7). If Gemini fails while running, the agent falls back to rules.

### 3. Reasoning with constraints (Ch 4)

**Hard constraints** (a restaurant must pass all of them):

1. cuisine = Turkish
2. neighborhood is in Downtown Baltimore (Downtown, Inner Harbor, Mount Vernon, Harbor East)
3. cost for two ≤ $65
4. a table is free on Thursday at 19:30
5. window table with a garden or street view

Each restaurant is tested against each constraint, and the log shows which ones it fails.

**Soft constraints** (preferences) become a **utility score** out of 100:

```
utility = 100 × (0.4 × rating + 0.2 × savings + 0.2 × view + 0.2 × closeness)
```

* rating: 3 stars → 0, 5 stars → 1
* savings: how much of the budget is left
* view: garden (named first) = 1.0, street = 0.8
* closeness: closer to the city center scores higher

## Expected output

| Rank | Restaurant | Price for 2 | View | Utility |
|---|---|---|---|---|
| 1 | Anatolia Grill | $58 | garden | 71.2 |
| 2 | Istanbul Corner Cafe | $46 | street | 59.8 |
| 3 | Aegean Table | $63 | garden | 56.6 |
| 4 | Bosphorus Kitchen | $62 | street | 54.9 |

The other 9 restaurants are rejected, and the log says which constraint each one failed.

## Note on the data

`restaurants.csv` is a **simulated dataset** with fictional restaurants. A custom CSV is used
because public APIs like Yelp do not provide window views or table availability, which this
scenario needs.
