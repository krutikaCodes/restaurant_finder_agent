"""
SENG 691 HW3 - Restaurant Recommendation Agent (Deterministic Planning)

Pipeline:  understand query (Gemini or rules)  ->  plan steps (A* search)
           ->  load data  ->  check constraints  ->  rank by utility  ->  explain

Run:  python agent.py
"""
import heapq
import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

QUERY = ("Find a Turkish restaurant in Downtown Baltimore, MD for two people to have dinner "
         "under $65 on Thursday night at 7:30 pm with a table for two near a window with a "
         "view of the garden or the street.")

DOWNTOWN = ["Downtown", "Inner Harbor", "Mount Vernon", "Harbor East"]


def log(tag, message):
    """Print one line of the agent's reasoning, e.g. [PLAN] ..."""
    print(f"[{tag}] {message}")


# =====================================================================================
# STEP 1 - Understand the query (Natural Language Processing)
# =====================================================================================
def parse_with_gemini(query):
    """Ask Gemini to turn the sentence into a JSON dictionary of constraints."""
    from google import genai

    client = genai.Client()  # reads the GEMINI_API_KEY environment variable
    prompt = (
        "Extract restaurant search constraints from the request below. Reply with JSON only, "
        "using these keys: cuisine, area, party_size (number), max_price (number, total for the "
        "party), day (e.g. Thursday), time (24-hour HH:MM), views (list, e.g. [\"garden\"]).\n\n"
        "Request: " + query
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    return json.loads(response.text)


def parse_with_rules(query):
    """Backup parser that uses simple keyword and regex matching (works offline)."""
    q = query.lower()
    hour, minute, ampm = re.search(r"(\d{1,2}):?(\d{2})?\s*(am|pm)", q).groups()
    hour = int(hour) + (12 if ampm == "pm" and int(hour) < 12 else 0)
    return {
        "cuisine": re.search(r"(turkish|greek|italian|lebanese|vietnamese)", q).group(1),
        "area": "downtown" if "downtown" in q else "",
        "party_size": 2 if "two" in q else int(re.search(r"for (\d+)", q).group(1)),
        "max_price": float(re.search(r"under \$?(\d+)", q).group(1)),
        "day": re.search(r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", q).group(1),
        "time": f"{hour:02d}:{minute or '00'}",
        "views": [v for v in ["garden", "street"] if v in q],
    }


# =====================================================================================
# STEP 2 - Deterministic planning: states, actions, goal  (textbook Ch 6)
# A state is a set of facts. Each action has preconditions and effects (STRIPS style).
# =====================================================================================
ACTIONS = {
    #  name                 preconditions                           effects           cost
    "parse_with_gemini": ({"have_query", "gemini_available"},   {"query_understood"}, 1),
    "parse_with_rules":  ({"have_query"},                        {"query_understood"}, 3),
    "load_restaurants":  ({"query_understood"},                  {"data_loaded"},      1),
    "check_constraints": ({"data_loaded"},                       {"filtered"},         1),
    "rank_results":      ({"filtered"},                          {"ranked"},           1),
    "explain_results":   ({"ranked"},                            {"explained"},        1),
}
GOAL = {"query_understood", "data_loaded", "filtered", "ranked", "explained"}


def a_star_plan(start_state):
    """Find the cheapest list of actions from start_state to GOAL using A* search (Ch 3).

    g = cost so far, h = number of goal facts still missing, f = g + h.
    """
    frontier = [(len(GOAL - start_state), 0, sorted(start_state), [])]
    visited = set()
    while frontier:
        f, g, state, plan = heapq.heappop(frontier)
        state = set(state)
        log("SEARCH", f"expand plan={plan or ['start']}  g={g}  h={f - g}  f={f}")
        if GOAL <= state:
            return plan
        if frozenset(state) in visited:
            continue
        visited.add(frozenset(state))
        for name, (pre, effects, cost) in ACTIONS.items():
            if pre <= state and not effects <= state:
                new_state = state | effects
                h = len(GOAL - new_state)
                heapq.heappush(frontier, (g + cost + h, g + cost, sorted(new_state), plan + [name]))
    return None


# =====================================================================================
# STEP 3 - Reasoning with constraints  (textbook Ch 4)
# Hard constraints must ALL be true. Soft constraints (preferences) are used for ranking.
# =====================================================================================
def hard_constraints(c):
    """Each constraint is (description, test). c = the parsed query."""
    return [
        (f"cuisine is {c['cuisine']}",      lambda r: r.cuisine.lower() == c["cuisine"].lower()),
        ("located in Downtown Baltimore",   lambda r: r.neighborhood in DOWNTOWN),
        (f"dinner for {c['party_size']} <= ${c['max_price']:.0f}",
                                            lambda r: r.cost_for_two / 2 * c["party_size"] <= c["max_price"]),
        (f"table free {c['day'].title()} {c['time']}",
                                            lambda r: c["time"] in r.thursday_slots.split(";")),
        (f"window with {' or '.join(c['views'])} view",
                                            lambda r: r.window_view in c["views"]),
    ]


def utility(r, c):
    """Soft constraints combined into one score out of 100 (higher is better)."""
    rating = (r.rating - 3) / 2                                  # 3 stars -> 0, 5 stars -> 1
    savings = (c["max_price"] - r.cost_for_two) / c["max_price"]  # more money left -> better
    view = 1.0 if r.window_view == c["views"][0] else 0.8        # first-named view preferred
    close = 1 - r.distance_km / 2                                 # closer to downtown -> better
    return round(100 * (0.4 * rating + 0.2 * savings + 0.2 * view + 0.2 * close), 1)


# =====================================================================================
# The agent: perceive -> plan -> act
# =====================================================================================
def run_agent(query):
    print("=" * 80)
    log("PERCEIVE", f'User request: "{query}"')
    state = {"have_query"} # initial state: we have the query, but nothing else yet
    if os.getenv("GEMINI_API_KEY"):
        state.add("gemini_available")
        log("PERCEIVE", "Gemini API key found")
    else:
        log("PERCEIVE", "No Gemini API key, the rule-based parser will be used")

    print("=" * 80)
    log("PLAN", f"Start state: {sorted(state)}")
    log("PLAN", f"Goal: {sorted(GOAL)}")
    plan = a_star_plan(state) 
    log("PLAN", "Plan found: " + " -> ".join(plan))

    print("=" * 80)
    constraints = data = passed = ranked = None # initialize variables for later steps
    for action in plan:
        log("ACT", f"Executing: {action}")

        if action == "parse_with_gemini":
            try:
                constraints = parse_with_gemini(query)
            except Exception as error:
                log("ACT", f"Gemini failed ({error}); falling back to rules")
                constraints = parse_with_rules(query)
            log("NLP", f"Constraints: {constraints}")

        elif action == "parse_with_rules":
            constraints = parse_with_rules(query)
            log("NLP", f"Constraints: {constraints}")

        elif action == "load_restaurants":
            data = pd.read_csv("restaurants.csv")
            log("DATA", f"Loaded {len(data)} restaurants from restaurants.csv")

        elif action == "check_constraints":
            passed = []
            for r in data.itertuples():
                failed = [name for name, test in hard_constraints(constraints) if not test(r)]
                if failed:
                    log("REJECT", f"{r.name}: fails {', '.join(failed)}")
                else:
                    log("ACCEPT", f"{r.name}: meets every constraint")
                    passed.append(r)
            log("FILTER", f"{len(passed)} of {len(data)} restaurants meet all hard constraints")

        elif action == "rank_results":
            ranked = sorted(passed, key=lambda r: utility(r, constraints), reverse=True)
            for r in ranked:
                log("RANK", f"{r.name}: utility = {utility(r, constraints)}")

        elif action == "explain_results":
            show_results(ranked, constraints)
    log("GOAL", "Goal reached")


def show_results(ranked, c):
    print("=" * 80)
    if not ranked:
        print("No restaurant meets all the constraints. See the REJECT lines above.")
        return
    print("FINAL RANKED RESULTS\n")
    for i, r in enumerate(ranked, 1):
        cost = r.cost_for_two / 2 * c["party_size"]
        print(f"#{i}  {r.name}   (utility {utility(r, c)}/100)")
        print(f"    ✔ Location: {r.neighborhood}, Downtown Baltimore, MD ({r.distance_km} km from center)")
        print(f"    ✔ Cuisine:  {r.cuisine}")
        print(f"    ✔ Price:    about ${cost:.0f} for {c['party_size']}, under the ${c['max_price']:.0f} budget")
        print(f"    ✔ Time:     table free {c['day'].title()} at {c['time']}")
        print(f"    ✔ View:     window table with a {r.window_view} view")
        print(f"    ★ Rating:   {r.rating} stars\n")

    names = [r.name for r in ranked][::-1]
    scores = [utility(r, c) for r in ranked][::-1]
    plt.figure(figsize=(8, 4))
    plt.barh(names, scores, color="#2F5D8A")
    plt.xlabel("Utility score (out of 100)")
    plt.title("Turkish restaurants in Downtown Baltimore, ranked")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig("results.png")
    print("Chart saved to results.png")


if __name__ == "__main__":
    run_agent(QUERY)
