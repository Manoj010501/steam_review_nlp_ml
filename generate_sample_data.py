"""
generate_sample_data.py
-----------------------
Generates realistic sample Steam review data for testing the NLP/ML pipeline.
Run this first before any other script.
"""

import pandas as pd
import numpy as np
import random
import json
import os
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ── Sample review templates ────────────────────────────────────────────────────

POSITIVE_REVIEWS = [
    "Absolutely love this game! The gameplay is smooth and the story is engaging.",
    "Best purchase I've made this year. Graphics are stunning and controls are responsive.",
    "Amazing game, highly recommend. Spent 100+ hours and still not bored.",
    "The developers really listen to the community. Updates keep improving the game.",
    "Great optimization, runs perfectly on my mid-range PC. Love the art style.",
    "Incredible storyline with memorable characters. The voice acting is top notch.",
    "Multiplayer is super fun with friends. Matchmaking is fast and fair.",
    "Worth every penny. The open world is huge and full of things to do.",
    "Performance is great, no bugs encountered. Very polished experience.",
    "10/10 would buy again. The music and atmosphere are phenomenal.",
    "Combat feels satisfying and deep. Lots of build variety to experiment with.",
    "The crafting system is well thought out. Hours of content to explore.",
    "Solid game with regular updates. Dev team is responsive on forums.",
    "Loved every minute of it. The puzzles are clever and rewarding.",
    "Runs at a stable 60fps on ultra settings. Visually breathtaking.",
]

NEGATIVE_REVIEWS = [
    "Terrible optimization. Constant stuttering even on high-end hardware.",
    "Game crashes every 30 minutes. Unplayable in its current state.",
    "Way too expensive for the amount of content. Feel cheated.",
    "The developers abandoned this game after launch. No updates in 6 months.",
    "Full of microtransactions and pay-to-win mechanics. Avoid.",
    "Story is boring and predictable. Characters have zero depth.",
    "Multiplayer servers are always down. Can't even play online.",
    "Riddled with bugs that break progression. Lost 10 hours of save data.",
    "Controls are clunky and unresponsive. Feels unfinished.",
    "False advertising. The trailers showed nothing like the actual game.",
    "Performance issues ruined my experience. Even low settings don't help.",
    "The UI is confusing and poorly designed. Hard to navigate menus.",
    "No controller support. Keyboard only on a game that screams for gamepad.",
    "Toxic community and no anti-cheat system. Hackers everywhere online.",
    "Repetitive gameplay with nothing new after the first hour.",
]

NEUTRAL_REVIEWS = [
    "Decent game but nothing groundbreaking. Gets repetitive after a while.",
    "Graphics are nice but gameplay is mediocre. Worth it on sale.",
    "Has potential but needs more polish. Might revisit after updates.",
    "Mixed feelings. Some parts are great, others feel rushed.",
    "Average experience overall. Neither impressed nor disappointed.",
    "Okay game for the price. Not my favorite genre but playable.",
    "Some good ideas poorly executed. The concept is better than the result.",
    "Finished it once, won't replay. Story is fine, gameplay serviceable.",
    "Runs okay on my PC. Not the best not the worst in the genre.",
    "Would recommend only if you're a fan of this type of game specifically.",
]

TOPICS = {
    "gameplay": [
        "The combat system is {adj}. {extra}",
        "Gameplay mechanics feel {adj}. {extra}",
        "The controls are {adj} and {adj2}.",
    ],
    "performance": [
        "Game runs {adj} on my PC. {extra}",
        "Performance is {adj}. Getting {fps}fps on {settings} settings.",
        "Optimization is {adj}. {extra}",
    ],
    "story": [
        "The storyline is {adj}. {extra}",
        "Characters are {adj} and well-written.",
        "The narrative is {adj}. {extra}",
    ],
}

MONTHS = [
    "2023-01", "2023-02", "2023-03", "2023-04",
    "2023-05", "2023-06", "2023-07", "2023-08",
    "2023-09", "2023-10", "2023-11", "2023-12",
    "2024-01", "2024-02", "2024-03",
]

GAMES = [
    {"app_id": "1091500", "name": "Cyberpunk 2077"},
    {"app_id": "1245620", "name": "Elden Ring"},
    {"app_id": "1174180", "name": "Red Dead Redemption 2"},
    {"app_id": "892970",  "name": "Valheim"},
    {"app_id": "1203220", "name": "DEATH STRANDING"},
]


def generate_reviews(n: int = 300) -> pd.DataFrame:
    """Generate n synthetic Steam reviews across multiple games and months."""
    records = []

    for _ in range(n):
        game = random.choice(GAMES)
        month = random.choice(MONTHS)

        # weighted: 55% pos, 30% neg, 15% neutral
        bucket = random.choices(
            ["positive", "negative", "neutral"],
            weights=[0.55, 0.30, 0.15]
        )[0]

        if bucket == "positive":
            text = random.choice(POSITIVE_REVIEWS)
        elif bucket == "negative":
            text = random.choice(NEGATIVE_REVIEWS)
        else:
            text = random.choice(NEUTRAL_REVIEWS)

        # random play time
        hours = round(random.expovariate(1 / 50) + 1, 1)

        # parse month to a random day within that month
        year, mon = map(int, month.split("-"))
        day = random.randint(1, 28)
        date_str = f"{year}-{mon:02d}-{day:02d}"

        records.append({
            "review_id":       f"rev_{len(records):05d}",
            "app_id":          game["app_id"],
            "game_name":       game["name"],
            "review_text":     text,
            "voted_up":        bucket == "positive",
            "playtime_hours":  hours,
            "date":            date_str,
            "month":           month,
            "true_sentiment":  bucket,   # ground truth for evaluation
        })

    df = pd.DataFrame(records)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/sample_reviews.csv", index=False)
    print(f"✅  Generated {len(df)} sample reviews → data/sample_reviews.csv")
    return df


if __name__ == "__main__":
    df = generate_reviews(300)
    print(df.head())
    print("\nSentiment distribution:\n", df["true_sentiment"].value_counts())
    print("\nGames:\n", df["game_name"].value_counts())