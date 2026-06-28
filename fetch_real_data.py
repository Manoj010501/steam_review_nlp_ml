import requests
import pandas as pd

def get_reviews(app_id, total_reviews=200):
    reviews = []
    cursor = "*"
    while len(reviews) < total_reviews:
        params = {
            "json": 1,
            "language": "english",
            "num_per_page": 100,
            "cursor": cursor,
        }
        response = requests.get(
            f"https://store.steampowered.com/appreviews/{app_id}",
            params=params
        )
        data = response.json()
        if "reviews" not in data:
            break
        reviews.extend(data["reviews"])
        cursor = data["cursor"]
        if len(data["reviews"]) == 0:
            break
    return reviews[:total_reviews]

app_ids = {
    "Cyberpunk 2077":       "1091500",
    "Elden Ring":           "1245620",
    "Valheim":              "892970",
    "Red Dead Redemption 2":"1174180",
    "DEATH STRANDING":      "1203220",
}

all_reviews = []
for game_name, app_id in app_ids.items():
    print(f"Fetching {game_name}...")
    reviews = get_reviews(app_id, total_reviews=200)
    for r in reviews:
        all_reviews.append({
            "review_id":      r["recommendationid"],
            "app_id":         app_id,
            "game_name":      game_name,
            "review_text":    r["review"],
            "voted_up":       r["voted_up"],
            "playtime_hours": round(r["author"]["playtime_forever"] / 60, 1),
            "date":           str(pd.to_datetime(
                                  r["timestamp_created"], unit="s").date()),
            "month":          pd.to_datetime(
                                  r["timestamp_created"],
                                  unit="s").strftime("%Y-%m"),
        })
    print(f"  Got {len(reviews)} reviews for {game_name}")

df = pd.DataFrame(all_reviews)
df.to_csv("data/sample_reviews.csv", index=False)
print(f"\nDone! Saved {len(df)} real reviews to data/sample_reviews.csv")
print(df[["game_name", "review_text", "month"]].head())