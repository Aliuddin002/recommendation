import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

# Load data
track_df = pd.read_csv("track_df_cleaned.csv")

# ─── Helper Functions ─────────────────────────────
def clean_string(val):
    return str(val).strip().lower() if pd.notna(val) else ""

def recommend_based_on_genre_artist(track_id, track_df, top_k=5):
    if track_id not in track_df['track_id'].values:
        return pd.DataFrame()

    input_track = track_df[track_df['track_id'] == track_id].iloc[0]
    genre = clean_string(input_track['genre_top'])
    artist = clean_string(input_track['artist_name'])

    candidates = track_df[track_df['track_id'] != track_id].copy()
    candidates['genre_top'] = candidates['genre_top'].apply(clean_string)
    candidates['artist_name'] = candidates['artist_name'].apply(clean_string)

    candidates['similarity'] = 0.0
    candidates.loc[candidates['genre_top'] == genre, 'similarity'] += 0.7
    candidates.loc[candidates['artist_name'] == artist, 'similarity'] += 0.3

    top = candidates.sort_values(by='similarity', ascending=False).head(top_k * 2)
    diverse = candidates[candidates['genre_top'] != genre].sample(n=min(5, len(candidates)), random_state=42)
    return pd.concat([top.head(top_k), diverse]).drop_duplicates(subset='track_id').head(top_k + 3)

def recommend_from_favorites(fav_ids, track_df, top_k=10):
    all_recs = pd.DataFrame()
    for tid in fav_ids:
        recs = recommend_based_on_genre_artist(tid, track_df, top_k=3)
        all_recs = pd.concat([all_recs, recs])
    return all_recs.drop_duplicates(subset='track_id').sort_values(by='similarity', ascending=False).head(top_k)

def recommend_from_watch_history(history_ids, track_df, top_k=10):
    return recommend_from_favorites(history_ids, track_df, top_k)

# ─── FastAPI Setup ────────────────────────────────
app = FastAPI(title="Recommendation API")

class RecommendationRequest(BaseModel):
    track_ids: List[int]
    source: str  # "history" or "favorites"
    top_k: int = 10

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(req: RecommendationRequest):
    if not req.track_ids:
        raise HTTPException(status_code=400, detail="No track IDs provided.")

    if req.source == "history":
        recs = recommend_from_watch_history(req.track_ids, track_df, req.top_k)
    elif req.source == "favorites":
        recs = recommend_from_favorites(req.track_ids, track_df, req.top_k)
    else:
        raise HTTPException(status_code=400, detail="Invalid source. Use 'history' or 'favorites'.")

    return recs[['track_id', 'title', 'genre_top', 'artist_name', 'similarity']].to_dict(orient='records')

# ─── Run Uvicorn ──────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("recommend_api:app", host="0.0.0.0", port=8001, reload=True)
