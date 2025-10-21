from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and prepare data
movies = pd.read_csv('TMDB Dataset (10K Movies).csv')
movies['genre'] = movies['genre'].fillna('').str.lower().str.replace(' ', '')
movies['genre'] = movies['genre'].apply(lambda x: x.split(',') if x else [])

# Encode genres using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies['genre'])

# Compute cosine similarity
genre_sim = cosine_similarity(genre_matrix)

# Title to index mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# FastAPI app
app = FastAPI(title="Movie Recommender (Genre-Based)")

# Optional: Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for React apps
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Welcome to the Genre-Based Movie Recommendation API"}

@app.get("/recommend")
def recommend(title: str = Query(..., description="Movie title to base recommendations on")):
    if title not in indices:
        raise HTTPException(status_code=404, detail="Movie not found")

    idx = indices[title]
    sim_scores = list(enumerate(genre_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]

    results = movies.iloc[movie_indices][['id', 'title', 'genre', 'release_date']].to_dict(orient='records')
    return {"recommendations": results}
