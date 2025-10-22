import os
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import pandas as pd

# ========================
#  FASTAPI SETUP
# ========================
app = FastAPI(
    title="Movie Recommendation API",
    description="Content-based movie recommendation system using TMDB 5000 dataset.",
    version="1.0.0",
)

# Allow all origins (you can restrict this later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
#  LOAD AND PREPROCESS DATA
# ========================
print("Loading dataset...")
movies = pd.read_csv("movies_trimmed.csv")

# Clean and normalize genre strings
movies["genres"] = movies["genres"].fillna("").str.replace(" ", "").str.lower()

# Precompute TF-IDF and cosine similarity for efficiency
print("Vectorizing genres...")
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create index mapping (title → index)
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

print("✅ Dataset loaded and model ready.")


# ========================
#  ROUTE: RECOMMEND BY TITLE
# ========================
@app.get("/recommend")
def recommend(title: str):
    """
    Recommend similar movies based on a given title using cosine similarity.
    """
    # Normalize input
    title = title.strip().lower()

    # Find exact match
    if title not in indices.index.str.lower():
        # Try fuzzy match (e.g. "avangers" -> "Avengers: Endgame")
        close = get_close_matches(title, movies["title"].str.lower().tolist(), n=1, cutoff=0.6)
        if close:
            title = close[0]
        else:
            return {
                "recommendations": [],
                "message": f"'{title}' not found in dataset.",
                "status": "not_found",
            }

    # Get actual title in correct case
    matched_title = movies["title"][movies["title"].str.lower() == title].iloc[0]
    idx = indices[matched_title]

    # Compute similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]

    # Build recommendations
    recommendations = movies.iloc[movie_indices][["title", "id"]].to_dict(orient="records")

    return {
        "status": "success",
        "message": f"Top recommendations for '{matched_title}'",
        "recommendations": recommendations,
    }


# ========================
#  ROUTE: RECOMMEND BY USER PREFERENCES
# ========================
@app.get("/recommend/by_user")
def recommend_by_user(liked_ids: str = "", disliked_ids: str = ""):
    """
    Recommend movies based on a user's liked and disliked movies.
    """
    if not liked_ids:
        return {
            "status": "error",
            "message": "Please provide at least one liked movie ID.",
            "recommendations": [],
        }

    # Parse IDs
    liked = [int(i) for i in liked_ids.split(",") if i]
    disliked = [int(i) for i in disliked_ids.split(",") if i]

    # Filter valid IDs
    liked_valid = [i for i in liked if i in movies["id"].values]
    disliked_valid = [i for i in disliked if i in movies["id"].values]

    # Handle no valid movies
    if not liked_valid:
        return {
            "status": "not_found",
            "message": "None of the liked movies exist in the dataset.",
            "recommendations": [],
        }

    # Extract genre vectors
    genre_cols = TfidfVectorizer(stop_words="english").fit(movies["genres"]).get_feature_names_out()
    tfidf_matrix_user = tfidf_matrix  # reuse precomputed matrix

    # Create user preference vector (sum of liked movie vectors)
    liked_indices = movies[movies["id"].isin(liked_valid)].index
    user_profile = np.asarray(tfidf_matrix_user[liked_indices].mean(axis=0)).ravel().reshape(1, -1)

    # Compute similarity between user profile and all movies
    similarity_scores = cosine_similarity(user_profile, tfidf_matrix_user).flatten()

    # Exclude liked and disliked movies
    excluded_ids = set(liked_valid + disliked_valid)
    movie_scores = [
        (movies.iloc[i]["id"], similarity_scores[i])
        for i in range(len(movies))
        if movies.iloc[i]["id"] not in excluded_ids
    ]

    # Get top 10 recommendations
    top_movies = sorted(movie_scores, key=lambda x: x[1], reverse=True)[:10]
    top_movie_ids = [mid for mid, _ in top_movies]
    recommendations = movies[movies["id"].isin(top_movie_ids)][["title", "id"]].to_dict(orient="records")

    return {
        "status": "success",
        "message": "Personalized recommendations generated successfully.",
        "recommendations": recommendations,
    }


# ========================
#  MAIN ENTRY POINT
# ========================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)