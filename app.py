from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify Netlify domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/recommend")
def recommend(title: str):
    # Load trimmed data on each request
    movies = pd.read_csv("movies_trimmed.csv")
    movies['genre'] = movies['genre'].fillna('').str.replace(' ', '').str.lower()

    # TF-IDF on genre only
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genre'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

    if title not in indices:
        raise HTTPException(status_code=404, detail="Movie not found")

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]

    recommendations = movies.iloc[movie_indices][['title', 'id']].to_dict(orient='records')
    return {"recommendations": recommendations}
