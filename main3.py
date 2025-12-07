import os
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from nltk.stem.porter import PorterStemmer


# -------------------------------------------------
# 1. CÀRREGA DE DADES (ml-latest-small)
# -------------------------------------------------

def load_ml_latest_small(base_path: str):
    """
    Carrega els datasets de MovieLens ml-latest-small:

    - movies.csv
    - ratings.csv
    - tags.csv
    - links.csv
    """
    movies = pd.read_csv(os.path.join(base_path, "movies.csv"))
    ratings = pd.read_csv(os.path.join(base_path, "ratings.csv"))
    tags = pd.read_csv(os.path.join(base_path, "tags.csv"))
    links = pd.read_csv(os.path.join(base_path, "links.csv"))

    return movies, ratings, tags, links


# -------------------------------------------------
# 2. CONSTRUCCIÓ DE TAGS PER PEL·LÍCULA
# -------------------------------------------------

def build_movie_tags(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """
    Crea una columna 'tags' a partir de:
      - genres de movies.csv
      - tags de tags.csv (agregats per movieId)
    """

    movies = movies.copy()

    # 2.1 Processar els gèneres (genres: "Adventure|Animation|Children|Comedy|Fantasy")
    # → llista de paraules
    movies["genres"] = movies["genres"].fillna("")
    movies["genre_list"] = movies["genres"].apply(lambda x: x.replace("|", " ").lower().split())

    # 2.2 Agregar tags dels usuaris per movieId
    tags = tags.copy()
    tags["tag"] = tags["tag"].fillna("").astype(str)

    movie_to_tags = defaultdict(list)
    for _, row in tags.iterrows():
        mid = row["movieId"]
        tag_text = str(row["tag"]).lower().replace(",", " ")
        movie_to_tags[mid].append(tag_text)

    # convertim a string per cada movieId
    movies["user_tags"] = movies["movieId"].map(
        lambda mid: " ".join(movie_to_tags.get(mid, []))
    )

    # 2.3 Construir 'tags' finals: generes + user_tags + (opcional) títol
    movies["title_clean"] = movies["title"].fillna("").str.lower().str.replace("[^a-z0-9 ]", " ", regex=True)

    def combine_tags(row):
        genre_part = " ".join(row["genre_list"])
        user_tags = row["user_tags"] if isinstance(row["user_tags"], str) else ""
        title_part = row["title_clean"]
        return f"{genre_part} {user_tags} {title_part}".strip()

    movies["tags"] = movies.apply(combine_tags, axis=1)

    # Neteja bàsica: múltiples espais → 1 espai
    movies["tags"] = movies["tags"].str.replace(r"\s+", " ", regex=True)

    # Ens quedem amb columnes útils
    movies_final = movies[["movieId", "title", "genres", "tags"]].copy()

    return movies_final


# -------------------------------------------------
# 3. STEMMING I VECTORS DE CONTINGUT
# -------------------------------------------------

def apply_stemming_to_tags(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica PorterStemmer a la columna 'tags'.
    """
    ps = PorterStemmer()

    def stem(text):
        y = []
        for word in text.split():
            y.append(ps.stem(word))
        return " ".join(y)

    movies = movies.copy()
    movies["tags"] = movies["tags"].fillna("").astype(str)
    movies["tags"] = movies["tags"].apply(stem)
    return movies


def build_vectors_and_similarity(movies: pd.DataFrame, max_features: int = 5000):
    """
    Vectoritza 'tags' i calcula la matriu de similitud cosinus.
    """
    cv = CountVectorizer(max_features=max_features, stop_words="english")
    vectors = cv.fit_transform(movies["tags"]).toarray()
    similarity = cosine_similarity(vectors)
    return cv, vectors, similarity


# -------------------------------------------------
# 4. RECOMANADOR CONTENT-BASED PER TÍTOL
# -------------------------------------------------

def recommend_by_title(movie_title: str, movies: pd.DataFrame,
                       similarity: np.ndarray, top_n: int = 5):
    """
    Recomana pel·lícules semblants a una pel·lícula donada pel títol.
    """
    if movie_title not in movies["title"].values:
        print(f"'{movie_title}' no es troba al catàleg.")
        return

    index = movies[movies["title"] == movie_title].index[0]
    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, reverse=True, key=lambda x: x[1])

    print(f"Pel·lícules similars a '{movie_title}':")
    for i in distances[1 : top_n + 1]:
        print(f"- {movies.iloc[i[0]].title}")


# -------------------------------------------------
# 5. DATASET ML PER PREDIR RÀTINGS (train/test)
# -------------------------------------------------

def build_ml_dataset(movies: pd.DataFrame,
                     vectors: np.ndarray,
                     ratings: pd.DataFrame):
    """
    Construeix dataset supervisat:

      Per cada fila de ratings:
        Feature: [userId, vector_de_contingut_de_la_pel·li]
        Label: rating
    """

    # map movieId -> index de la fila a 'movies' (mateix ordre que vectors)
    movieid_to_index = pd.Series(movies.index.values, index=movies["movieId"])

    # Ens quedem només amb ratings de pel·lícules que tenim a movies
    ratings_filtered = ratings[ratings["movieId"].isin(movieid_to_index.index)].copy()

    ratings_filtered["userId"] = ratings_filtered["userId"].astype(int)
    ratings_filtered["movieId"] = ratings_filtered["movieId"].astype(int)

    row_features = []
    labels = []

    for _, row in ratings_filtered.iterrows():
        uid = row["userId"]
        mid = row["movieId"]
        rating = row["rating"]

        idx = movieid_to_index[mid]
        movie_vec = vectors[idx]  # vector de contingut

        feature = np.concatenate(([uid], movie_vec))
        row_features.append(feature)
        labels.append(rating)

    X = np.array(row_features)
    y = np.array(labels, dtype=float)

    return X, y, ratings_filtered, movieid_to_index


def train_test_split_ml(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Separa el dataset ML en train i test.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_rating_model(X_train, y_train, n_estimators: int = 50, random_state: int = 42):
    """
    Entrena un RandomForestRegressor per predir ratings.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_rating_model(model, X_test, y_test):
    """
    Avaluació del model amb RMSE.
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE al test: {rmse:.4f}")
    return rmse


# -------------------------------------------------
# 6. RECOMANADOR ML PER USUARI
# -------------------------------------------------

def recommend_for_user_ml(
    user_id: int,
    model,
    movies: pd.DataFrame,
    vectors: np.ndarray,
    ratings_filtered: pd.DataFrame,
    movieid_to_index: pd.Series,
    top_n: int = 5,
):
    """
    Recomana pel·lícules per a un usuari utilitzant el model ML:

    - No recomanem pel·lícules que l'usuari ja ha valorat
    - Per cada candidata: feature = [userId, vector_contingut_pel·li]
    - Predim rating i triem les millors
    """

    # Pel·lícules ja valorades per l'usuari
    seen_movie_ids = ratings_filtered.loc[
        ratings_filtered["userId"] == user_id, "movieId"
    ].unique()

    # Candidates = totes les pel·lis que l'usuari no ha valorat
    candidate_movies = movies[~movies["movieId"].isin(seen_movie_ids)].copy()

    if candidate_movies.empty:
        print(f"L'usuari {user_id} ja ha valorat totes les pel·lícules disponibles.")
        return

    candidate_features = []
    candidate_indices = []

    for idx, row in candidate_movies.iterrows():
        mid = row["movieId"]
        if mid not in movieid_to_index.index:
            continue
        vec_idx = movieid_to_index[mid]
        movie_vec = vectors[vec_idx]
        feature = np.concatenate(([user_id], movie_vec))
        candidate_features.append(feature)
        candidate_indices.append(idx)

    if not candidate_features:
        print("No hi ha candidates amb vectors disponibles.")
        return

    X_new = np.array(candidate_features)
    predicted_ratings = model.predict(X_new)

    candidate_movies = candidate_movies.loc[candidate_indices].copy()
    candidate_movies["predicted_rating"] = predicted_ratings

    top_recs = candidate_movies.sort_values(
        by="predicted_rating", ascending=False
    ).head(top_n)

    print(f"Recomanacions ML per a l'usuari {user_id}:")
    for _, row in top_recs.iterrows():
        print(f"- {row['title']} (movieId={row['movieId']}), predicció: {row['predicted_rating']:.2f}")

    return top_recs


# -------------------------------------------------
# 7. PIPELINE COMPLET
# -------------------------------------------------

if __name__ == "__main__":
    # Canvia aquesta ruta per on tinguis el ml-latest-small
    base_path = "./"   # per exemple si tens movies.csv, ratings.csv, etc. al mateix directori

    # 1) Carregar dades
    movies_raw, ratings, tags, links = load_ml_latest_small(base_path)

    # 2) Construir tags per pel·lícula
    movies = build_movie_tags(movies_raw, tags)

    # 3) Stemming + vectors + similitud
    movies = apply_stemming_to_tags(movies)
    cv, vectors, similarity = build_vectors_and_similarity(movies)

    # 4) Dataset ML (user + contingut -> rating) i train/test
    X, y, ratings_filtered, movieid_to_index = build_ml_dataset(movies, vectors, ratings)
    X_train, X_test, y_train, y_test = train_test_split_ml(X, y)

    # 5) Entrenar model i avaluar
    model = train_rating_model(X_train, y_train)
    evaluate_rating_model(model, X_test, y_test)

    # 6) Exemple recomanador ML per usuari
    example_user_id = 1
    recommend_for_user_ml(
        user_id=example_user_id,
        model=model,
        movies=movies,
        vectors=vectors,
        ratings_filtered=ratings_filtered,
        movieid_to_index=movieid_to_index,
        top_n=5,
    )

    # 7) Exemple recomanador content-based per títol
    print("\nRecomanacions content-based per títol ('Toy Story (1995)'):")
    recommend_by_title("Toy Story (1995)", movies, similarity, top_n=5)
