import os
import ast
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from nltk.stem.porter import PorterStemmer



# -------------------------------------------------
# 1. CÀRREGA DE DADES (COM AL NOTEBOOK)
# -------------------------------------------------

def load_raw_datasets(base_path: str):
    """
    Carrega els CSV del The Movies Dataset (mateixos que al notebook).
    """
    credits = pd.read_csv(os.path.join(base_path, "credits.csv"))
    keywords = pd.read_csv(os.path.join(base_path, "keywords.csv"))
    links = pd.read_csv(os.path.join(base_path, "links.csv"))
    links_small = pd.read_csv(os.path.join(base_path, "links_small.csv"))
    movies_metadata = pd.read_csv(os.path.join(base_path, "movies_metadata.csv"))
    ratings = pd.read_csv(os.path.join(base_path, "ratings.csv"))
    ratings_small = pd.read_csv(os.path.join(base_path, "ratings_small.csv"))

    return credits, keywords, links, links_small, movies_metadata, ratings, ratings_small


# -------------------------------------------------
# 2. NETEJA I MERGE DE MOVIES (COM AL NOTEBOOK)
# -------------------------------------------------

def clean_and_merge_movies(
    credits: pd.DataFrame,
    keywords: pd.DataFrame,
    links: pd.DataFrame,
    links_small: pd.DataFrame,
    movies_metadata: pd.DataFrame,
):
    """
    Reprodueix la lògica del notebook:
      - Elimina duplicats
      - Concat links & links_small -> all_links
      - Merge credits + keywords -> credits_keywords
      - Converteix id a numèric
      - Merge movies_metadata + credits_keywords + all_links -> movies
      - Elimina columnes sobrants
    """

    # 1) Eliminar duplicats com feies al notebook
    credits = credits.drop_duplicates()
    credits = credits.drop_duplicates(subset=["id"])

    keywords = keywords.drop_duplicates()

    movies_metadata = movies_metadata.drop_duplicates(subset=["id"])

    links = links.drop_duplicates(subset=["tmdbId"])
    links_small = links_small.drop_duplicates(subset=["tmdbId"])

    # 2) Concat de links
    all_links = pd.concat([links, links_small])

    # 3) Merge credits + keywords
    credits_keywords = credits.merge(keywords, on="id")

    # 4) id numèrica
    movies_metadata["id"] = pd.to_numeric(movies_metadata["id"], errors="coerce")
    credits_keywords["id"] = pd.to_numeric(credits_keywords["id"], errors="coerce")
    all_links["tmdbId"] = pd.to_numeric(all_links["tmdbId"], errors="coerce")

    movies_metadata = movies_metadata.dropna(subset=["id"])
    credits_keywords = credits_keywords.dropna(subset=["id"])
    all_links = all_links.dropna(subset=["tmdbId"])

    movies_metadata["id"] = movies_metadata["id"].astype("Int64")
    credits_keywords["id"] = credits_keywords["id"].astype("Int64")
    all_links["tmdbId"] = all_links["tmdbId"].astype("Int64")

    # 5) movies = movies_metadata + credits_keywords
    movies = movies_metadata.merge(credits_keywords, on="id")

    # 6) Afegir movieId de MovieLens via tmdbId
    movies = movies.merge(all_links, left_on="id", right_on="tmdbId")

    # 7) Duplicats
    movies = movies.drop_duplicates()
    movies = movies.drop_duplicates(subset=["id"])

    # 8) Drop de columnes sobrants (aprox com al notebook)
    cols_to_drop = [
        "belongs_to_collection",
        "homepage",
        "imdb_id",
        "poster_path",
        "status",
        "tagline",
        "video",
        "vote_average",
        "tmdbId",
        "imdbId",
        "adult",
        "original_language",
        "original_title",
        "spoken_languages",
        "production_countries",
        "runtime",
    ]
    cols_to_drop = [c for c in cols_to_drop if c in movies.columns]
    movies = movies.drop(cols_to_drop, axis=1)

    movies = movies.reset_index(drop=True)

    return movies


# -------------------------------------------------
# 3. CREACIÓ DE TAGS (MATEIX ENFOC QUE EL NOTEBOOK)
# -------------------------------------------------

def extract_names(text_list):
    """
    Funció del notebook: parseja string de llista de dicts i extreu 'name'.
    """
    try:
        data_list = ast.literal_eval(text_list)
        names = [item["name"] for item in data_list]
        return names
    except (ValueError, SyntaxError, TypeError):
        return []


def build_tags_column(movies: pd.DataFrame):
    """
    Reprodueix la construcció de 'tags' del notebook:
      - extract_names en: genres, production_companies, keywords, cast, crew
      - overview -> split
      - tags = genres + overview + production_companies + cast + crew + keywords
      - minúscules, join, etc.
    """
    movies = movies.copy()

    # Ens assegurem que les columnes existeixen
    for col in ["genres", "production_companies", "keywords", "cast", "crew", "overview"]:
        if col not in movies.columns:
            movies[col] = ""

    movies["genres"] = movies["genres"].apply(extract_names)
    movies["production_companies"] = movies["production_companies"].apply(extract_names)
    movies["keywords"] = movies["keywords"].apply(extract_names)
    movies["cast"] = movies["cast"].apply(extract_names)
    movies["crew"] = movies["crew"].apply(extract_names)

    # overview com llista de paraules
    movies["overview"] = movies["overview"].fillna("")
    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    # Construcció de tags (com al notebook)
    movies["tags"] = (
        movies["genres"]
        + movies["overview"]
        + movies["production_companies"]
        + movies["cast"]
        + movies["crew"]
        + movies["keywords"]
    )

    movies["tags"] = movies["tags"].apply(lambda x: " ".join(x))
    movies["tags"] = movies["tags"].apply(lambda x: x.lower())

    # Eliminem columnes que ja no calen (com al notebook)
    columns_to_drop = ["genres", "overview", "production_companies", "cast", "crew", "keywords"]
    columns_to_drop = [c for c in columns_to_drop if c in movies.columns]
    movies = movies.drop(columns_to_drop, axis=1, errors="ignore")

    return movies


# -------------------------------------------------
# 4. VECTORS, SIMILARITY I STEMMING (COM AL NOTEBOOK)
# -------------------------------------------------

def apply_stemming_to_tags(movies: pd.DataFrame):
    """
    Aplica PorterStemmer a la columna 'tags' (com feies al notebook).
    """
    ps = PorterStemmer()

    def stem(text):
        y = []
        for word in text.split():
            y.append(ps.stem(word))
        return " ".join(y)

    movies = movies.copy()
    movies["tags"] = movies["tags"].apply(stem)
    return movies


def build_vectors_and_similarity(movies: pd.DataFrame, max_features: int = 5000):
    """
    Fa:
      - CountVectorizer sobre 'tags'
      - vectors = .toarray()
      - similarity = cosine_similarity(vectors)
    """
    cv = CountVectorizer(max_features=max_features, stop_words="english")
    vectors = cv.fit_transform(movies["tags"]).toarray()
    similarity = cosine_similarity(vectors)
    return cv, vectors, similarity


# -------------------------------------------------
# 5. RECOMANADOR (COM EL NOTEBOOK, PER TÍTOL)
# -------------------------------------------------

def recommend_by_title(movie_title: str, movies: pd.DataFrame, similarity: np.ndarray, top_n: int = 5):
    """
    Idèntic al recommend del notebook:
    recomana pel·lícules semblants a una pel·lícula donada pel seu títol.
    """
    if movie_title not in movies["title"].values:
        print(f"'{movie_title}' no es troba al catàleg.")
        return

    index = movies[movies["title"] == movie_title].index[0]
    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, reverse=True, key=lambda x: x[1])

    for i in distances[1 : top_n + 1]:
        print(movies.iloc[i[0]].title)


def recommend_hybrid_by_popularity(movie_title: str, movies: pd.DataFrame,
                                   similarity: np.ndarray, top_n: int = 5, candidate_pool: int = 50):
    """
    Versió híbrid (com al teu notebook):
      - agafa les pel·lícules més similars (content-based)
      - ordena per 'popularity'
    """
    if movie_title not in movies["title"].values:
        print(f"'{movie_title}' no es troba al catàleg.")
        return

    movies = movies.copy()
    movies["popularity"] = pd.to_numeric(movies["popularity"], errors="coerce")

    index = movies[movies["title"] == movie_title].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    similar_indices = [i[0] for i in distances[1 : candidate_pool + 1]]
    similar_movies = movies.iloc[similar_indices]

    sorted_by_popularity = similar_movies.sort_values("popularity", ascending=False)

    for title in sorted_by_popularity["title"].head(top_n):
        print(title)


# -------------------------------------------------
# 6. MACHINE LEARNING: TRAIN / TEST SOBRE RÀTINGS
# -------------------------------------------------

def build_all_ratings(ratings: pd.DataFrame, ratings_small: pd.DataFrame):
    """
    Replica el all_ratings = concat(ratings, ratings_small) del notebook.
    """
    all_ratings = pd.concat([ratings, ratings_small])
    # assegurem tipus
    all_ratings["userId"] = all_ratings["userId"].astype(int)
    all_ratings["movieId"] = all_ratings["movieId"].astype(int)
    return all_ratings


def build_ml_dataset(movies: pd.DataFrame,
                     vectors: np.ndarray,
                     all_ratings: pd.DataFrame):
    """
    Crea un dataset supervisat per ML:
      - Features: [userId, vector_de_tags_de_la_pel·li]
      - Label: rating

    Ús clau del notebook:
      - movies ja té 'movieId'
      - vectors és en el mateix ordre que movies (index -> fila)
    """

    # Map movieId -> index al DataFrame movies (el mateix ordre que vectors)
    if "movieId" not in movies.columns:
        raise ValueError("El DataFrame 'movies' ha de tenir la columna 'movieId' (ve de links).")

    movieid_to_index = pd.Series(movies.index.values, index=movies["movieId"])

    # Ens quedem amb ratings només de pel·lícules que tenim a 'movies'
    ratings_filtered = all_ratings[all_ratings["movieId"].isin(movieid_to_index.index)].copy()

    # Per cada rating, agafem el vector de la pel·lícula
    row_features = []
    labels = []

    for _, row in ratings_filtered.iterrows():
        mid = row["movieId"]
        uid = row["userId"]
        rating = row["rating"]

        idx = movieid_to_index[mid]
        movie_vec = vectors[idx]  # vector de tags de la pel·li

        # Feature = [userId, movie_vec...]
        feature = np.concatenate(([uid], movie_vec))
        row_features.append(feature)
        labels.append(rating)

    X = np.array(row_features)
    y = np.array(labels, dtype=float)

    return X, y, ratings_filtered, movieid_to_index


def train_test_split_ml(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Separa en train / test (tal com et demanen).
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_rating_model(X_train, y_train, n_estimators: int = 50, random_state: int = 42):
    """
    Entrena un RandomForestRegressor per predir ratings.
    (pots canviar el model si vols provar altres coses).
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
    Avaluació amb RMSE.
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE al test: {rmse:.4f}")
    return rmse


# -------------------------------------------------
# 7. RECOMANADOR ML PER USUARI (AMB MODEL ENTRENAT)
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
    Recomana pel·lícules per a un usuari fent servir el model ML:

      - No recomanem pel·lis que aquest usuari ja ha valorat
      - Per cada candidata: feature = [userId, vector_tags_pel·li]
      - Predim rating i triem les millors
    """

    # Pel·lícules ja valorades per l'usuari
    seen_movie_ids = ratings_filtered.loc[
        ratings_filtered["userId"] == user_id, "movieId"
    ].unique()

    # Candidates: pel·lícules de 'movies' que l'usuari no ha vist
    candidate_movies = movies[~movies["movieId"].isin(seen_movie_ids)].copy()

    if candidate_movies.empty:
        print(f"L'usuari {user_id} ja ha valorat totes les pel·lícules disponibles.")
        return

    # Preparem features per a totes les candidates
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
# 8. PIPELINE COMPLET (COM EN UN MAIN)
# -------------------------------------------------

if __name__ == "__main__":
    relative_path = "dataset"
    base_path = os.path.join(os.path.dirname(__file__), relative_path)

    # 1) Carrega dades crues (mateix que al notebook)
    credits, keywords, links, links_small, movies_metadata, ratings, ratings_small = load_raw_datasets(base_path)

    # 2) Crea movies (merge + neteja) com al notebook
    movies = clean_and_merge_movies(credits, keywords, links, links_small, movies_metadata)

    # 3) Construeix tags
    movies = build_tags_column(movies)

    # 4) Stemming + vectors + similarity (per la part content-based)
    movies = apply_stemming_to_tags(movies)
    cv, vectors, similarity = build_vectors_and_similarity(movies)

    # 5) Part ML: dataset de ratings i split train/test
    all_ratings = build_all_ratings(ratings, ratings_small)
    X, y, ratings_filtered, movieid_to_index = build_ml_dataset(movies, vectors, all_ratings)
    X_train, X_test, y_train, y_test = train_test_split_ml(X, y)

    # 6) Entrenament model + avaluació
    model = train_rating_model(X_train, y_train)
    evaluate_rating_model(model, X_test, y_test)

    # 7) Exemple de recomanació ML per usuari
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

    # 8) (Opcional) Recomanacions content-based com al notebook
    print("\nRecomanacions content-based per títol ('3 Idiots'):")
    recommend_by_title("3 Idiots", movies, similarity, top_n=5)

    print("\nRecomanacions híbrides per popularitat ('Jumanji'):")
    recommend_hybrid_by_popularity("Jumanji", movies, similarity, top_n=5)
