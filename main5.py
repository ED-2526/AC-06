import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------
# 1. Carregar i validar dades
# ---------------------------------------------------------
def load_datasets(base_path):
    movies = pd.read_csv(os.path.join(base_path, "movies.csv"))
    ratings = pd.read_csv(os.path.join(base_path, "ratings.csv"))

    required_cols = {"userId", "movieId", "rating"}
    if not required_cols.issubset(ratings.columns):
        raise ValueError("ratings.csv no conté les columnes obligatòries")

    ratings = ratings.dropna()
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    # Garantim ratings entre 1 i 5
    ratings = ratings[(ratings["rating"] >= 1.0) & (ratings["rating"] <= 5.0)]

    return movies, ratings


# ---------------------------------------------------------
# 1b. Filtrar per usuaris i pel·lícules amb ≥ k ratings
# ---------------------------------------------------------
def filter_ratings_by_min_counts(ratings, k_user=5, k_item=5):
    """
    Deixa només:
      - usuaris amb k_user o més valoracions
      - pel·lícules amb k_item o més valoracions

    Es fa iterativament fins que es compleix per a tots.
    """
    filtered = ratings.copy()

    while True:
        before = len(filtered) # Nombre de files del dataframe

        user_counts = filtered.groupby("userId").size() # Ens diu quantes pel·lícules ha valorat cada usuari
        item_counts = filtered.groupby("movieId").size() # Ens diu quantes valoracions ha rebut cada pel·lícula

        valid_users = user_counts[user_counts >= k_user].index
        valid_items = item_counts[item_counts >= k_item].index
        
        filtered = filtered[
            filtered["userId"].isin(valid_users) &
            filtered["movieId"].isin(valid_items)
        ]

        after = len(filtered)
        if after == before:
            break
    return filtered


# ---------------------------------------------------------
# 2. Leave-One-Out per usuari
# ---------------------------------------------------------
def leave_one_out_split(ratings):
    # Per cada usuari, treiem una valoració(userid, movieid, rating, timestamp) que hagi fet
    test = ratings.groupby("userId").sample(1, random_state=42) # Guardem el contingut i l'índex de la fila en la que estava del ratings.csv filtrat
    train = ratings.drop(test.index) # Traiem la fila que hem fet servir pel test amb l'índex
    return train, test


# ---------------------------------------------------------
# 3. Matriu usuari-item
# ---------------------------------------------------------
def build_matrix(ratings):
    return ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)
    # Treiem tots els valors que no existeixin o que no siguin un número i li assignem 0 (no valorat).


# ---------------------------------------------------------
# 4. Similitud Pearson (User–User)
# ---------------------------------------------------------
def pearson_similarity(vec_a, vec_b):
    mask = (vec_a > 0) & (vec_b > 0)
    if mask.sum() == 0:
        return 0

    a_centered = vec_a[mask] - np.mean(vec_a[mask])
    b_centered = vec_b[mask] - np.mean(vec_b[mask])

    denom = np.sqrt(np.sum(a_centered ** 2)) * np.sqrt(np.sum(b_centered ** 2))
    if denom == 0:
        return 0

    return np.sum(a_centered * b_centered) / denom


# ---------------------------------------------------------
# 5. Predicció User–User (Col·laboratiu, Pearson)
# ---------------------------------------------------------
def predict_user_user(user, item, matrix, k=10):
    sims = []

    # Similitud amb cada altre usuari que hagi valorat l'ítem
    for other in matrix.index: # Mirem cada usuari de la matriu
        if other != user and item in matrix.columns and matrix.loc[other, item] > 0: # Comprovem que l'usuari hagi valorat la pel·lícula
            sim = pearson_similarity(matrix.loc[user], matrix.loc[other])
            sims.append((sim, matrix.loc[other, item], other)) #Afegim el valor de la similitud, el valor que li dona l'usuari i el userID

    sims.sort(key=lambda x: x[0], reverse=True)
    sims = sims[:k]

    user_ratings = matrix.loc[user][matrix.loc[user] > 0]

    if len(sims) == 0:
        # Fallback: mitjana de l'usuari o 3.0
        mean_user = float(user_ratings.mean()) if len(user_ratings) else 3.0
        return max(1.0, min(5.0, mean_user))

    mean_user = float(user_ratings.mean()) if len(user_ratings) else 3.0

    num, den = 0.0, 0.0
    for sim, rating, uid in sims:
        other_ratings = matrix.loc[uid][matrix.loc[uid] > 0]
        mean_other = float(other_ratings.mean()) if len(other_ratings) else 3.0
        num += sim * (rating - mean_other)
        den += abs(sim)

    prediction = mean_user if den == 0 else mean_user + num / den

    prediction = max(1.0, min(5.0, float(prediction)))
    return prediction


# ---------------------------------------------------------
# 6. Similitud Item–Item (Col·laboratiu, Adjusted Cosine)
# ---------------------------------------------------------
def adjusted_cosine(i_vec, j_vec):
    mask = (i_vec > 0) & (j_vec > 0)
    if mask.sum() == 0:
        return 0

    mean_user = (i_vec[mask] + j_vec[mask]) / 2
    i_c = i_vec[mask] - mean_user
    j_c = j_vec[mask] - mean_user

    denom = np.sqrt(np.sum(i_c ** 2)) * np.sqrt(np.sum(j_c ** 2))
    if denom == 0:
        return 0

    return np.sum(i_c * j_c) / denom


def predict_item_item(user, item, matrix, k=10):
    sims = []
    user_row = matrix.loc[user]

    if item not in matrix.columns:
        # Fallback: mitjana de l'usuari o 3.0
        user_ratings = user_row[user_row > 0]
        fallback = float(user_ratings.mean()) if len(user_ratings) else 3.0
        return max(1.0, min(5.0, fallback))

    rated_items = matrix.columns[user_row > 0]

    for rated_item in rated_items:
        sim = adjusted_cosine(matrix[rated_item], matrix[item])
        sims.append((sim, matrix.loc[user, rated_item]))

    user_ratings = user_row[user_row > 0]

    if not sims:
        fallback = float(user_ratings.mean()) if len(user_ratings) else 3.0
        return max(1.0, min(5.0, fallback))

    sims.sort(key=lambda x: x[0], reverse=True)
    sims = sims[:k]

    num = sum(sim * rating for sim, rating in sims)
    den = sum(abs(sim) for sim, _ in sims)

    if den == 0:
        prediction = float(user_ratings.mean()) if len(user_ratings) else 3.0
    else:
        prediction = num / den

    prediction = max(1.0, min(5.0, float(prediction)))
    return prediction


# ---------------------------------------------------------
# 7. Avaluació RMSE (collaborative)
# ---------------------------------------------------------
def evaluate(test, matrix, mode):
    real, pred = [], []

    for _, row in test.iterrows():
        user, item = int(row.userId), int(row.movieId)

        if user not in matrix.index or item not in matrix.columns:
            continue

        if mode == "user":
            p = predict_user_user(user, item, matrix)
        elif mode == "item":
            p = predict_item_item(user, item, matrix)
        else:
            raise ValueError(f"Mode '{mode}' desconegut a evaluate().")

        real.append(float(row.rating))
        pred.append(p)

    return np.sqrt(mean_squared_error(real, pred)) if len(real) else float("nan")


# ---------------------------------------------------------
# 8. Recomanació basada en contingut (gèneres)
# ---------------------------------------------------------
def build_genre_matrix(movies):
    """
    Construeix una matriu movieId x gènere amb 0/1 indicant si la pel·lícula té aquell gènere.
    """
    all_genres = set()
    for gen_str in movies["genres"].fillna(""):
        for g in str(gen_str).split("|"):
            if g and g != "(no genres listed)":
                all_genres.add(g)

    all_genres = sorted(all_genres)

    genre_matrix = pd.DataFrame(
        0,
        index=movies["movieId"].astype(int),
        columns=all_genres,
        dtype=int
    )

    for _, row in movies.iterrows():
        m_id = int(row["movieId"])
        for g in str(row["genres"]).split("|"):
            if g in genre_matrix.columns:
                genre_matrix.at[m_id, g] = 1

    return genre_matrix


def build_user_profiles(train, genre_matrix):
    """
    Perfil de gèneres per usuari: suma (rating * vector de gèneres).
    Retorna: dict userId -> vector numpy.
    """
    profiles = {}
    grouped = train.groupby("userId")

    for user, group in grouped:
        profile = np.zeros(genre_matrix.shape[1], dtype=float)
        for _, r in group.iterrows():
            m_id = int(r["movieId"])
            rating = float(r["rating"])
            if m_id in genre_matrix.index:
                profile += rating * genre_matrix.loc[m_id].values

        if np.linalg.norm(profile) > 0: # Si la norma del vector és 0 no el guardem (és a dir no té valors)
            profiles[user] = profile

    return profiles


def predict_content_based(user, item, user_profiles, genre_matrix, user_means):
    """
    Prediu la nota amb similitud de cosinus entre perfil usuari i vector de gèneres.
    """
    base = float(user_means.get(user, 3.0))

    # Clamp base
    base = max(1.0, min(5.0, base))

    if item not in genre_matrix.index:
        return base

    movie_vec = genre_matrix.loc[item].values

    if user not in user_profiles:
        return base

    user_vec = user_profiles[user]

    num = float(np.dot(user_vec, movie_vec))
    den = float(np.linalg.norm(user_vec) * np.linalg.norm(movie_vec))

    if den == 0:
        pred = base
    else:
        sim = num / den
        pred = base + sim * (5.0 - base)

    pred = max(1.0, min(5.0, float(pred)))
    return pred


def evaluate_content_based(movies, train, test):
    """
    Avalua el sistema basat en contingut amb RMSE i, a més,
    calcula les 5 millors recomanacions per a un usuari concret.
    Retorna: (rmse, llista_top5, target_user)
    """
    genre_matrix = build_genre_matrix(movies)
    user_profiles = build_user_profiles(train, genre_matrix)
    user_means = train.groupby("userId")["rating"].mean()

    # RMSE sobre test
    real, pred = [], []
    for _, row in test.iterrows():
        user = int(row["userId"])
        item = int(row["movieId"])
        rating_real = float(row["rating"])

        p = predict_content_based(user, item, user_profiles, genre_matrix, user_means)
        real.append(rating_real)
        pred.append(p)

    rmse = np.sqrt(mean_squared_error(real, pred)) if len(real) else float("nan")

    return rmse

# ---------------------------------------------------------
# 9. Funció general de recomanació
# ---------------------------------------------------------
def recommend(user, matrix, movies, mode="user", k=10, n_recs=5,
              user_profiles=None, genre_matrix=None, user_means=None):
    """
    Retorna una llista de (movieId, title, predicció) per a un usuari i mode donat.
    mode ∈ {"user", "item", "content"}
    """
    if user not in matrix.index:
        raise ValueError(f"L'usuari {user} no és a la matriu de ratings.")

    # Pel·lícules ja vistes
    seen = set(matrix.loc[user][matrix.loc[user] > 0].index.astype(int))
    all_movies = set(movies["movieId"].astype(int).values)
    candidates = list(all_movies - seen)

    id_to_title = movies.set_index("movieId")["title"].to_dict()

    recs = []

    for m in candidates:
        if mode == "user":
            if m not in matrix.columns:
                continue
            p = predict_user_user(user, m, matrix, k=k)
        elif mode == "item":
            if m not in matrix.columns:
                continue
            p = predict_item_item(user, m, matrix, k=k)
        elif mode == "content":
            if user_profiles is None or genre_matrix is None or user_means is None:
                raise ValueError("Per al mode 'content' calen user_profiles, genre_matrix i user_means.")
            p = predict_content_based(user, m, user_profiles, genre_matrix, user_means)
        else:
            raise ValueError(f"Mode '{mode}' no reconegut.")

        recs.append((m, id_to_title.get(m, "Títol desconegut"), p))

    recs.sort(key=lambda x: x[2], reverse=True)
    return recs[:n_recs]


# ---------------------------------------------------------
# 10. MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    path = "./ml_latest_small"  # directori de movies.csv i ratings.csv

    movies, ratings = load_datasets(path)

    # train_original, test_original = leave_one_out_split(ratings)
    # matrix_original = build_matrix(train_original)
    # print("Mida de la matriu usuari-item original:", matrix_original.shape)

    # Filtre per usuaris i pel·lícules amb almenys k valoracions
    k = 5
    ratings_filtered = filter_ratings_by_min_counts(ratings, k_user=k, k_item=k)

    train, test = leave_one_out_split(ratings_filtered)
    matrix = build_matrix(train)
    print("Mida de la matriu usuari-item filtrada:", matrix.shape)

    # Filtrat col·laboratiu (RMSE)
    # rmse_user = evaluate(test, matrix, mode="user")
    # rmse_item = evaluate(test, matrix, mode="item")

    # Recomanació basada en contingut: RMSE (fem servir la funció existent)
    rmse_content = evaluate_content_based(movies, train, test)

    # Per a les recomanacions content-based necessitem les estructures:
    genre_matrix = build_genre_matrix(movies)
    user_profiles = build_user_profiles(train, genre_matrix)
    user_means = train.groupby("userId")["rating"].mean()

    # Triem un usuari objectiu (per exemple, el primer del test)
    target_user = int(test.iloc[0]["userId"])

    # TOP-N recomanacions per a cada mètode
    top_n = 5
    recs_user = recommend(
        target_user, matrix, movies,
        mode="user", k=10, n_recs=top_n
    )
    recs_item = recommend(
        target_user, matrix, movies,
        mode="item", k=10, n_recs=top_n
    )
    recs_content = recommend(
        target_user, matrix, movies,
        mode="content", n_recs=top_n,
        user_profiles=user_profiles,
        genre_matrix=genre_matrix,
        user_means=user_means
    )

    print("\n📍 RESULTATS (Leave One Out + filtratge per usuaris i pel·lícules):")
    print(f"RMSE User–User (Pearson)        = {rmse_user:.4f}")
    print(f"RMSE Item–Item (AdjCosine)      = {rmse_item:.4f}")
    print(f"RMSE Content-Based (Gèneres)    = {rmse_content:.4f}")

    print(f"\n🎬 TOP {top_n} recomanacions USER–USER per a l'usuari {target_user}:")
    for m_id, title, score in recs_user:
        print(f"- {title} (movieId={m_id}) → predicció = {score:.2f}")

    print(f"\n🎬 TOP {top_n} recomanacions ITEM–ITEM per a l'usuari {target_user}:")
    for m_id, title, score in recs_item:
        print(f"- {title} (movieId={m_id}) → predicció = {score:.2f}")

    print(f"\n🎬 TOP {top_n} recomanacions CONTENT-BASED per a l'usuari {target_user}:")
    for m_id, title, score in recs_content:
        print(f"- {title} (movieId={m_id}) → predicció = {score:.2f}")
