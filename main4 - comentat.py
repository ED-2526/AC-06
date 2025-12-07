import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------
# 1. Carregar i validar dades
# ---------------------------------------------------------
def load_datasets(base_path):
    # Càrrega de les dades
    movies = pd.read_csv(os.path.join(base_path, "movies.csv"))
    ratings = pd.read_csv(os.path.join(base_path, "ratings.csv"))

    required_cols = {"userId", "movieId", "rating"}
    if not required_cols.issubset(ratings.columns):
        raise ValueError("❌ ratings.csv no conté les columnes obligatòries")

    # Selecció de les columnes desitjades
    ratings = ratings.dropna()
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    # Filtrat de ratings (nota entre 1-5)
    ratings = ratings[(ratings["rating"] >= 0.5) & (ratings["rating"] <= 5.0)]

    return movies, ratings


# ---------------------------------------------------------
# 2. Leave-One-Out per usuari
# ---------------------------------------------------------
def leave_one_out_split(ratings):
    # Escollim un usuari a l'atzar per fer el test
    test = ratings.groupby("userId").sample(1, random_state=42)
    # La resta train
    train = ratings.drop(test.index)
    return train, test


# ---------------------------------------------------------
# 3. Matriu usuari-item
# ---------------------------------------------------------
def build_matrix(ratings):
    # Creem una matriu on les files són usuaris i les columnes pel·lícules
    return ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)


# ---------------------------------------------------------
# 4. Similitud Pearson (User–User)
# ---------------------------------------------------------
def pearson_similarity(vec_a, vec_b):
    # Càlcul de la similitud de Pearson
    mask = (vec_a > 0) & (vec_b > 0)
    if mask.sum() == 0:
        return 0

    a_centered = vec_a[mask] - np.mean(vec_a[mask])
    b_centered = vec_b[mask] - np.mean(vec_b[mask])

    denominator = np.sqrt(np.sum(a_centered ** 2)) * np.sqrt(np.sum(b_centered ** 2))
    if denominator == 0:
        return 0

    return np.sum(a_centered * b_centered) / denominator


# ---------------------------------------------------------
# 5. Predicción User–User
# ---------------------------------------------------------
def predict_user_user(user, item, matrix, k=10):
    sims = []

    # Càlcul de similitud per cada usuari que ha valorat la pel·lícula
    for other in matrix.index:
        if other != user and matrix.loc[other, item] > 0:
            sim = pearson_similarity(matrix.loc[user], matrix.loc[other])
            # Llista de túples amb la similitud, valoració a l'ítem i l'id de l'altre usuari
            sims.append((sim, matrix.loc[other, item], other))

    # Ordenem per similitud i ens quedem amb els k primers
    sims.sort(key=lambda x: x[0], reverse=True)
    sims = sims[:k]

    # Si ningú ha puntuat la pel·lícula, retornem la mitjana de l'usuari (com diu la fòrmula)
    if len(sims) == 0:
        rated = matrix.loc[user][matrix.loc[user] > 0]
        return rated.mean() if len(rated) else 0

    # Càlcul de la fòrmula
    user_mean = np.mean(matrix.loc[user][matrix.loc[user] > 0])

    numerator = sum(sim * (rating - np.mean(matrix.loc[uid][matrix.loc[uid] > 0]))
                    for sim, rating, uid in sims)
    denominator = sum(abs(sim) for sim, _, _ in sims)

    return user_mean + numerator / (denominator + 1e-8)


# ---------------------------------------------------------
# 6. Similitud Item–Item
# ---------------------------------------------------------
def adjusted_cosine(i_vec, j_vec):
    # Càlcul de la distància d'adjusted cosine
    mask = (i_vec > 0) & (j_vec > 0) # Màscara on els usuaris han valorat les dues pel·lícules
    
    if mask.sum() == 0:
        return 0

    mean_user = (i_vec[mask] + j_vec[mask]) / 2
    i_c = i_vec[mask] - mean_user
    j_c = j_vec[mask] - mean_user

    denominator = np.sqrt(np.sum(i_c ** 2)) * np.sqrt(np.sum(j_c ** 2))
    if denominator == 0:
        return 0

    return np.sum(i_c * j_c) / denominator


def predict_item_item(user, item, matrix, k=10):
    sims = []

    # Càlcul de similitud per cada pel·lícula que ha valorat l'usuari
    for rated_item in matrix.columns[matrix.loc[user] > 0]:
        sim = adjusted_cosine(matrix[rated_item], matrix[item])
        sims.append((sim, matrix.loc[user, rated_item]))

    sims.sort(key=lambda x: x[0], reverse=True)
    sims = sims[:k]

    numerator = sum(sim * rating for sim, rating in sims)
    denominator = sum(abs(sim) for sim, _ in sims)

    return numerator / (denominator + 1e-8) if denominator > 0 else 0


# ---------------------------------------------------------
# 7. Evaluació RMSE
# ---------------------------------------------------------
def evaluate(test, matrix, mode):
    real, pred = [], []

    # Suma de l'error RMSE
    for _, row in test.iterrows():
        user, item = row.userId, row.movieId

        if user not in matrix.index or item not in matrix.columns:
            continue

        prediction = predict_user_user(user, item, matrix) if mode == "user" \
                     else predict_item_item(user, item, matrix)

        real.append(row.rating)
        pred.append(prediction)

    return np.sqrt(mean_squared_error(real, pred))


# ---------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    path = "./ml_latest_small"

    movies, ratings = load_datasets(path)
    train, test = leave_one_out_split(ratings)
    matrix = build_matrix(train)

    rmse_user = evaluate(test, matrix, "user")
    rmse_item = evaluate(test, matrix, "item")

    print("\n📍 RESULTATS (Leave One Out):")
    print(f"RMSE User–User (Pearson)      = {rmse_user:.4f}")
    print(f"RMSE Item–Item (AdjCosine)    = {rmse_item:.4f}")
