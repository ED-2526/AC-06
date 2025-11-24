import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------
# 1. Cargar y validar datos
# ---------------------------------------------------------
def load_datasets(base_path):
    movies = pd.read_csv(os.path.join(base_path, "movies.csv"))
    ratings = pd.read_csv(os.path.join(base_path, "ratings.csv"))

    required_cols = {"userId", "movieId", "rating"}
    if not required_cols.issubset(ratings.columns):
        raise ValueError("❌ ratings.csv no contiene las columnas obligatorias")

    ratings = ratings.dropna()
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    ratings = ratings[(ratings["rating"] >= 0.5) & (ratings["rating"] <= 5.0)]

    return movies, ratings


# ---------------------------------------------------------
# 2. Leave-One-Out por usuario
# ---------------------------------------------------------
def leave_one_out_split(ratings):
    test = ratings.groupby("userId").sample(1, random_state=42)
    train = ratings.drop(test.index)
    return train, test


# ---------------------------------------------------------
# 3. Matriz usuario-item
# ---------------------------------------------------------
def build_matrix(ratings):
    return ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)


# ---------------------------------------------------------
# 4. Similaridad Pearson (User–User)
# ---------------------------------------------------------
def pearson_similarity(vec_a, vec_b):
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

    for other in matrix.index:
        if other != user and matrix.loc[other, item] > 0:
            sim = pearson_similarity(matrix.loc[user], matrix.loc[other])
            sims.append((sim, matrix.loc[other, item], other))

    sims.sort(key=lambda x: x[0], reverse=True)
    sims = sims[:k]

    if len(sims) == 0:
        rated = matrix.loc[user][matrix.loc[user] > 0]
        return rated.mean() if len(rated) else 0

    user_mean = np.mean(matrix.loc[user][matrix.loc[user] > 0])

    numerator = sum(sim * (rating - np.mean(matrix.loc[uid][matrix.loc[uid] > 0]))
                    for sim, rating, uid in sims)
    denominator = sum(abs(sim) for sim, _, _ in sims)

    return user_mean + numerator / (denominator + 1e-8)


# ---------------------------------------------------------
# 6. Similaridad Item–Item
# ---------------------------------------------------------
def adjusted_cosine(i_vec, j_vec):
    mask = (i_vec > 0) & (j_vec > 0)
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

    for rated_item in matrix.columns[matrix.loc[user] > 0]:
        sim = adjusted_cosine(matrix[rated_item], matrix[item])
        sims.append((sim, matrix.loc[user, rated_item]))

    sims.sort(key=lambda x: x[0], reverse=True)
    sims = sims[:k]

    numerator = sum(sim * rating for sim, rating in sims)
    denominator = sum(abs(sim) for sim, _ in sims)

    return numerator / (denominator + 1e-8) if denominator > 0 else 0


# ---------------------------------------------------------
# 7. Evaluación RMSE
# ---------------------------------------------------------
def evaluate(test, matrix, mode):
    real, pred = [], []

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

    print("\n📍 RESULTADOS (Leave One Out):")
    print(f"RMSE User–User (Pearson)      = {rmse_user:.4f}")
    print(f"RMSE Item–Item (AdjCosine)    = {rmse_item:.4f}")
