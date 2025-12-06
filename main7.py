import pickle
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
    filtered = ratings.copy()
    while True:
        before = len(filtered)
        user_counts = filtered.groupby("userId").size()
        item_counts = filtered.groupby("movieId").size()

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
    test = ratings.groupby("userId").sample(1, random_state=42)
    train = ratings.drop(test.index)
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


# ---------------------------------------------------------
# 4. Similitud Pearson (User–User)
# ---------------------------------------------------------
def pearson_similarity(vec_a, vec_b, mitjana_a, mitjana_b):
    mask = (vec_a > 0) & (vec_b > 0)
    if mask.sum() == 0:
        return 0
    if mask.sum() == 1:
        return 0

    a_centered = vec_a[mask] - mitjana_a
    b_centered = vec_b[mask] - mitjana_b

    denom = np.sqrt(np.sum(a_centered ** 2)) * np.sqrt(np.sum(b_centered ** 2))
    if denom == 0:
        return 0

    return np.sum(a_centered * b_centered) / denom


# ---------------------------------------------------------
# 5. Predicció User–User (Col·laboratiu, Pearson)
# ---------------------------------------------------------
def predict_user_user(user, item, matrix, user_means, k=20):
    sims = []
    for other in matrix.index:
        if other != user and item in matrix.columns and matrix.loc[other, item] > 0:
            sim = pearson_similarity(matrix.loc[user], matrix.loc[other], user_means[user], user_means[other])
            sims.append((sim, matrix.loc[other, item], other))

    sims.sort(key=lambda x: x[0], reverse=True)
    sims = sims[:k]

    mean_user = user_means[user] if user in user_means else 3.0

    if len(sims) == 0:
        return max(1.0, min(5.0, mean_user))

    num, den = 0.0, 0.0
    for sim, rating, uid in sims:
        other_ratings = matrix.loc[uid][matrix.loc[uid] > 0]
        mean_other = float(other_ratings.mean()) if len(other_ratings) else 3.0
        num += sim * (rating - mean_other)
        den += abs(sim)

    prediction = mean_user if den == 0 else mean_user + num / den
    prediction = max(1.0, min(5.0, float(prediction)))

    # Nota: Comento el print per no embrutar el menú, descomentar si es vol debug
    # print(f"ITEM {item} MITJANA USUARI {user_means[user]} PREDICTION {prediction} NUM/DEN {num / den}")
    return prediction


# ---------------------------------------------------------
# 6. Similitud Item–Item (Col·laboratiu, Adjusted Cosine)
# ---------------------------------------------------------
def adjusted_cosine(i_vec, j_vec, user_means):
    mask = (i_vec > 0) & (j_vec > 0)
    if mask.sum() == 0 or mask.sum() == 1:
        return 0

    mean_user = user_means[mask]
    i_c = i_vec[mask] - mean_user
    j_c = j_vec[mask] - mean_user

    denom = np.sqrt(np.sum(i_c ** 2)) * np.sqrt(np.sum(j_c ** 2))
    if denom == 0:
        return 0

    ret = np.sum(i_c * j_c) / denom
    if ret == 1:
        return ret
    return ret


def predict_item_item(user, item, matrix, user_means, k=10):
    sims = []
    user_row = matrix.loc[user]

    if item not in matrix.columns:
        user_ratings = user_row[user_row > 0]
        fallback = float(user_ratings.mean()) if len(user_ratings) else 3.0
        return max(1.0, min(5.0, fallback))

    rated_items = matrix.columns[user_row > 0]

    for rated_item in rated_items:
        sim = adjusted_cosine(matrix[rated_item], matrix[item], user_means)
        sims.append((sim, matrix.loc[user, rated_item]))

    user_ratings = user_row[user_row > 0]

    if not sims:
        fallback = float(user_ratings.mean()) if len(user_ratings) else 3.0
        return max(1.0, min(5.0, fallback))

    sims.sort(key=lambda x: x[0], reverse=True)
    sims = [x for x in sims if 0.0 <= x[0] <= 1.0]
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
def evaluate(test, matrix, mode, user_means=None):
    real, pred = [], []

    for _, row in test.iterrows():
        user, item = int(row.userId), int(row.movieId)

        if user not in matrix.index or item not in matrix.columns:
            continue

        if mode == "user":
            p = predict_user_user(user, item, matrix, user_means)
        elif mode == "item":
            p = predict_item_item(user, item, matrix, user_means)
        else:
            raise ValueError(f"Mode '{mode}' desconegut a evaluate().")

        real.append(float(row.rating))
        pred.append(p)

    return np.sqrt(mean_squared_error(real, pred)) if len(real) else float("nan")


# ---------------------------------------------------------
# 8. Recomanació basada en contingut (gèneres)
# ---------------------------------------------------------
def build_genre_matrix(movies):
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
    profiles = {}
    grouped = train.groupby("userId")

    for user, group in grouped:
        profile = np.zeros(genre_matrix.shape[1], dtype=float)
        for _, r in group.iterrows():
            m_id = int(r["movieId"])
            rating = float(r["rating"])
            if m_id in genre_matrix.index:
                profile += rating * genre_matrix.loc[m_id].values

        if np.linalg.norm(profile) > 0:
            profiles[user] = profile

    return profiles


def predict_content_based(user, item, user_profiles, genre_matrix, user_means):
    base = float(user_means.get(user, 3.0))
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
    genre_matrix = build_genre_matrix(movies)
    user_profiles = build_user_profiles(train, genre_matrix)
    user_means = train.groupby("userId")["rating"].mean()

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
    if user not in matrix.index:
        raise ValueError(f"L'usuari {user} no és a la matriu de ratings.")

    seen = set(matrix.loc[user][matrix.loc[user] > 0].index.astype(int))
    all_movies = set(movies["movieId"].astype(int).values)
    candidates = list(all_movies - seen)

    id_to_title = movies.set_index("movieId")["title"].to_dict()
    recs = []

    for m in candidates:
        if mode == "user":
            if m not in matrix.columns:
                continue
            p = predict_user_user(user, m, matrix, user_means, k=k)
        elif mode == "item":
            if m not in matrix.columns:
                continue
            p = predict_item_item(user, m, matrix, user_means, k=k)
        elif mode == "content":
            if user_profiles is None or genre_matrix is None or user_means is None:
                raise ValueError("Error intern: Falten dades per content-based.")
            p = predict_content_based(user, m, user_profiles, genre_matrix, user_means)
        else:
            continue

        recs.append((m, id_to_title.get(m, "Títol desconegut"), p))

    recs.sort(key=lambda x: x[2], reverse=True)
    return recs[:n_recs]



# ---------------------------------------------------------
# 10. GESTIÓ DE CACHE I MENÚ
# ---------------------------------------------------------

CACHE_FILE = "recsys_data.pkl"


def save_data_to_pickle(data):
    """Guarda el diccionari de dades a un fitxer pickle."""
    print(f"💾 Guardant dades processades a '{CACHE_FILE}' per a futurs usos...")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)
    print("✅ Dades guardades correctament.")


def load_data_from_pickle():
    """Carrega les dades del fitxer pickle."""
    print(f"📂 Carregant dades des de '{CACHE_FILE}'...")
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


def compute_all_data(base_path):
    """
    Executa tot el pipeline de càrrega i processament (lent).
    Retorna un diccionari amb totes les variables necessàries.
    """
    print("⚙️  Processant dades des de zero (això pot trigar una mica)...")

    # 1. Càrrega
    movies, ratings = load_datasets(base_path)

    # 2. Filtratge
    k = 5
    ratings_filtered = filter_ratings_by_min_counts(ratings, k_user=k, k_item=k)

    # 3. Split i Matrius Col·laboratives
    train, test = leave_one_out_split(ratings_filtered)
    matrix = build_matrix(train)
    user_means = train.groupby("userId")["rating"].mean()

    # 4. Estructures Content-Based
    genre_matrix = build_genre_matrix(movies)
    user_profiles = build_user_profiles(train, genre_matrix)

    # Empaquetem tot en un diccionari
    data = {
        "movies": movies,
        "train": train,
        "test": test,
        "matrix": matrix,
        "user_means": user_means,
        "genre_matrix": genre_matrix,
        "user_profiles": user_profiles
    }

    return data


def input_user_id(matrix):
    while True:
        try:
            uid = int(input("Introdueix l'ID de l'usuari: "))
            if uid in matrix.index:
                return uid
            else:
                print(f"❌ L'usuari {uid} no és a la matriu. Prova'n un altre.")
        except ValueError:
            print("⚠️ Introdueix un número vàlid.")


def input_movie_id(movies_df):
    while True:
        try:
            mid = int(input("Introdueix l'ID de la pel·lícula: "))
            if mid in movies_df["movieId"].values:
                return mid
            else:
                print(f"❌ La pel·lícula {mid} no existeix.")
        except ValueError:
            print("⚠️ Introdueix un número vàlid.")


def select_algorithm():
    while True:
        print("\n   [ Algorisme ]")
        print("   u. User-User (Pearson)")
        print("   i. Item-Item (Adjusted Cosine)")
        print("   c. Content-Based (Gèneres)")
        choice = input("   Selecciona (u/i/c): ").lower().strip()
        if choice in ['u', 'i', 'c']:
            return {'u': 'user', 'i': 'item', 'c': 'content'}[choice]
        print("⚠️ Opció no vàlida.")


def main_menu():
    path = "./ml_latest_small"

    # LÒGICA DE CÀRREGA INTEL·LIGENT
    if os.path.exists(CACHE_FILE):
        print(f"ℹ️  S'ha detectat un fitxer de cache '{CACHE_FILE}'.")
        use_cache = input("Vols carregar les dades ràpidament? (S/n): ").lower().strip()
        if use_cache != 'n':
            try:
                data = load_data_from_pickle()
            except Exception as e:
                print(f"⚠️ Error llegint el pickle ({e}). Es recalcularà tot.")
                data = compute_all_data(path)
                save_data_to_pickle(data)
        else:
            data = compute_all_data(path)
            save_data_to_pickle(data)
    else:
        # No existeix cache, calculem i guardem
        if not os.path.exists(path):
            print(f"❌ Error: No s'ha trobat el directori '{path}'.")
            return
        data = compute_all_data(path)
        save_data_to_pickle(data)

    # Desempaquetem les dades per fer-les servir fàcilment
    movies = data["movies"]
    # train = data["train"] # No l'usem directament al menú, però està guardat
    test = data["test"]
    matrix = data["matrix"]
    user_means = data["user_means"]
    genre_matrix = data["genre_matrix"]
    user_profiles = data["user_profiles"]

    print(f"✅ Sistema llest! Usuaris: {matrix.shape[0]}, Pel·lícules: {matrix.shape[1]}")

    while True:
        print("\n" + "=" * 40)
        print("       SISTEMA DE RECOMANACIÓ")
        print("=" * 40)
        print("1. Predir una valoració")
        print("2. Suggerir rànquing (Top-N)")
        print("3. Avaluar model (Mètriques)")
        print("4. Forçar recàrrega de dades (esborrar cache)")
        print("5. Sortir")
        print("-" * 40)

        opcio = input("Selecciona una opció: ").strip()

        if opcio == "1":
            user_id = input_user_id(matrix)
            movie_id = input_movie_id(movies)
            algo = select_algorithm()

            titol_row = movies[movies.movieId == movie_id]['title']
            titol = titol_row.values[0] if not titol_row.empty else "Desconegut"

            print(f"\n🔮 Predint '{titol}'...")
            try:
                if algo == "user":
                    pred = predict_user_user(user_id, movie_id, matrix, user_means)
                elif algo == "item":
                    pred = predict_item_item(user_id, movie_id, matrix, user_means)
                elif algo == "content":
                    pred = predict_content_based(user_id, movie_id, user_profiles, genre_matrix, user_means)
                print(f"⭐ Predicció: {pred:.4f}")
            except Exception as e:
                print(f"❌ Error: {e}")

        elif opcio == "2":
            user_id = input_user_id(matrix)
            algo = select_algorithm()
            try:
                n = int(input("   Quantes pel·lícules? (Default 10): ") or 10)
            except:
                n = 10

            print(f"\n🔎 Buscant recomanacions...")
            recs = recommend(user_id, matrix, movies, mode=algo, n_recs=n,
                             user_means=user_means, user_profiles=user_profiles, genre_matrix=genre_matrix)

            print(f"\n🎬 TOP {n} RECOMANACIONS:")
            for i, (mid, title, score) in enumerate(recs, 1):
                print(f"{i}. [{score:.2f}] {title}")

        elif opcio == "3":
            print("\n📊 AVALUANT MODELS...")
            print(f"   RMSE User-User:     {evaluate(test, matrix, 'user', user_means):.4f}")
            print(f"   RMSE Item-Item:     {evaluate(test, matrix, 'item', user_means):.4f}")
            print(f"   RMSE Content-Based: {evaluate_content_based(movies, data['train'], test):.4f}")

        elif opcio == "4":
            print("\n🔄 Recalculant dades des dels CSV...")
            data = compute_all_data(path)
            save_data_to_pickle(data)
            # Actualitzem les variables locals
            movies, test, matrix = data["movies"], data["test"], data["matrix"]
            user_means, genre_matrix, user_profiles = data["user_means"], data["genre_matrix"], data["user_profiles"]
            print("✅ Dades actualitzades.")

        elif opcio == "5":
            print("Adéu! 👋")
            break

        else:
            print("⚠️ Opció no reconeguda.")


if __name__ == "__main__":
    main_menu()
