import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import time
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import seaborn as sns
import ast # Per processar els gèneres del dataset de Kaggle

# =========================================================
# 0. GESTIÓ DE CONFIGURACIÓ I CACHE (NOU)
# =========================================================

# Diccionari de datasets disponibles
# ASSEGURA'T DE POSAR LA RUTA CORRECTA ON TINGUIS EL DATASET DE KAGGLE
DATASETS_CONFIG = {
    "1": {"name": "small", "path": "./ml_latest_small", "desc": "MovieLens Small (Recomanat per proves)"},
    "2": {"name": "kaggle_large", "path": "./the_movies_dataset", "desc": "Kaggle Full Dataset (Pot necessitar molta RAM)"}
}

class Config:
    """Classe per guardar l'estat de la configuració actual"""
    def __init__(self):
        self.dataset_name = "small"
        self.base_path = "./ml_latest_small"
        self.k_filter = 5
        self.cache_file = "recsys_data_small_k5.pkl"

    def update(self, dataset_key, k):
        ds = DATASETS_CONFIG.get(dataset_key)
        if ds:
            self.dataset_name = ds["name"]
            self.base_path = ds["path"]
            self.k_filter = k
            # El nom del fitxer depèn del dataset i de la K. Així no es barregen.
            self.cache_file = f"recsys_data_{self.dataset_name}_k{self.k_filter}.pkl"
            return True
        return False

# Instància global de configuració
CURRENT_CONFIG = Config()

# =========================================================
# 1. FUNCIONS DE BASE I DATA WRANGLING (ADAPTAT PER KAGGLE)
# =========================================================

def parse_genres_kaggle(genres_str):
    """Converteix el string JSON de Kaggle [{'name': 'Action'}] a format 'Action|Adventure'"""
    try:
        if pd.isna(genres_str): return "(no genres listed)"
        genres_list = ast.literal_eval(genres_str)
        names = [g['name'] for g in genres_list]
        return "|".join(names)
    except:
        return "(no genres listed)"

def load_datasets(base_path):
    print(f"📂 Llegint fitxers des de: {base_path}")
    
    # 1. Càrrega de MOVIES (Detectem si és format estàndard o Kaggle)
    if os.path.exists(os.path.join(base_path, "movies.csv")):
        # Format Estàndard MovieLens
        movies = pd.read_csv(os.path.join(base_path, "movies.csv"), dtype={'movieId': str})
    
    elif os.path.exists(os.path.join(base_path, "movies_metadata.csv")):
        # Format Kaggle
        print("   -> Detectat format Kaggle (movies_metadata.csv). Adaptant...")
        movies = pd.read_csv(os.path.join(base_path, "movies_metadata.csv"), low_memory=False)
        
        # Neteja bàsica per Kaggle
        movies = movies.rename(columns={'id': 'movieId', 'original_title': 'title'})
        movies = movies[['movieId', 'title', 'genres']] # Ens quedem el que interessa
        # Els IDs a vegades tenen errors al dataset de Kaggle, netegem no numèrics
        movies = movies[pd.to_numeric(movies['movieId'], errors='coerce').notnull()]
        movies['genres'] = movies['genres'].apply(parse_genres_kaggle)
    else:
        raise FileNotFoundError(f"No s'ha trobat 'movies.csv' ni 'movies_metadata.csv' a {base_path}")

    # Assegurem tipus correctes
    movies['movieId'] = movies['movieId'].astype(str)

    # 2. Càrrega de RATINGS
    if not os.path.exists(os.path.join(base_path, "ratings.csv")):
        raise FileNotFoundError(f"No s'ha trobat 'ratings.csv' a {base_path}")
        
    ratings = pd.read_csv(os.path.join(base_path, "ratings.csv"))
    
    # Validació columnes
    required_cols = {"userId", "movieId", "rating"}
    if not required_cols.issubset(ratings.columns):
        raise ValueError("ratings.csv no conté les columnes obligatòries")

    ratings = ratings.dropna()
    # Convertim IDs a string per garantir merge correcte entre els dos datasets
    ratings["movieId"] = ratings["movieId"].astype(str)
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    # Garantim ratings vàlids
    ratings = ratings[(ratings["rating"] >= 0.5) & (ratings["rating"] <= 5.0)]

    # 3. Sincronització: Només ratings de pel·lícules que tenim
    # (El dataset de Kaggle té IDs que no quadren a vegades, això ho neteja)
    valid_movie_ids = set(movies['movieId'].unique())
    initial_len = len(ratings)
    ratings = ratings[ratings['movieId'].isin(valid_movie_ids)]
    if len(ratings) < initial_len:
        print(f"   ⚠️ S'han descartat {initial_len - len(ratings)} ratings que no tenien pel·lícula corresponent.")

    return movies, ratings

def filter_ratings_by_min_counts(ratings, k_user, k_item):
    """Filtra usuaris i items amb menys de k valoracions."""
    filtered = ratings.copy()
    print(f"   -> Filtrant dades (min {k_user} vots/usuari, min {k_item} vots/item)...")
    
    while True:
        before = len(filtered)
        user_counts = filtered.groupby("userId").size()
        item_counts = filtered.groupby("movieId").size()
        
        valid_users = user_counts[user_counts >= k_user].index
        valid_items = item_counts[item_counts >= k_item].index
        
        filtered = filtered[filtered["userId"].isin(valid_users) & filtered["movieId"].isin(valid_items)]
        
        after = len(filtered)
        if after == before:
            break
            
    print(f"      - Inicial: {len(ratings)} -> Final: {len(filtered)} ratings.")
    return filtered

def leave_one_out_split(ratings):
    test = ratings.groupby("userId").sample(1, random_state=42)
    train = ratings.drop(test.index)
    return train, test

def build_matrix(ratings):
    return ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)

# =========================================================
# 2. FUNCIONS DE SIMILITUD MANUALS (PER AL MODE LENT)
# =========================================================

def pearson_similarity_manual(vec_a, vec_b, mitjana_a, mitjana_b):
    mask = (vec_a > 0) & (vec_b > 0)
    if mask.sum() <= 1: return 0.0

    ratings_a, ratings_b = vec_a[mask], vec_b[mask]
    centered_a = ratings_a - mitjana_a
    centered_b = ratings_b - mitjana_b

    numerator = np.dot(centered_a, centered_b)
    denominator = np.linalg.norm(centered_a) * np.linalg.norm(centered_b)

    return float(numerator / denominator) if denominator != 0 else 0.0

def adjusted_cosine_manual(item_a_ratings, item_b_ratings, user_means_series):
    mask = (item_a_ratings > 0) & (item_b_ratings > 0)
    if sum(mask) <= 1: return 0.0

    ratings_a = item_a_ratings.loc[mask]
    ratings_b = item_b_ratings.loc[mask]
    means = user_means_series.loc[mask]

    centered_a = ratings_a - means
    centered_b = ratings_b - means

    numerator = np.dot(centered_a.values, centered_b.values)
    denominator = np.linalg.norm(centered_a.values) * np.linalg.norm(centered_b.values)

    return float(numerator / denominator) if denominator != 0 else 0.0

# =========================================================
# 3. FUNCIONS DE PRE-CÀLCUL
# =========================================================

def compute_similarity_matrices(matrix):
    print("   ... calculant matriu de similitud User-User (Pearson)...")
    matrix_nan = matrix.replace(0, np.nan)
    sim_user_df = matrix_nan.T.corr(method='pearson')

    print("   ... calculant matriu de similitud Item-Item (Adjusted Cosine)...")
    user_means = matrix_nan.mean(axis=1)
    matrix_centered = matrix_nan.sub(user_means, axis=0).fillna(0)

    sim_item_matrix = cosine_similarity(matrix_centered.T)  # Retorna una matriu numpy
    sim_item_df = pd.DataFrame(sim_item_matrix, index=matrix.columns, columns=matrix.columns)  # La transformem a pandas

    return sim_user_df, sim_item_df

def compute_similarity_matrices_manual(matrix, user_means_series):
    print("\n   ... INICIANT CÀLCUL MANUAL (Pot trigar MINUTS)...")
    start_time = time.time()
    user_ids = matrix.index
    movie_ids = matrix.columns

    # 1. User-User Manual
    print("   ... calculant matriu User-User manualment...")
    sim_user_matrix = np.zeros((len(user_ids), len(user_ids)))
    for i in range(len(user_ids)):
        u_i = user_ids[i]
        ratings_i = matrix.loc[u_i]
        mean_i = user_means_series.loc[u_i]
        for j in range(i, len(user_ids)):
            u_j = user_ids[j]
            ratings_j = matrix.loc[u_j]
            mean_j = user_means_series.loc[u_j]
            similarity = pearson_similarity_manual(ratings_i, ratings_j, mean_i, mean_j)
            sim_user_matrix[i, j] = similarity
            sim_user_matrix[j, i] = similarity
    sim_user_df = pd.DataFrame(sim_user_matrix, index=user_ids, columns=user_ids)

    # 2. Item-Item Manual
    print("   ... calculant matriu Item-Item manualment...")
    sim_item_matrix = np.zeros((len(movie_ids), len(movie_ids)))
    for i in range(len(movie_ids)):
        m_i = movie_ids[i]
        ratings_i = matrix[m_i]
        for j in range(i, len(movie_ids)):
            m_j = movie_ids[j]
            ratings_j = matrix[m_j]
            similarity = adjusted_cosine_manual(ratings_i, ratings_j, user_means_series)
            sim_item_matrix[i, j] = similarity
            sim_item_matrix[j, i] = similarity
    sim_item_df = pd.DataFrame(sim_item_matrix, index=movie_ids, columns=movie_ids)

    print(f"   ... CÀLCUL MANUAL FINALITZAT en {time.time() - start_time:.2f} segons.")
    return sim_user_df, sim_item_df

# =========================================================
# 4. FUNCIONS FUNK SVD (NOVES)
# =========================================================

def train_funk_svd(matrix, n_factors=20, n_epochs=20, lr=0.005, reg=0.02, verbose=True):
    """
    Entrena un model Funk SVD sobre la matriu user-item (només valors > 0).
    """
    R = matrix.values.astype(float)
    num_users, num_items = R.shape

    # Factors inicialitzats aleatòriament
    P = 0.1 * np.random.randn(num_users, n_factors)
    Q = 0.1 * np.random.randn(num_items, n_factors)

    user_index = {u: idx for idx, u in enumerate(matrix.index)}
    item_index = {i: idx for idx, i in enumerate(matrix.columns)}

    # Llista de mostres (només ratings existents)
    rows, cols = np.where(R > 0)
    samples = list(zip(rows, cols))

    if len(samples) == 0:
        return {"P": P, "Q": Q, "user_index": user_index, "item_index": item_index}

    for epoch in range(n_epochs):
        np.random.shuffle(samples)
        for u_idx, i_idx in samples:
            r_ui = R[u_idx, i_idx]
            pred = np.dot(P[u_idx], Q[i_idx])
            err = r_ui - pred

            # Actualització gradient descendent
            P[u_idx] += lr * (err * Q[i_idx] - reg * P[u_idx])
            Q[i_idx] += lr * (err * P[u_idx] - reg * Q[i_idx])

        if verbose:
            # Càlcul ràpid de l'error en entrenament per mostrar progrés
            mse = 0.0
            for u_idx, i_idx in samples:
                r_ui = R[u_idx, i_idx]
                pred = np.dot(P[u_idx], Q[i_idx])
                mse += (r_ui - pred) ** 2
            mse /= len(samples)
            print(f"   [FunkSVD] Epoch {epoch + 1}/{n_epochs} - RMSE train: {np.sqrt(mse):.4f}")

    return {
        "P": P,
        "Q": Q,
        "user_index": user_index,
        "item_index": item_index
    }

def predict_funk_svd(user, item, funk_model, user_means):
    """
    Predicció d'una valoració amb Funk SVD.
    """
    if funk_model is None:
        return user_means.get(user, 3.0)

    u_idx = funk_model["user_index"].get(user)
    i_idx = funk_model["item_index"].get(item)

    if u_idx is None or i_idx is None:
        return user_means.get(user, 3.0)

    P = funk_model["P"]
    Q = funk_model["Q"]

    pred = float(np.dot(P[u_idx], Q[i_idx]))
    # Acotem a [1, 5]
    return max(1.0, min(5.0, pred))

def evaluate_funk_svd(test, matrix, user_means, funk_model):
    """
    Calcula el RMSE sobre el test utilitzant un model Funk SVD ja entrenat.
    """
    if funk_model is None: return float("nan")

    real, pred = [], []
    for _, row in test.iterrows():
        user, item = int(row.userId), int(row.movieId)
        # SVD pot predir encara que l'usuari no estigui a la matriu original si tenim mitjanes,
        # però per consistència comprovem índexs.
        p = predict_funk_svd(user, item, funk_model, user_means)
        real.append(float(row.rating))
        pred.append(p)

    return np.sqrt(mean_squared_error(real, pred)) if len(real) else float("nan")

# =========================================================
# 5. FUNCIONS DE PREDICCIÓ (LLIBRERIA I CONTENT)
# =========================================================

def predict_user_user_fast(user, item, matrix, user_means, sim_user_df, k=10):
    if item not in matrix.columns or user not in sim_user_df.index: return user_means.get(user, 3.0)
    sim_series = sim_user_df.loc[user]
    rated_users_mask = matrix[item] > 0
    valid_sims = sim_series[rated_users_mask & (sim_series.index != user)]
    valid_sims = valid_sims[valid_sims > 0]
    top_k = valid_sims.nlargest(k)
    if top_k.empty: return user_means.get(user, 3.0)

    num, den = 0.0, 0.0
    for other_user, sim_val in top_k.items():
        rating_other = matrix.loc[other_user, item]
        mean_other = user_means[other_user]
        num += sim_val * (rating_other - mean_other)
        den += abs(sim_val)

    mean_user = user_means.get(user, 3.0)
    pred = mean_user + num / den if den != 0 else mean_user
    return max(1.0, min(5.0, pred))

def predict_item_item_fast(user, item, matrix, user_means, sim_item_df, k=10):
    if item not in sim_item_df.index or user not in matrix.index: return user_means.get(user, 3.0)
    user_row = matrix.loc[user]
    rated_items = user_row[user_row > 0].index
    sims = sim_item_df.loc[item, rated_items]
    sims = sims[sims > 0]
    top_k = sims.nlargest(k)
    if top_k.empty: return user_means.get(user, 3.0)

    num, den = 0.0, 0.0
    for rated_item_id, sim_val in top_k.items():
        rating = matrix.loc[user, rated_item_id]
        num += sim_val * rating
        den += abs(sim_val)
    pred = num / den if den != 0 else user_means.get(user, 3.0)
    return max(1.0, min(5.0, pred))

def build_genre_matrix(movies):
    all_genres = set()
    for gen_str in movies["genres"].fillna(""):
        for g in str(gen_str).split("|"):
            if g and g != "(no genres listed)": all_genres.add(g)
    genre_matrix = pd.DataFrame(0, index=movies["movieId"].astype(int), columns=sorted(all_genres), dtype=int)
    for _, row in movies.iterrows():
        m_id = int(row["movieId"])
        for g in str(row["genres"]).split("|"):
            if g in genre_matrix.columns: genre_matrix.at[m_id, g] = 1
    return genre_matrix

def build_user_profiles(train, genre_matrix):
    profiles = {}
    grouped = train.groupby("userId")
    for user, group in grouped:
        profile = np.zeros(genre_matrix.shape[1], dtype=float)
        for _, r in group.iterrows():
            m_id = int(r["movieId"])
            if m_id in genre_matrix.index:
                profile += float(r["rating"]) * genre_matrix.loc[m_id].values
        if np.linalg.norm(profile) > 0: profiles[user] = profile
    return profiles

def predict_content_based(user, item, user_profiles, genre_matrix, user_means):
    base = user_means.get(user, 3.0)
    if item not in genre_matrix.index or user not in user_profiles: return base
    user_vec, movie_vec = user_profiles[user], genre_matrix.loc[item].values
    num = np.dot(user_vec, movie_vec)
    den = np.linalg.norm(user_vec) * np.linalg.norm(movie_vec)
    if den == 0:
        pred = base
    else:
        sim = num / den
        pred = base + sim * (5.0 - base)
    return max(1.0, min(5.0, float(pred)))

# =========================================================
# 6. FUNCIONS WRAPPER I RECOMANACIÓ
# =========================================================

def get_similarity_matrix_for_mode(mode_alg, mode_sim, data):
    matrix = None
    is_stored = False

    if mode_alg == 'user':
        if mode_sim == 'library':
            matrix = data.get("sim_user_df_fast")
            is_stored = True
        elif mode_sim == 'manual':
            matrix = data.get("sim_user_df_manual")
            is_stored = matrix is not None

    elif mode_alg == 'item':
        if mode_sim == 'library':
            matrix = data.get("sim_item_df_fast")
            is_stored = True
        elif mode_sim == 'manual':
            matrix = data.get("sim_item_df_manual")
            is_stored = matrix is not None

    return matrix, is_stored

def predict_single_rating(user, item, mode_alg, mode_sim, data, k=10):
    matrix = data["matrix"]
    user_means = data["user_means"]

    if mode_alg == 'content':
        return predict_content_based(user, item, data["user_profiles"], data["genre_matrix"], user_means)

    # 4. Funk SVD
    if mode_alg == 'funk':
        # Fem servir el model per defecte guardat al pickle
        return predict_funk_svd(user, item, data.get("funk_model"), user_means)

    # Col·laboratiu (User/Item)
    sim_df, is_stored = get_similarity_matrix_for_mode(mode_alg, mode_sim, data)

    # CAS A: Tenim matriu precalculada
    if is_stored and sim_df is not None:
        if mode_alg == 'user':
            return predict_user_user_fast(user, item, matrix, user_means, sim_df, k)
        elif mode_alg == 'item':
            return predict_item_item_fast(user, item, matrix, user_means, sim_df, k)

    # CAS B: Manual On-the-fly (per un sol punt)
    else:
        if mode_alg == 'user':
            target_vec = matrix.loc[user]
            mean_user = user_means.get(user, 3.0)
            others = matrix.index[matrix[item] > 0]
            sims = []
            for other in others:
                if other == user: continue
                other_vec = matrix.loc[other]
                mean_other = user_means.get(other, 3.0)
                sim = pearson_similarity_manual(target_vec, other_vec, mean_user, mean_other)
                if sim > 0:
                    sims.append((sim, matrix.loc[other, item], other))

            sims.sort(key=lambda x: x[0], reverse=True)
            top_k = sims[:k]
            if not top_k: return mean_user
            num, den = 0.0, 0.0
            for s, r, uid in top_k:
                mean_other = user_means[uid]
                num += s * (r - mean_other)
                den += abs(s)
            return mean_user + num / den if den != 0 else mean_user

        elif mode_alg == 'item':
            user_row = matrix.loc[user]
            rated_items = user_row[user_row > 0].index
            target_item_vec = matrix[item]
            sims = []
            for r_item in rated_items:
                r_item_vec = matrix[r_item]
                sim = adjusted_cosine_manual(target_item_vec, r_item_vec, user_means)
                if sim > 0:
                    sims.append((sim, user_row.loc[r_item]))
            sims.sort(key=lambda x: x[0], reverse=True)
            top_k = sims[:k]
            if not top_k: return user_means.get(user, 3.0)
            num = sum(s * r for s, r in top_k)
            den = sum(abs(s) for s, _ in top_k)
            return num / den if den != 0 else user_means.get(user, 3.0)
    return 0.0

def recommend(user, matrix, movies, mode, sim_mode, k=10, n_recs=5, data=None):
    if user not in matrix.index: return []

    user_means = data["user_means"]
    user_row = matrix.loc[user]
    seen_items = set(user_row[user_row > 0].index)
    candidates = [m for m in matrix.columns if m not in seen_items]
    id_to_title = movies.set_index("movieId")["title"].to_dict()
    recs = []

    # 1. Funk SVD (Tractament especial)
    if mode == 'funk':
        print(f"\n[Mode Funk SVD] Calculant prediccions...")
        start_time = time.time()
        funk_model = data.get("funk_model")
        for m in candidates:
            p = predict_funk_svd(user, m, funk_model, user_means)
            recs.append((m, id_to_title.get(m, "Títol desconegut"), p))
        print(f"⏱️ Temps de càlcul: {time.time() - start_time:.4f} segons.")
        recs.sort(key=lambda x: x[2], reverse=True)
        return recs[:n_recs]

    # Determinem mode col·laboratiu
    is_manual_mode = sim_mode == 'manual' and mode in ['user', 'item']
    manual_stored_exists = is_manual_mode and (
        data.get("sim_user_df_manual") is not None if mode == 'user' else data.get("sim_item_df_manual") is not None)

    # RUTA 1: CÀLCUL MANUAL 'ON-THE-FLY'
    if is_manual_mode and not manual_stored_exists:
        print("\n[Mode Manual ON-THE-FLY] Calculant similituds per a CADA predicció (MOLT LENT)...")
        start_time = time.time()
        target_vec = user_row

        for item in candidates:
            p, num, den = 0.0, 0.0, 0.0
            if mode == 'user':
                mean_user = user_means.get(user, 3.0)
                for other_user, other_user_ratings in matrix.iterrows():
                    if other_user == user: continue
                    if other_user_ratings.get(item, 0) > 0:
                        mean_other = user_means.get(other_user, 3.0)
                        sim_val = pearson_similarity_manual(target_vec, other_user_ratings, mean_user, mean_other)
                        if sim_val > 0:
                            num += sim_val * (other_user_ratings.loc[item] - mean_other)
                            den += abs(sim_val)
                p = mean_user + num / den if den != 0 else mean_user

            elif mode == 'item':
                rated_items = user_row[user_row > 0].index
                item_ratings = matrix[item]
                for rated_item in rated_items:
                    rated_item_ratings = matrix[rated_item]
                    sim_val = adjusted_cosine_manual(item_ratings, rated_item_ratings, user_means)
                    if sim_val > 0:
                        num += sim_val * user_row.loc[rated_item]
                        den += abs(sim_val)
                p = num / den if den != 0 else user_means.get(user, 3.0)

            recs.append((item, id_to_title.get(item, "Títol desconegut"), max(1.0, min(5.0, float(p)))))

        print(f"⏱️ Temps de càlcul manual ON-THE-FLY: {time.time() - start_time:.2f} segons.")
        recs.sort(key=lambda x: x[2], reverse=True)
        return recs[:n_recs]

    # RUTA 2: MODES RÀPIDS (Llibreria FAST o Manual STORED) / Content Based
    else:
        sim_df = None
        if mode != 'content':
            # Obtenim la matriu adequada
            sim_df, is_stored = get_similarity_matrix_for_mode(mode, sim_mode, data)
            msg_type = "Manual STORED" if (sim_mode == 'manual' and is_stored) else "Llibreria FAST"
            print(f"\n[Mode {msg_type}] Calculant prediccions (RÀPID)...")
        else:
            print(f"\n[Mode Content Based] Calculant prediccions...")

        user_profiles = data.get("user_profiles")
        genre_matrix = data.get("genre_matrix")
        start_time = time.time()

        for m in candidates:
            if mode == "user":
                p = predict_user_user_fast(user, m, matrix, user_means, sim_df, k=k)
            elif mode == "item":
                p = predict_item_item_fast(user, m, matrix, user_means, sim_df, k=k)
            elif mode == "content":
                p = predict_content_based(user, m, user_profiles, genre_matrix, user_means)
            recs.append((m, id_to_title.get(m, "Títol desconegut"), p))

        print(f"⏱️ Temps de càlcul RÀPID: {time.time() - start_time:.4f} segons.")
        recs.sort(key=lambda x: x[2], reverse=True)
        return recs[:n_recs]

# =========================================================
# 7. AVALUACIÓ I GRÀFICS
# =========================================================

def evaluate(test, matrix, mode, user_means, sim_df, k=10):
    real, pred = [], []
    for _, row in test.iterrows():
        user, item = int(row.userId), int(row.movieId)
        if user not in matrix.index or item not in matrix.columns: continue
        if mode == "user":
            p = predict_user_user_fast(user, item, matrix, user_means, sim_df, k)
        elif mode == "item":
            p = predict_item_item_fast(user, item, matrix, user_means, sim_df, k)
        real.append(float(row.rating))
        pred.append(p)
    return np.sqrt(mean_squared_error(real, pred)) if len(real) else float("nan")

def evaluate_content_based(movies, train, test):
    genre_matrix = build_genre_matrix(movies)
    user_profiles = build_user_profiles(train, genre_matrix)
    user_means = train.groupby("userId")["rating"].mean()
    real, pred = [], []
    for _, row in test.iterrows():
        p = predict_content_based(int(row["userId"]), int(row["movieId"]), user_profiles, genre_matrix, user_means)
        real.append(float(row["rating"]))
        pred.append(p)
    return np.sqrt(mean_squared_error(real, pred)) if len(real) else float("nan")

def generate_metrics_and_plots(movies, train, test, matrix, user_means, sim_user_df_fast, sim_item_df_fast,
                               sim_user_df_manual, sim_item_df_manual):
    """
    Genera mètriques (RMSE) i gràfics per comparar tots els algorismes.
    """
    print("\n📊 Generant mètriques i gràfics d'avaluació...")
    print("   (Això pot trigar una mica perquè s'entrenen diversos models)")

    # 1. RMSE Global
    print("   -> Calculant RMSE global per cada algorisme...")

    rmse_user_fast = evaluate(test, matrix, 'user', user_means, sim_user_df_fast, k=20)
    rmse_item_fast = evaluate(test, matrix, 'item', user_means, sim_item_df_fast, k=20)
    rmse_content = evaluate_content_based(movies, train, test)

    funk_model_default = train_funk_svd(matrix, n_factors=20, n_epochs=15, verbose=False)
    rmse_funk_default = evaluate_funk_svd(test, matrix, user_means, funk_model_default)

    rmse_user_manual, rmse_item_manual = float('nan'), float('nan')
    if sim_user_df_manual is not None:
        rmse_user_manual = evaluate(test, matrix, 'user', user_means, sim_user_df_manual, k=20)
    if sim_item_df_manual is not None:
        rmse_item_manual = evaluate(test, matrix, 'item', user_means, sim_item_df_manual, k=20)

    print(f"   RMSE User-User (Fast k=20):  {rmse_user_fast:.4f}")
    if sim_user_df_manual is not None:
        print(f"   RMSE User-User (Manual k=20): {rmse_user_manual:.4f}")
    print(f"   RMSE Item-Item (Fast k=20):  {rmse_item_fast:.4f}")
    if sim_item_df_manual is not None:
        print(f"   RMSE Item-Item (Manual k=20): {rmse_item_manual:.4f}")
    print(f"   RMSE Content-Based:          {rmse_content:.4f}")
    print(f"   RMSE Funk-SVD (f=20):        {rmse_funk_default:.4f}")

    # Barplot
    print("\n   -> Gràfic comparació RMSE Similitud Manual vs Fast (k=20)...")
    comp_names = []
    comp_rmse = []
    comp_colors = []

    if sim_user_df_manual is not None:
        comp_names.extend(["User-User (Fast)", "User-User (Manual)"])
        comp_rmse.extend([rmse_user_fast, rmse_user_manual])
        comp_colors.extend(['skyblue', 'darkblue'])
    else:
        comp_names.extend(["User-User (Fast)"])
        comp_rmse.extend([rmse_user_fast])
        comp_colors.extend(['skyblue'])

    if sim_item_df_manual is not None:
        comp_names.extend(["Item-Item (Fast)", "Item-Item (Manual)"])
        comp_rmse.extend([rmse_item_fast, rmse_item_manual])
        comp_colors.extend(['lightgreen', 'darkgreen'])
    else:
        comp_names.extend(["Item-Item (Fast)"])
        comp_rmse.extend([rmse_item_fast])
        comp_colors.extend(['lightgreen'])

    comp_names.extend(["Content-Based", "Funk-SVD"])
    comp_rmse.extend([rmse_content, rmse_funk_default])
    comp_colors.extend(['orange', 'salmon'])

    if comp_names:
        plt.figure(figsize=(8, 5))
        plt.title("RMSE (k=20): Similitud Ràpida (Llibreria) vs Manual")
        bars = plt.bar(comp_names, comp_rmse, color=comp_colors)
        plt.ylabel("RMSE")
        plt.xticks(rotation=45, ha='right')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 4), ha='center', va='bottom')
        plt.tight_layout()
        plt.show()

        # Sweep de k
        print("\n   -> Sweep de k (veïns) per User-User i Item-Item (mode Fast i Manual)...")
        k_list = [5, 10, 20, 40, 60, 80]

        rmse_user_k_fast = [evaluate(test, matrix, 'user', user_means, sim_user_df_fast, k=k) for k in k_list]
        rmse_item_k_fast = [evaluate(test, matrix, 'item', user_means, sim_item_df_fast, k=k) for k in k_list]

        if sim_user_df_manual is not None and sim_item_df_manual is not None:
            rmse_user_k_manual = [evaluate(test, matrix, 'user', user_means, sim_user_df_manual, k=k) for k in k_list]
            rmse_item_k_manual = [evaluate(test, matrix, 'item', user_means, sim_item_df_manual, k=k) for k in k_list]
            has_manual_sweep = True
        else:
            has_manual_sweep = False

        plt.figure(figsize=(10, 6))
        plt.title("RMSE vs k (veïns) - Comparació Fast vs Manual")
        plt.plot(k_list, rmse_user_k_fast, marker="o", linestyle='-', color='blue', label="User-User (Fast)")
        plt.plot(k_list, rmse_item_k_fast, marker="s", linestyle='-', color='green', label="Item-Item (Fast)")
        if has_manual_sweep:
            plt.plot(k_list, rmse_user_k_manual, marker="o", linestyle='--', color='darkblue',
                     label="User-User (Manual)")
            plt.plot(k_list, rmse_item_k_manual, marker="s", linestyle='--', color='darkgreen',
                     label="Item-Item (Manual)")
        plt.xlabel("k (nombre de veïns)")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Sweep de factors
    print("\n   -> Sweep de nombre de factors per Funk-SVD...")
    factors_list = [5, 10, 20, 40, 60, 80, 150, 300]
    rmse_funk_list = []
    for f in factors_list:
        model_f = train_funk_svd(matrix, n_factors=f, n_epochs=15, verbose=False)
        rmse_f = evaluate_funk_svd(test, matrix, user_means, model_f)
        rmse_funk_list.append(rmse_f)

    plt.figure(figsize=(8, 5))
    plt.title("RMSE vs nombre de factors (Funk-SVD)")
    plt.plot(factors_list, rmse_funk_list, marker="o", color='salmon')
    plt.xlabel("Nombre de factors latents")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n✅ Mètriques i gràfics generats correctament.")

def compare_filter_bias(ratings_raw, movies):
    print("\n📊 --- COMPARATIVA DE FILTRATGE (Bias Shift) ---")
    
    # Definim els escenaris a comparar
    scenarios = {
        "Raw (k=0)": (0, 0),
        "Moderat (k=5)": (5, 5),
        "Estricte (k=10)": (10, 10)
    }
    
    # Preparem les dades per a cada escenari
    data_store = {}
    print("   -> Aplicant filtres per generar comparatives...")
    for label, (k_u, k_i) in scenarios.items():
        if k_u == 0:
            df = ratings_raw.copy()
        else:
            df = filter_ratings_by_min_counts(ratings_raw, k_user=k_u, k_item=k_i)
        data_store[label] = df
        print(f"      - {label}: {len(df)} valoracions restants.")

    # Configuració del plot
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Impacte del Filtratge en el Biaix del Dataset', fontsize=16)

    # ---------------------------------------------------------
    # 1. DISTRIBUCIÓ DE NOTES (CORREGIT AMB BARPLOT)
    # ---------------------------------------------------------
    # Pre-calculem els percentatges manualment per assegurar que surten tots els decimals
    plot_data_list = []
    
    for label, df in data_store.items():
        # Comptem freqüència de cada nota i normalitzem a %
        counts = df['rating'].value_counts(normalize=True).sort_index() * 100
        
        # Guardem al format que necessita Seaborn
        for rating, pct in counts.items():
            plot_data_list.append({
                'Nota': rating,
                'Percentatge': pct,
                'Filtre': label
            })
            
    df_plot = pd.DataFrame(plot_data_list)
    
    # Dibuixem amb Barplot (tracta la nota com a categoria, no com a número continu)
    sns.barplot(
        data=df_plot, 
        x='Nota', 
        y='Percentatge', 
        hue='Filtre', 
        ax=axes[0, 0], 
        palette="viridis"
    )
    
    axes[0, 0].set_title("Distribució de Notes (Percentatge)")
    axes[0, 0].set_ylabel("% del total de vots")
    axes[0, 0].set_xlabel("Rating")

    # ---------------------------------------------------------
    # 2. LONG TAIL (Popularitat)
    # ---------------------------------------------------------
    ax = axes[0, 1]
    for label, df in data_store.items():
        item_counts = df.groupby('movieId').size().sort_values(ascending=False).values
        # Normalitzem eix X
        x_norm = np.linspace(0, 100, len(item_counts))
        ax.plot(x_norm, item_counts, label=label, linewidth=2)
    
    ax.set_title("Long Tail (Popularitat)")
    ax.set_xlabel("% del Catàleg de Pel·lícules (ordenat)")
    ax.set_ylabel("Vots (Log Scale)")
    ax.set_yscale('log')
    ax.legend()

    # ---------------------------------------------------------
    # 3. MITJANES D'USUARI (KDE)
    # ---------------------------------------------------------
    ax = axes[1, 0]
    for label, df in data_store.items():
        user_means = df.groupby('userId')['rating'].mean()
        sns.kdeplot(user_means, ax=ax, label=label, fill=True, alpha=0.1)
    
    ax.set_title("Canvi en la Mitjana dels Usuaris")
    ax.set_xlabel("Nota Mitjana")
    ax.legend()

    # ---------------------------------------------------------
    # 4. ACTIVITAT USUARI
    # ---------------------------------------------------------
    ax = axes[1, 1]
    for label, df in data_store.items():
        user_activity = df.groupby('userId').size().sort_values(ascending=False).values
        ax.plot(user_activity, label=label, linewidth=2)
        
    ax.set_title("Pèrdua d'Usuaris (Activitat)")
    ax.set_xlabel("Nombre d'Usuaris")
    ax.set_ylabel("Vots fets (Log Scale)")
    ax.set_yscale('log')
    ax.legend()

    plt.tight_layout()
    plt.show()
    print("✅ Gràfics comparatius generats.")

# =========================================================
# 8. MENÚ I DATA FLOW
# =========================================================

def select_algorithm(ask_for_sim_mode=False):
    while True:
        print("\n   [ Algorisme de Recomanació ]")
        print("   u. User-User (Pearson)")
        print("   i. Item-Item (Adjusted Cosine)")
        print("   c. Content-Based (Gèneres)")
        print("   f. Funk SVD (Factorització)")
        alg_choice = input("   Selecciona (u/i/c/f): ").lower().strip()

        if alg_choice in ['u', 'i', 'c', 'f']:
            alg_map = {'u': 'user', 'i': 'item', 'c': 'content', 'f': 'funk'}
            mode_alg = alg_map[alg_choice]
            mode_sim = 'library'  # Default

            if ask_for_sim_mode and alg_choice in ['u', 'i']:
                print("\n   [ Mode de Càlcul de Similitud ]")
                print("   L. Llibreries / Cache (RÀPID)")
                print("   M. Manual (RÀPID si està guardat, LENT si no)")
                sim_choice = input("   Selecciona (L/M): ").lower().strip()
                if sim_choice == 'm':
                    mode_sim = 'manual'

            return mode_alg, mode_sim
        print("⚠️ Opció no vàlida.")

def compute_all_data(base_path):
    print("⚙️  Processant dades: Lectura, filtratge i càlcul de matrius...")
    movies, ratings = load_datasets(base_path)
    # Important: Guardem el dataframe filtrat per l'anàlisi de biaix
    ratings_filtered = filter_ratings_by_min_counts(ratings, k_user=5, k_item=5)
    train, test = leave_one_out_split(ratings_filtered)

    matrix = build_matrix(train)
    matrix_nan = matrix.replace(0, np.nan)
    user_means = matrix_nan.mean(axis=1).fillna(3.0)

    # 1. Similituds Ràpides
    print("\n   [1/3] Calculant matrius RÀPIDES (Llibreria)...")
    sim_user_df_fast, sim_item_df_fast = compute_similarity_matrices(matrix)

    # 2. Entrenament Funk SVD
    print("\n   [2/3] Entrenant model Funk SVD (factors=20, epochs=20)...")
    funk_model = train_funk_svd(matrix, n_factors=20, n_epochs=20, verbose=True)

    # 3. Opcional Manual
    print("\n   [3/3] Mode de Càlcul LENT")
    print("   M. Voleu calcular i guardar la matriu MANUAL? (MOLT LENT)")
    calc_manual = input("   Calcula la matriu manual (S/N): ").lower().strip()
    sim_user_df_manual, sim_item_df_manual = (None, None)
    if calc_manual == 's':
        sim_user_df_manual, sim_item_df_manual = compute_similarity_matrices_manual(matrix, user_means)

    genre_matrix = build_genre_matrix(movies)
    user_profiles = build_user_profiles(train, genre_matrix)

    data = {
        "movies": movies, "ratings_filtered": ratings_filtered, 
        "ratings_raw":ratings, "train": train, "test": test,
        "matrix": matrix, "user_means": user_means,
        "sim_user_df_fast": sim_user_df_fast,
        "sim_item_df_fast": sim_item_df_fast,
        "sim_user_df_manual": sim_user_df_manual,
        "sim_item_df_manual": sim_item_df_manual,
        "funk_model": funk_model,
        "genre_matrix": genre_matrix, "user_profiles": user_profiles
    }
    return data

def main_menu():
    if not os.path.exists(PATH_DATA):
        print(f"❌ Error: El directori de dades '{PATH_DATA}' no existeix.")
        return

    data = None
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f: data = pickle.load(f)
            print(f"✅ Cache '{CACHE_FILE}' carregat! Sistema a punt.")
        except:
            print(f"⚠️ Error llegint cache. Recalculant.")

    if data is None:
        data = compute_all_data(PATH_DATA)
        with open(CACHE_FILE, "wb") as f: pickle.dump(data, f)
        print("✅ Dades processades i guardades a cache.")

    matrix = data["matrix"]
    movies = data["movies"]
    
    while True:
        print("\n" + "=" * 45)
        print(" SISTEMA DE RECOMANACIÓ - COMPLET (v3.1)")
        print("=" * 45)
        print("1. Predir una valoració")
        print("2. Recomanar rànquing (Top-N)")
        print("3. Generar Informe Mètriques i Gràfics")
        print("4. Recalcular i regenerar cache")
        print("5. Sortir")
        print("6. Analitzar Biaix del Dataset (NOU)")
        print("-" * 45)

        op = input("Selecciona una opció: ").strip()

        if op == "1":
            try:
                u = int(input("  User ID: "))
                m = int(input("  Movie ID: "))
                if u not in matrix.index: print("❌ User no existeix"); continue
                mode_alg, mode_sim = select_algorithm(ask_for_sim_mode=True)
                p = predict_single_rating(u, m, mode_alg, mode_sim, data)
                print(f"⭐ Predicció: {p:.4f}")
            except ValueError: print("Error d'entrada.")

        elif op == "2":
            try:
                u = int(input("  User ID: "))
                if u not in matrix.index: print("❌ User no existeix"); continue
                mode_alg, mode_sim = select_algorithm(ask_for_sim_mode=True)
                try: n = int(input("   Quantes pel·lícules? (10): ") or 10)
                except: n = 10
                recs = recommend(u, matrix, movies, mode=mode_alg, sim_mode=mode_sim, k=10, n_recs=n, data=data)
                print(f"\n🎬 TOP {n} RECOMANACIONS:")
                for i, x in enumerate(recs, 1): print(f"{i}. [{x[2]:.2f}] {x[1]}")
            except ValueError: print("Error d'entrada.")

        elif op == "3":
            generate_metrics_and_plots(
                data["movies"], data["train"], data["test"], data["matrix"],
                data["user_means"], data["sim_user_df_fast"], data["sim_item_df_fast"],
                data["sim_user_df_manual"], data["sim_item_df_manual"]
            )

        elif op == "4":
            data = compute_all_data(PATH_DATA)
            with open(CACHE_FILE, "wb") as f: pickle.dump(data, f)
            matrix = data["matrix"]
            movies = data["movies"]
            print("✅ Cache regenerada.")

        elif op == "5":
            print("Adéu! 👋")
            break

        elif op == "6":
            movies, ratings = load_datasets(PATH_DATA)
            data["ratings_raw"] = ratings
            with open(CACHE_FILE, "wb") as f: pickle.dump(data, f)
            # Ara usem 'ratings_raw' per poder filtrar des de zero
            if "ratings_raw" in data:
                compare_filter_bias(data["ratings_raw"], data["movies"])
            else:
                print("⚠️ Dades crues no trobades. Si us plau, selecciona l'opció 4 per Recalcular i regenerar cache.")

        else:
            print("⚠️ Opció no reconeguda.")

if __name__ == "__main__":
    main_menu()