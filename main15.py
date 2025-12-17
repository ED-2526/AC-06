import os
import pickle
import numpy as np
import pandas as pd
import ast
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import time
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed  # <--- NOU: Per paral·lelitzar

# =========================================================
# CONFIGURACIÓ DE DATASETS I PLOTS
# =========================================================

PLOTS_DIR = "plots"

DATASETS = {
    "small": {
        "path": "./ml_latest_small",
        "movies_file": "movies.csv",
        "ratings_file": "ratings.csv",
        "type": "simple"
    },
    "kaggle": {
        "path": "./kaggle_dataset",
        "movies_file": "movies_metadata.csv",
        "ratings_file": "ratings.csv",
        "type": "json"
    }
}

# =========================================================
# 1. UTILITATS
# =========================================================

def ensure_plots_dir():
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

def save_plot_to_disk(filename_prefix):
    """
    Guarda el gràfic a disc i després el mostra per pantalla.
    """
    ensure_plots_dir()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.png"
    path = os.path.join(PLOTS_DIR, filename)
    
    # 1. Guardar
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"💾 Gràfica guardada a: {path}")
    
    # 2. Mostrar
    plt.show()

def get_cache_filename(dataset_name, k_filter):
    return f"cache_{dataset_name}_k{k_filter}.pkl"

def parse_genres_json(x):
    try:
        data = ast.literal_eval(x)
        if isinstance(data, list):
            names = [d['name'] for d in data if 'name' in d]
            return "|".join(names)
    except:
        pass
    return ""

def parse_keywords_json(x):
    return parse_genres_json(x)

# =========================================================
# 2. CÀRREGA DE DADES
# =========================================================

def load_datasets(dataset_key):
    conf = DATASETS.get(dataset_key)
    if not conf:
        print(f"❌ Dataset '{dataset_key}' no configurat.")
        return None, None

    base_path = conf["path"]
    movies_path = os.path.join(base_path, conf["movies_file"])
    ratings_path = os.path.join(base_path, conf["ratings_file"])

    print(f"🔄 Carregant dataset '{dataset_key}' des de: {base_path} ...")

    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        print(f"❌ Error: No es troben els fitxers a {base_path}")
        return None, None

    # 1. CARREGAR MOVIES
    try:
        movies = pd.read_csv(movies_path, low_memory=False)
    except Exception as e:
        print(f"❌ Error llegint movies: {e}")
        return None, None

    if conf["type"] == "json":
        if 'id' in movies.columns:
            movies.rename(columns={'id': 'movieId'}, inplace=True)
        movies['movieId'] = pd.to_numeric(movies['movieId'], errors='coerce')
        movies.dropna(subset=['movieId'], inplace=True)
        print("   ... processant gèneres JSON...")
        movies['genres'] = movies['genres'].apply(parse_genres_json)
        
        # (Només Kaggle) carregar keywords.csv si existeix
        keywords_path = os.path.join(base_path, "keywords.csv")
        if os.path.exists(keywords_path):
            try:
                print("   ... carregant keywords.csv...")
                kw = pd.read_csv(keywords_path, low_memory=False)
                if 'id' in kw.columns:
                    kw.rename(columns={'id': 'movieId'}, inplace=True)
                kw['movieId'] = pd.to_numeric(kw['movieId'], errors='coerce')
                kw.dropna(subset=['movieId'], inplace=True)
                kw['movieId'] = kw['movieId'].astype(int)
                if 'keywords' in kw.columns:
                    kw['keywords'] = kw['keywords'].apply(parse_keywords_json)
                else:
                    kw['keywords'] = ""
                kw = kw[['movieId', 'keywords']]
                movies = movies.merge(kw, on='movieId', how='left')
            except Exception as e:
                print(f"   ⚠️  No s'han pogut carregar keywords.csv: {e}")
                movies['keywords'] = ""
        else:
            movies['keywords'] = ""

    movies.dropna(subset=['movieId'], inplace=True)
    movies['movieId'] = movies['movieId'].astype(int)
    movies['title'] = movies['title'].astype(str)

    if 'keywords' not in movies.columns:
        movies['keywords'] = ""

    # 2. CARREGAR RATINGS
    try:
        ratings = pd.read_csv(ratings_path)
    except Exception as e:
        print(f"❌ Error llegint ratings: {e}")
        return None, None

    ratings.dropna(subset=['movieId', 'userId', 'rating'], inplace=True)
    ratings['userId'] = ratings['userId'].astype(int)
    ratings['movieId'] = ratings['movieId'].astype(int)
    ratings['rating'] = ratings['rating'].astype(float)
    
    movies.sort_values('movieId', inplace=True)
    ratings.sort_values(['userId', 'movieId'], inplace=True)

    print(f"✅ Dades '{dataset_key}' carregades.")
    print(f"   Movies: {len(movies)} | Ratings: {len(ratings)}")
    
    return movies, ratings

def filter_ratings_by_min_counts(ratings, k_val):
    k_user = k_val
    k_item = k_val
    filtered = ratings.copy()
    while True:
        before = len(filtered)
        valid_users = filtered.groupby("userId").size()[lambda x: x >= k_user].index
        valid_items = filtered.groupby("movieId").size()[lambda x: x >= k_item].index
        filtered = filtered[filtered["userId"].isin(valid_users) & filtered["movieId"].isin(valid_items)]
        if len(filtered) == before: break
    print(f"   Users: {len(valid_users)} | Movies: {len(valid_items)} | Ratings: {len(filtered)}")
    return filtered

def leave_one_out_split(ratings):
    test = ratings.groupby("userId").sample(1, random_state=42)
    train = ratings.drop(test.index)
    return train, test

def build_matrix(ratings):
    return ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)

def round_rating_to_nearest_half(x):
    try:
        return round(float(x) * 2.0) / 2.0
    except:
        return x

# =========================================================
# 3. FUNCIONS DE SIMILITUD MANUALS
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
# 4. FUNCIONS DE PRE-CÀLCUL
# =========================================================

def compute_similarity_matrices(matrix):
    print("   ... calculant matriu de similitud User-User (Pearson)...")
    matrix_nan = matrix.replace(0, np.nan)
    sim_user_df = matrix_nan.T.corr(method='pearson')

    print("   ... calculant matriu de similitud Item-Item (Adjusted Cosine)...")
    user_means = matrix_nan.mean(axis=1)
    matrix_centered = matrix_nan.sub(user_means, axis=0).fillna(0)
    sim_item_matrix = cosine_similarity(matrix_centered.T)
    sim_item_df = pd.DataFrame(sim_item_matrix, index=matrix.columns, columns=matrix.columns)

    return sim_user_df, sim_item_df

def compute_similarity_matrices_manual(matrix, user_means_series):
    print("\n   ... INICIANT CÀLCUL MANUAL (Pot trigar MOLT)...")
    start_time = time.time()
    user_ids = matrix.index
    movie_ids = matrix.columns

    # User-User
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
            sim = pearson_similarity_manual(ratings_i, ratings_j, mean_i, mean_j)
            sim_user_matrix[i, j] = sim
            sim_user_matrix[j, i] = sim
    sim_user_df = pd.DataFrame(sim_user_matrix, index=user_ids, columns=user_ids)

    # Item-Item
    print("   ... calculant matriu Item-Item manualment...")
    sim_item_matrix = np.zeros((len(movie_ids), len(movie_ids)))
    for i in range(len(movie_ids)):
        m_i = movie_ids[i]
        ratings_i = matrix[m_i]
        for j in range(i, len(movie_ids)):
            m_j = movie_ids[j]
            ratings_j = matrix[m_j]
            sim = adjusted_cosine_manual(ratings_i, ratings_j, user_means_series)
            sim_item_matrix[i, j] = sim
            sim_item_matrix[j, i] = sim
    sim_item_df = pd.DataFrame(sim_item_matrix, index=movie_ids, columns=movie_ids)

    print(f"   ... CÀLCUL MANUAL FINALITZAT en {time.time() - start_time:.2f} s.")
    return sim_user_df, sim_item_df

# =========================================================
# 5. FUNK SVD
# =========================================================

def train_funk_svd(matrix, n_factors=20, n_epochs=20, lr=0.001, reg=0.02, verbose=True):
    R = matrix.values.astype(float)
    num_users, num_items = R.shape
    np.random.seed(10)
    P = 0.1 * np.random.randn(num_users, n_factors)
    Q = 0.1 * np.random.randn(num_items, n_factors)
    
    user_index = {u: idx for idx, u in enumerate(matrix.index)}
    item_index = {i: idx for idx, i in enumerate(matrix.columns)}

    rows, cols = np.where(R > 0)
    samples = list(zip(rows, cols))

    if not samples: return {"P": P, "Q": Q, "user_index": user_index, "item_index": item_index}

    for epoch in range(n_epochs):
        np.random.shuffle(samples)
        for u_idx, i_idx in samples:
            r_ui = R[u_idx, i_idx]
            pred = np.dot(P[u_idx], Q[i_idx])
            err = r_ui - pred
            P[u_idx] += lr * (err * Q[i_idx] - reg * P[u_idx])
            Q[i_idx] += lr * (err * P[u_idx] - reg * Q[i_idx])
        print(f"Epoch {epoch}/{n_epochs} N factors {n_factors} LR {lr} Reg {reg}")
        if verbose:
            mse = 0.0
            for u_idx, i_idx in samples:
                mse += (R[u_idx, i_idx] - np.dot(P[u_idx], Q[i_idx])) ** 2
            mse /= len(samples)
            print(f"   [FunkSVD] Epoch {epoch + 1}/{n_epochs} - RMSE train: {np.sqrt(mse):.4f}")

    return {"P": P, "Q": Q, "user_index": user_index, "item_index": item_index}

def predict_funk_svd(user, item, funk_model, user_means):
    if funk_model is None: return user_means.get(user, 3.0)
    u_idx = funk_model["user_index"].get(user)
    i_idx = funk_model["item_index"].get(item)
    if u_idx is None or i_idx is None: 
        pred = user_means.get(user, 3.0)
    else:
        pred = float(np.dot(funk_model["P"][u_idx], funk_model["Q"][i_idx]))
    
    return max(1.0, min(5.0, pred))

# =========================================================
# 6. PREDICCIÓ (LLIBRERIA I CONTENT)
# =========================================================

def predict_user_user_fast(user, item, matrix, user_means, sim_user_df, k=10):
    default = user_means.get(user, 3.0)
    if item not in matrix.columns or user not in sim_user_df.index: return default
    
    sim_series = sim_user_df.loc[user]
    rated_mask = matrix[item] > 0
    valid_sims = sim_series[rated_mask & (sim_series.index != user)]
    valid_sims = valid_sims[valid_sims > 0].nlargest(k)
    
    if valid_sims.empty: return default
    
    num, den = 0.0, 0.0
    for other, sim in valid_sims.items():
        num += sim * (matrix.loc[other, item] - user_means[other])
        den += abs(sim)
    
    pred = default + (num / den if den != 0 else 0)
    return max(1.0, min(5.0, pred))

def predict_item_item_fast(user, item, matrix, user_means, sim_item_df, k=10):
    default = user_means.get(user, 3.0)
    if item not in sim_item_df.index or user not in matrix.index: return default
    
    user_row = matrix.loc[user]
    rated = user_row[user_row > 0].index
    sims = sim_item_df.loc[item, rated]
    sims = sims[sims > 0].nlargest(k)
    
    if sims.empty: return default
    
    num = sum(sim * matrix.loc[user, r_id] for r_id, sim in sims.items())
    den = sum(abs(sim) for sim in sims)
    
    pred = num / den if den != 0 else default
    return max(1.0, min(5.0, pred))

def build_genre_matrix(movies):
    all_genres = set()
    for gen_str in movies["genres"].fillna(""):
        for g in str(gen_str).split("|"):
            if g and g != "(no genres listed)": all_genres.add(g)
    
    genre_matrix = pd.DataFrame(0, index=movies["movieId"], columns=sorted(all_genres), dtype=int)
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
    user_vec = user_profiles[user]
    movie_vec = genre_matrix.loc[item].values
    num = np.dot(user_vec, movie_vec)
    den = np.linalg.norm(user_vec) * np.linalg.norm(movie_vec)
    if den == 0: return base
    pred = base + (num/den) * (5.0 - base)
    return max(1.0, min(5.0, pred))

# =========================================================
# 6B. CONTENT-BASED (TF-IDF) - (Kaggle: gèneres + keywords)
# =========================================================

# MODIFICAT: Ara accepta max_features
def build_tfidf_movie_matrix(movies, use_keywords=True, max_features=None):
    """Construeix una matriu TF-IDF per pel·lícula. Accepta max_features per limitar les paraules."""
    text_series = movies['genres'].fillna("").astype(str)
    if use_keywords:
        text_series = (text_series + " " + movies.get('keywords', "").fillna("").astype(str)).str.strip()

    text_series = text_series.str.replace("|", " ", regex=False)

    # Utilitzem 'english' stop words per eliminar soroll si utilitzem keywords
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf = vectorizer.fit_transform(text_series)

    movie_ids = movies['movieId'].astype(int).tolist()
    movieid_to_row = {mid: idx for idx, mid in enumerate(movie_ids)}
    return tfidf, movieid_to_row, vectorizer

def build_user_profiles_tfidf(train, tfidf_matrix, movieid_to_row):
    profiles = {}
    grouped = train.groupby("userId")
    for user, group in grouped:
        profile_vec = None
        for _, r in group.iterrows():
            m_id = int(r["movieId"])
            row = movieid_to_row.get(m_id)
            if row is None: 
                continue
            vec = tfidf_matrix[row]
            weighted = vec.multiply(float(r["rating"]))
            profile_vec = weighted if profile_vec is None else (profile_vec + weighted)

        if profile_vec is not None and profile_vec.nnz > 0:
            norm = sp.linalg.norm(profile_vec)
            profiles[user] = (profile_vec / norm) if norm != 0 else profile_vec
    return profiles

def predict_content_based_tfidf(user, item, user_profiles, tfidf_matrix, movieid_to_row, user_means):
    base = user_means.get(user, 3.0)
    if user not in user_profiles: 
        return base
    row = movieid_to_row.get(item)
    if row is None: 
        return base

    user_vec = user_profiles[user]
    movie_vec = tfidf_matrix[row]
    sim = float(cosine_similarity(user_vec, movie_vec)[0][0])
    pred = base + sim * (5.0 - base)
    return max(1.0, min(5.0, pred))

# =========================================================
# 7. WRAPPERS, RECOMANACIÓ I EVALUACIÓ
# =========================================================

def predict_single_rating(user, item, mode_alg, mode_sim, data, k=10):
    matrix = data["matrix"]
    user_means = data["user_means"]

    if mode_alg == 'content':
        if data.get('dataset_name') == 'kaggle' and data.get('tfidf_matrix') is not None:
            pred = predict_content_based_tfidf(user, item, data.get('user_profiles_tfidf', {}), data['tfidf_matrix'], data.get('movieid_to_row_tfidf', {}), user_means)
        else:
            pred = predict_content_based(user, item, data["user_profiles"], data["genre_matrix"], user_means)
        return round_rating_to_nearest_half(pred)
    if mode_alg == 'funk':
        pred = predict_funk_svd(user, item, data.get("funk_model"), user_means)
        return round_rating_to_nearest_half(pred)

    sim_df = None
    if mode_sim == 'library':
        sim_df = data.get("sim_user_df_fast") if mode_alg == 'user' else data.get("sim_item_df_fast")
    elif mode_sim == 'manual' and (mode_alg == 'user' and data.get("sim_user_df_manual") is not None):
        sim_df = data.get("sim_user_df_manual")
    elif mode_sim == 'manual' and (mode_alg == 'item' and data.get("sim_item_df_manual") is not None):
        sim_df = data.get("sim_item_df_manual")

    if sim_df is not None:
        if mode_alg == 'user':
            return round_rating_to_nearest_half(predict_user_user_fast(user, item, matrix, user_means, sim_df, k))
        else:
            return round_rating_to_nearest_half(predict_item_item_fast(user, item, matrix, user_means, sim_df, k))
    
    # Manual On-the-fly
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
            pred = mean_user + num / den if den != 0 else mean_user
            return round_rating_to_nearest_half(max(1.0, min(5.0, pred)))

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
            pred = num / den if den != 0 else user_means.get(user, 3.0)
            return round_rating_to_nearest_half(max(1.0, min(5.0, pred)))
    return 0.0

def recommend(user, matrix, movies, mode, sim_mode, k=10, n_recs=5, data=None):
    if user not in matrix.index: return []
    user_row = matrix.loc[user]
    seen = set(user_row[user_row > 0].index)
    candidates = [m for m in matrix.columns if m not in seen]
    id_to_title = movies.set_index("movieId")["title"].to_dict()
    
    recs = []
    print(f"\nGenerant recomanacions mode={mode}...")
    for m in candidates:
        p = predict_single_rating(user, m, mode, data, k)
        recs.append((m, id_to_title.get(m, "Desc"), p))
    
    recs.sort(key=lambda x: x[2], reverse=True)
    return recs[:n_recs]

def evaluate_model(data, mode, k=20, override_model=None, override_sim_df=None, override_content_data=None):
    """
    Funció unificada d'avaluació. Itera sobre el test set i calcula RMSE.
    """
    test = data["test"]
    matrix = data["matrix"]
    user_means = data["user_means"]
    real, pred = [], []
    
    if mode == 'user':
        sim_df = override_sim_df if override_sim_df is not None else data["sim_user_df_fast"]
        predict_fn = lambda u, i: predict_user_user_fast(u, i, matrix, user_means, sim_df, k)
    elif mode == 'item':
        sim_df = override_sim_df if override_sim_df is not None else data["sim_item_df_fast"]
        predict_fn = lambda u, i: predict_item_item_fast(u, i, matrix, user_means, sim_df, k)
    elif mode == 'content':
        if override_content_data is not None:
            tfidf_m, m_to_r, u_prof = override_content_data
            predict_fn = lambda u, i: predict_content_based_tfidf(u, i, u_prof, tfidf_m, m_to_r, user_means)
        elif data.get('dataset_name') == 'kaggle' and data.get('tfidf_matrix') is not None:
            u_prof = data['user_profiles_tfidf']
            tfidf_m = data['tfidf_matrix']
            movieid_to_row = data.get('movieid_to_row_tfidf', {})
            predict_fn = lambda u, i: predict_content_based_tfidf(u, i, u_prof, tfidf_m, movieid_to_row, user_means)
        else:
            u_prof = data['user_profiles']
            g_mat = data['genre_matrix']
            predict_fn = lambda u, i: predict_content_based(u, i, u_prof, g_mat, user_means)
    elif mode == 'funk':
        model = override_model if override_model else data["funk_model"]
        predict_fn = lambda u, i: predict_funk_svd(u, i, model, user_means)
    else:
        return float('nan')

    for _, row in test.iterrows():
        u, i = int(row.userId), int(row.movieId)
        if u in matrix.index and i in matrix.columns:
            r_pred = predict_fn(u, i)
            r_pred = max(1.0, min(5.0, r_pred))
            real.append(float(row.rating))
            pred.append(r_pred)
            
    return np.sqrt(mean_squared_error(real, pred)) if real else float("nan")

# =========================================================
# 8. GRÀFICS (ADAPTATIUS & MULTI-GRID & PARALLEL)
# =========================================================

# --- FUNCIONS AUXILIARS PARAL·LELES ---
def _evaluate_single_k(data, mode, k, override_sim_df=None):
    print(f"Mode {mode}, k {k}")
    return evaluate_model(data, mode, k=k, override_sim_df=override_sim_df)

def _evaluate_single_svd_config(matrix, data, f, lr, reg, epochs):
    print(f"Executant amb factors {f}, lr {lr}, reg {reg}, epochs {epochs}")
    temp_model = train_funk_svd(matrix, n_factors=f, n_epochs=epochs, lr=lr, reg=reg, verbose=False)
    return evaluate_model(data, 'funk', override_model=temp_model)

def _evaluate_single_content_keywords(movies, train, n, data):
    print(f"Num keywords {n}")
    tfidf, m_to_r, _ = build_tfidf_movie_matrix(movies, use_keywords=True, max_features=n)
    u_prof = build_user_profiles_tfidf(train, tfidf, m_to_r)
    return evaluate_model(data, 'content', override_content_data=(tfidf, m_to_r, u_prof))

def plot_rmse_comparison(data):
    print("\n📊 Generant mètriques i gràfics d'avaluació...")
    print("   (Això pot trigar una mica perquè s'avaluen diversos models)")

    # Utilitzem Parallel per calcular els RMSEs base
    # Tasques: [UserFast, ItemFast, Content, Funk]
    def run_eval(mode, sim_df=None):
        return evaluate_model(data, mode, k=20, override_sim_df=sim_df)

    results = Parallel(n_jobs=-1)(
        [
            delayed(run_eval)('user', data.get("sim_user_df_fast")),
            delayed(run_eval)('item', data.get("sim_item_df_fast")),
            delayed(run_eval)('content'),
            delayed(run_eval)('funk'),
        ]
    )
    rmse_user_fast, rmse_item_fast, rmse_content, rmse_funk = results

    # Manuals (opcionals)
    rmse_user_manual = float('nan')
    rmse_item_manual = float('nan')
    
    tasks_manual = []
    if data.get("sim_user_df_manual") is not None:
        tasks_manual.append(('user', data["sim_user_df_manual"]))
    else:
        tasks_manual.append(None)
        
    if data.get("sim_item_df_manual") is not None:
        tasks_manual.append(('item', data["sim_item_df_manual"]))
    else:
        tasks_manual.append(None)
    
    if any(t is not None for t in tasks_manual):
        results_manual = Parallel(n_jobs=-1)(
            delayed(run_eval)(t[0], t[1]) if t else delayed(lambda: float('nan'))() for t in tasks_manual
        )
        rmse_user_manual, rmse_item_manual = results_manual

    comp_names = ["User (Fast)"]
    comp_rmse = [rmse_user_fast]
    comp_colors = ['skyblue']
    
    if not np.isnan(rmse_user_manual):
        comp_names.append("User (Manual)")
        comp_rmse.append(rmse_user_manual)
        comp_colors.append('darkblue')

    comp_names.append("Item (Fast)")
    comp_rmse.append(rmse_item_fast)
    comp_colors.append('lightgreen')
    
    if not np.isnan(rmse_item_manual):
        comp_names.append("Item (Manual)")
        comp_rmse.append(rmse_item_manual)
        comp_colors.append('darkgreen')

    comp_names.extend(["Content-Based", "Funk-SVD"])
    comp_rmse.extend([rmse_content, rmse_funk])
    comp_colors.extend(['orange', 'salmon'])

    plt.figure(figsize=(10, 6))
    plt.title("RMSE (k=20): Similitud Ràpida (Llibreria) vs Manual")
    bars = plt.bar(comp_names, comp_rmse, color=comp_colors)
    plt.ylabel("RMSE")
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 4), ha='center', va='bottom')
    
    plt.tight_layout()
    save_plot_to_disk("comparativa_rmse")

def plot_filter_bias(data):
    print("   -> Generant anàlisi de biaix de dades...")
    ratings_raw = data["ratings_raw"]
    current_dataset = data.get("dataset_name", "small")
    
    # Aquí no paral·lelitzo perquè és ràpid i pandas gestiona bé les operacions vectoritzades
    if current_dataset == 'small':
        scenarios = {"Raw (k=0)": 0, "Moderat (k=5)": 5, "Estricte (k=40)": 40}
    else:
        scenarios = {"Raw (k=0)": 0, "Moderat (k=300)": 300, "Estricte (k=600)": 600}

    data_store = {}
    for label, k in scenarios.items():
        if k == 0: df = ratings_raw.copy()
        else: df = filter_ratings_by_min_counts(ratings_raw, k)
        data_store[label] = df

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Impacte del Filtratge en el Biaix del Dataset', fontsize=16)

    plot_data_list = []
    for label, df in data_store.items():
        counts = df['rating'].value_counts(normalize=True).sort_index() * 100
        for rating, pct in counts.items():
            plot_data_list.append({'Nota': rating, 'Percentatge': pct, 'Filtre': label})
            
    df_plot = pd.DataFrame(plot_data_list)
    sns.barplot(data=df_plot, x='Nota', y='Percentatge', hue='Filtre', ax=axes[0, 0], palette="viridis")
    axes[0, 0].set_title("Distribució de Notes (Percentatge)")

    ax = axes[0, 1]
    for label, df in data_store.items():
        item_counts = df.groupby('movieId').size().sort_values(ascending=False).values
        x_norm = np.linspace(0, 100, len(item_counts))
        ax.plot(x_norm, item_counts, label=label, linewidth=2)
    ax.set_title("Long Tail (Popularitat)")
    ax.set_yscale('log')
    ax.legend()

    ax = axes[1, 0]
    for label, df in data_store.items():
        user_means = df.groupby('userId')['rating'].mean()
        sns.kdeplot(user_means, ax=ax, label=label, fill=True, alpha=0.1)
    ax.set_title("Canvi en la Mitjana dels Usuaris")
    ax.legend()

    ax = axes[1, 1]
    for label, df in data_store.items():
        user_activity = df.groupby('userId').size().sort_values(ascending=False).values
        ax.plot(user_activity, label=label, linewidth=2)
    ax.set_title("Pèrdua d'Usuaris (Activitat)")
    ax.set_yscale('log')
    ax.legend()

    plt.tight_layout()
    save_plot_to_disk("comparativa_filtre_bias")

def plot_rmse_vs_k_neighbors(data):
    current_dataset = data.get("dataset_name", "small")
    print(f"   -> Calculant RMSE per diferents k (veïns) PARAL·LEL...")
    
    if current_dataset == 'kaggle':
        k_list = [20, 50, 100, 300, 500, 1000, 1500, 2000, 3000, 5000, 7500, 10000]
    else:
        k_list = [5, 10, 20, 40, 60, 80]

    # Paral·lelització User-Fast
    rmse_user_k_fast = Parallel(n_jobs=-1)(
        delayed(_evaluate_single_k)(data, "user", k) for k in k_list
    )
    # Paral·lelització Item-Fast
    rmse_item_k_fast = Parallel(n_jobs=-1)(
        delayed(_evaluate_single_k)(data, "item", k) for k in k_list
    )

    has_manual_sweep = False
    if data["sim_user_df_manual"] is not None and data["sim_item_df_manual"] is not None:
        rmse_user_k_manual = Parallel(n_jobs=-1)(
            delayed(_evaluate_single_k)(data, "user", k, data["sim_user_df_manual"]) for k in k_list
        )
        rmse_item_k_manual = Parallel(n_jobs=-1)(
            delayed(_evaluate_single_k)(data, "item", k, data["sim_item_df_manual"]) for k in k_list
        )
        has_manual_sweep = True

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
    save_plot_to_disk("hiperparam_k_neighbors")

def plot_rmse_vs_factors_svd(data):
    print("   -> Entrenant SVD i avaluant hiperparàmetres PARAL·LEL...")
    matrix = data["matrix"]
    current_dataset = data.get("dataset_name", "small")

    # 1) Sweep n_factors
    print("   -> Sweep: n_factors (factors latents)")
    factors = [10, 50, 100, 200, 300, 400, 600, 800, 1000, 1200, 1500, 2000] if current_dataset == 'kaggle' else [10, 20, 50, 100, 200, 400]
    
    rmses_factors = Parallel(n_jobs=-1)(
        delayed(_evaluate_single_svd_config)(matrix, data, f, 0.005, 0.02, 20) for f in factors
    )

    plt.figure(figsize=(10, 6))
    plt.plot(factors, rmses_factors, marker='D', color='salmon', linestyle='-')
    plt.title("Influència de Factors Latents (SVD) en el RMSE")
    plt.xlabel("Nombre de Factors")
    plt.ylabel("RMSE")
    plt.grid(True)
    save_plot_to_disk("hiperparam_svd_factors")

    # 2) Sweep lr
    print("   -> Sweep: lr (lambda)")
    if current_dataset == 'kaggle':
        lr_values = [0.0005, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04]
        bf, br, be = 300, 0.02, 20
    else:
        lr_values = [0.0005, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.1]
        bf, br, be = 50, 0.1, 20
    
    rmses_lr = Parallel(n_jobs=-1)(
        delayed(_evaluate_single_svd_config)(matrix, data, bf, lr, br, be) for lr in lr_values
    )

    plt.figure(figsize=(10, 6))
    plt.plot(lr_values, rmses_lr, marker='o', color='salmon', linestyle='-')
    plt.title("Influència del Learning Rate")
    plt.xlabel("lr")
    plt.ylabel("RMSE")
    plt.grid(True)
    save_plot_to_disk("hiperparam_svd_lr")

    # 3) Sweep reg
    print("   -> Sweep: reg")
    reg_values = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4] if current_dataset == 'kaggle' else [0.0, 0.0025, 0.005, 0.02, 0.1, 0.2, 0.4]
    
    rmses_reg = Parallel(n_jobs=-1)(
        delayed(_evaluate_single_svd_config)(matrix, data, bf, 0.005, reg, be) for reg in reg_values
    )

    plt.figure(figsize=(10, 6))
    plt.plot(reg_values, rmses_reg, marker='s', color='salmon', linestyle='-')
    plt.title("Influència de la Regularització")
    plt.xlabel("reg")
    plt.ylabel("RMSE")
    plt.grid(True)
    save_plot_to_disk("hiperparam_svd_reg")

    bf, br = 300, 0.02
    # 4) Sweep n_epochs
    print("   -> Sweep: n_epochs")
    epochs_values = [1, 3, 10, 30, 50, 60, 80, 100] if current_dataset == 'kaggle' else [1, 3, 10, 20, 30, 50, 70]
    
    rmses_epochs = Parallel(n_jobs=-1)(
        delayed(_evaluate_single_svd_config)(matrix, data, bf, 0.005, br, ep) for ep in epochs_values
    )

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_values, rmses_epochs, marker='^', color='salmon', linestyle='-')
    plt.title("Influència del Nombre d'Epochs")
    plt.xlabel("n_epochs")
    plt.ylabel("RMSE")
    plt.grid(True)
    save_plot_to_disk("hiperparam_svd_epochs")

# --- GRID SEARCH PARAL·LEL ---
def plot_svd_grid_search(data):
    ds_name = data.get("dataset_name", "small")
    matrix = data["matrix"]
    print(f"\n🚀 Iniciant GRID SEARCH COMPLERT PARAL·LEL (dataset: {ds_name})...")

    # 2. LR vs REG
    print("\n   [2/2] Generant Heatmap: LR vs Regularització...")
    if ds_name == 'kaggle':
        lr_grid = [0.001, 0.005, 0.01, 0.02, 0.05]
        reg_grid_2 = [0.005, 0.01, 0.02, 0.05, 0.1]
        fixed_factors = 300
    else:
        lr_grid = [0.001, 0.005, 0.01, 0.02, 0.05]
        reg_grid_2 = [0.01, 0.05, 0.1, 0.2]
        fixed_factors = 20

    tasks2 = []
    for l in lr_grid:
        for r in reg_grid_2:
            tasks2.append((l, r))
            
    results2 = Parallel(n_jobs=-1)(
        delayed(_evaluate_single_svd_config)(matrix, data, fixed_factors, t[0], t[1], 20) for t in tasks2
    )
    
    results_mat2 = np.array(results2).reshape(len(lr_grid), len(reg_grid_2))

    plt.figure(figsize=(10, 7))
    sns.heatmap(results_mat2, annot=True, fmt=".4f", cmap="coolwarm_r", 
                xticklabels=reg_grid_2, yticklabels=lr_grid)
    plt.title(f"SVD Heatmap: LR vs Reg (Factors={fixed_factors}) - {ds_name}")
    plt.xlabel("Regularització (Reg)")
    plt.ylabel("Learning Rate (LR)")
    save_plot_to_disk("svd_heatmap_lr_reg")

    # 1. FACTORS vs REG
    print("   [1/2] Generant Heatmap: Factors vs Regularització...")
    if ds_name == 'kaggle':
        factors_grid = [300, 400, 500, 600, 700]
        regs_grid = [0.005, 0.01, 0.02, 0.05, 0.1]
    else:
        factors_grid = [5, 10, 20, 50, 80]
        regs_grid = [0.01, 0.05, 0.1, 0.2]

    # Creem llista de tasques (flatten)
    tasks = []
    for f in factors_grid:
        for r in regs_grid:
            tasks.append((f, r))
    
    results = Parallel(n_jobs=-1)(
        delayed(_evaluate_single_svd_config)(matrix, data, t[0], 0.005, t[1], 20) for t in tasks
    )
    
    # Reconstruim matriu
    results_mat = np.array(results).reshape(len(factors_grid), len(regs_grid))

    plt.figure(figsize=(10, 7))
    sns.heatmap(results_mat, annot=True, fmt=".4f", cmap="coolwarm_r", 
                xticklabels=regs_grid, yticklabels=factors_grid)
    plt.title(f"SVD Heatmap: Factors vs Reg (lr=0.01) - {ds_name}")
    plt.xlabel("Regularització (Reg)")
    plt.ylabel("Nombre de Factors")
    save_plot_to_disk("svd_heatmap_factors_reg")


# --- KEYWORDS PARAL·LEL ---
def plot_rmse_vs_keywords(data):
    ds_name = data.get("dataset_name", "small")
    print(f"   -> Avaluant Content-Based segons nombre de keywords PARAL·LEL...")
    
    if ds_name != 'kaggle':
        print("   ⚠️ Aquesta gràfica només té sentit amb el dataset Kaggle (keywords). Saltant...")
        return

    n_keywords_list = [50, 100, 500, 1000, 2000, 5000, 7500, 10000, 15000]
    movies = data["movies"]
    train = data["train"]
    
    rmses = Parallel(n_jobs=-1)(
        delayed(_evaluate_single_content_keywords)(movies, train, n, data) for n in n_keywords_list
    )

    plt.figure(figsize=(10, 6))
    plt.plot(n_keywords_list, rmses, marker='o', color='purple', linestyle='-')
    plt.title(f"Influència del Nombre de Keywords (Content-Based) en el RMSE")
    plt.xlabel("Nombre de Keywords (max_features)")
    plt.ylabel("RMSE")
    plt.grid(True)
    save_plot_to_disk("hiperparam_content_keywords")

def menu_graphics(data):
    ensure_plots_dir()
    while True:
        print("\n   [ M E N Ú   G R À F I C S ]")
        print("   1. Comparativa RMSE (User vs Item vs Content vs Funk)")
        print("   2. Anàlisi de Biaix (Dades Crues vs Filtrades)")
        print("   3. Hiperparàmetre: RMSE vs K Veïns (KNN)")
        print("   4. Hiperparàmetre: RMSE vs Factors Latents (SVD)")
        print("   5. Grid Search Heatmaps (Factors/Reg i LR/Reg)")
        print("   6. Hiperparàmetre: RMSE vs Num Keywords (Content - Kaggle Only)")
        print("   7. Tornar al menú principal")
        
        op = input("   Selecciona gràfica: ")
        
        if op == '1': plot_rmse_comparison(data)
        elif op == '2': plot_filter_bias(data)
        elif op == '3': plot_rmse_vs_k_neighbors(data)
        elif op == '4': plot_rmse_vs_factors_svd(data)
        elif op == '5': plot_svd_grid_search(data)
        elif op == '6': plot_rmse_vs_keywords(data)
        elif op == '7': break
        else: print("Opció no vàlida.")

# =========================================================
# 9. GESTIÓ DE DATASETS I MAIN
# =========================================================

def compute_all_data(dataset_key, k_filter):
    movies, ratings = load_datasets(dataset_key)
    if movies is None: return None
    
    ratings_raw = ratings.copy()
    print(f"⚙️  Aplicant filtre k={k_filter}...")
    ratings_filtered = filter_ratings_by_min_counts(ratings, k_filter)
    train, test = leave_one_out_split(ratings_filtered)
    
    matrix = build_matrix(train)
    matrix_nan = matrix.replace(0, np.nan)
    user_means = matrix_nan.mean(axis=1).fillna(3.0)

    print("⚙️  Calculant matrius ràpides...")
    sim_u_fast, sim_i_fast = compute_similarity_matrices(matrix)

    print("⚙️  Entrenant Funk SVD...")
    if dataset_key == 'kaggle':
        funk_model = train_funk_svd(matrix, n_epochs=100, n_factors=500, lr=0.001, reg=0.02, verbose=True)
    else:
        funk_model = train_funk_svd(matrix, n_epochs=20, n_factors=50, lr=0.01, reg=0.1, verbose=True)

    genre_matrix = build_genre_matrix(movies)
    user_profiles = build_user_profiles(train, genre_matrix)

    # (Content-Based avançat) Kaggle: TF-IDF amb gèneres + keywords
    tfidf_matrix = None
    movieid_to_row_tfidf = None
    tfidf_vectorizer = None
    user_profiles_tfidf = None
    if dataset_key == 'kaggle':
        try:
            tfidf_matrix, movieid_to_row_tfidf, tfidf_vectorizer = build_tfidf_movie_matrix(movies, use_keywords=True)
            user_profiles_tfidf = build_user_profiles_tfidf(train, tfidf_matrix, movieid_to_row_tfidf)
        except Exception as e:
            print(f"   ⚠️  No s'ha pogut construir TF-IDF (gèneres+keywords): {e}")

    return {
        "dataset_name": dataset_key,
        "movies": movies, "ratings_raw": ratings_raw,
        "train": train, "test": test, "matrix": matrix, "user_means": user_means,
        "sim_user_df_fast": sim_u_fast, "sim_item_df_fast": sim_i_fast,
        "sim_user_df_manual": None, "sim_item_df_manual": None, # Manual es calcula a demanda
        "funk_model": funk_model,
        "genre_matrix": genre_matrix, "user_profiles": user_profiles,
        # TF-IDF (només Kaggle)
        "tfidf_matrix": tfidf_matrix,
        "movieid_to_row_tfidf": movieid_to_row_tfidf,
        "tfidf_vectorizer": tfidf_vectorizer,
        "user_profiles_tfidf": user_profiles_tfidf
    }

def main_menu():
    data = None
    current_dataset = None
    current_k = 5

    while True:
        dataset_status = f"{current_dataset} (k={current_k})" if data else "Cap"
        print(f"\n=== SISTEMA MULTI-DATASET (Actual: {dataset_status}) ===")
        print("1. Seleccionar Dataset i Carregar (Small / Kaggle)")
        print("2. Predir Valoració")
        print("3. Recomanar Top-N")
        print("4. Mètriques i Gràfics")
        print("5. Calcular Matrius Manuals (Lent)")
        print("6. Sortir")
        
        op = input("Opció: ")

        if op == "1":
            print("\nDatasets disponibles:")
            print("   s. Small (ml-latest-small)")
            print("   k. Kaggle (Full Metadata)")
            ds_choice = input("Tria (s/k): ").lower().strip()
            key = "kaggle" if ds_choice == 'k' else "small"
            
            try:
                k_val = int(input("Defineix el filtre k (min vots per user/item) : ") or 5)
            except: k_val = 10 if key == 'small' else 500

            cache_file = get_cache_filename(key, k_val)
            
            if os.path.exists(cache_file):
                print(f"✅ Trobada cache: {cache_file}. Carregant...")
                with open(cache_file, "rb") as f: data = pickle.load(f)
            else:
                print(f"⚠️ No existeix cache per {key} amb k={k_val}. Calculant des de zero...")
                data = compute_all_data(key, k_val)
                if data:
                    with open(cache_file, "wb") as f: pickle.dump(data, f)
                    print("✅ Dades guardades a cache.")
            
            if data:
                current_dataset = key
                current_k = k_val

        elif op == "2" and data:
            try:
                u = int(input("User ID: "))
                m = int(input("Movie ID: "))
                alg = input("Algorisme (user/item/content/funk): ").lower()
                p = predict_single_rating(u, m, alg, 'library', data)
                print(f"⭐ Predicció: {p:.1f}")
            except: print("Error d'input.")

        elif op == "3" and data:
            try:
                u = int(input("User ID: "))
                alg = input("Algorisme (user/item/content/funk): ").lower()
                recs = recommend(u, data["matrix"], data["movies"], alg, 'library', data=data)
                for i, r in enumerate(recs, 1): print(f"{i}. {r[1]} ({r[2]:.1f})")
            except: print("Error.")

        elif op == "4" and data:
            menu_graphics(data)

        elif op == "5" and data:
            print("⚠️ Això pot trigar molt amb datasets grans!")
            sure = input("Segur? (s/n): ")
            if sure.lower() == 's':
                u_man, i_man = compute_similarity_matrices_manual(data["matrix"], data["user_means"])
                data["sim_user_df_manual"] = u_man
                data["sim_item_df_manual"] = i_man
                cache_file = get_cache_filename(current_dataset, current_k)
                with open(cache_file, "wb") as f: pickle.dump(data, f)
                print("✅ Matrius manuals guardades.")

        elif op == "6":
            print("Adéu! 👋")
            break

        else:
            print("Opció no vàlida o dades no carregades.")

if __name__ == "__main__":
    main_menu()