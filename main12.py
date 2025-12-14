import os
import pickle
import numpy as np
import pandas as pd
import ast
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import time
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # 2. Mostrar (Important: fer-ho després del savefig)
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

    movies.dropna(subset=['movieId'], inplace=True)
    movies['movieId'] = movies['movieId'].astype(int)
    movies['title'] = movies['title'].astype(str)

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

def train_funk_svd(matrix, n_factors=20, n_epochs=20, lr=0.005, reg=0.02, verbose=True):
    R = matrix.values.astype(float)
    num_users, num_items = R.shape
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
    if item not in matrix.columns or user not in sim_user_df.index: return user_means.get(user, 3.0)
    sim_series = sim_user_df.loc[user]
    rated_users_mask = matrix[item] > 0
    valid_sims = sim_series[rated_users_mask & (sim_series.index != user)]
    valid_sims = valid_sims[valid_sims > 0]
    top_k = valid_sims.nlargest(k)
    if top_k.empty: return user_means.get(user, 3.0)
    
    num, den = 0.0, 0.0
    for other, sim in top_k.items():
        num += sim * (matrix.loc[other, item] - user_means[other])
        den += abs(sim)
    
    mean_u = user_means.get(user, 3.0)
    pred = mean_u + num / den if den != 0 else mean_u
    return max(1.0, min(5.0, pred))

def predict_item_item_fast(user, item, matrix, user_means, sim_item_df, k=10):
    if item not in sim_item_df.index or user not in matrix.index: return user_means.get(user, 3.0)
    user_row = matrix.loc[user]
    rated = user_row[user_row > 0].index
    sims = sim_item_df.loc[item, rated]
    sims = sims[sims > 0]
    top_k = sims.nlargest(k)
    if top_k.empty: return user_means.get(user, 3.0)
    
    num, den = 0.0, 0.0
    for rated_id, sim in top_k.items():
        num += sim * matrix.loc[user, rated_id]
        den += abs(sim)
    pred = num / den if den != 0 else user_means.get(user, 3.0)
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
# 7. WRAPPERS, RECOMANACIÓ I EVALUACIÓ
# =========================================================

def predict_single_rating(user, item, mode_alg, mode_sim, data, k=10):
    matrix = data["matrix"]
    user_means = data["user_means"]

    if mode_alg == 'content':
        return predict_content_based(user, item, data["user_profiles"], data["genre_matrix"], user_means)
    if mode_alg == 'funk':
        return predict_funk_svd(user, item, data.get("funk_model"), user_means)

    sim_df = None
    if mode_sim == 'library':
        sim_df = data.get("sim_user_df_fast") if mode_alg == 'user' else data.get("sim_item_df_fast")
    elif mode_sim == 'manual' and (mode_alg == 'user' and data.get("sim_user_df_manual") is not None):
        sim_df = data.get("sim_user_df_manual")
    elif mode_sim == 'manual' and (mode_alg == 'item' and data.get("sim_item_df_manual") is not None):
        sim_df = data.get("sim_item_df_manual")

    if sim_df is not None:
        if mode_alg == 'user': return predict_user_user_fast(user, item, matrix, user_means, sim_df, k)
        else: return predict_item_item_fast(user, item, matrix, user_means, sim_df, k)
    
    # Manual On-the-fly (fallback)
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
            return max(1.0, min(5.0, pred))

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
            return max(1.0, min(5.0, pred))
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
        p = predict_single_rating(user, m, mode, sim_mode, data, k)
        recs.append((m, id_to_title.get(m, "Desc"), p))
    
    recs.sort(key=lambda x: x[2], reverse=True)
    return recs[:n_recs]

def evaluate_model(data, mode, k=10, override_model=None, override_sim_df=None):
    """
    Funció unificada d'avaluació. Itera sobre el test set i calcula RMSE.
    Admet 'override_sim_df' per avaluar matrius manuals.
    """
    test = data["test"]
    matrix = data["matrix"]
    user_means = data["user_means"]
    real, pred = [], []
    
    # Selecció de la funció de predicció segons el mode
    if mode == 'user':
        # Si passem un override (manual), l'usem. Si no, agafem la fast del data.
        sim_df = override_sim_df if override_sim_df is not None else data["sim_user_df_fast"]
        predict_fn = lambda u, i: predict_user_user_fast(u, i, matrix, user_means, sim_df, k)
    elif mode == 'item':
        sim_df = override_sim_df if override_sim_df is not None else data["sim_item_df_fast"]
        predict_fn = lambda u, i: predict_item_item_fast(u, i, matrix, user_means, sim_df, k)
    elif mode == 'content':
        u_prof = data["user_profiles"]
        g_mat = data["genre_matrix"]
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
            # Assegurem clamping aquí també per si de cas
            r_pred = max(1.0, min(5.0, r_pred))
            real.append(float(row.rating))
            pred.append(r_pred)
            
    return np.sqrt(mean_squared_error(real, pred)) if real else float("nan")

# =========================================================
# 8. GRÀFICS
# =========================================================

def plot_rmse_comparison(data):
    print("\n📊 Generant mètriques i gràfics d'avaluació...")
    print("   (Això pot trigar una mica perquè s'avaluen diversos models)")

    # 1. Calcular RMSE Fast (Llibreria)
    print("   -> Calculant RMSE global (Fast/Content/Funk)...")
    rmse_user_fast = evaluate_model(data, 'user', k=20)
    rmse_item_fast = evaluate_model(data, 'item', k=20)
    rmse_content = evaluate_model(data, 'content')
    rmse_funk = evaluate_model(data, 'funk')

    # 2. Calcular RMSE Manual (si existeix)
    rmse_user_manual = float('nan')
    if data.get("sim_user_df_manual") is not None:
        print("   -> Calculant RMSE User-User Manual...")
        rmse_user_manual = evaluate_model(data, 'user', k=20, override_sim_df=data["sim_user_df_manual"])

    rmse_item_manual = float('nan')
    if data.get("sim_item_df_manual") is not None:
        print("   -> Calculant RMSE Item-Item Manual...")
        rmse_item_manual = evaluate_model(data, 'item', k=20, override_sim_df=data["sim_item_df_manual"])

    # 3. Mostrar valors per consola
    print(f"   RMSE User-User (Fast k=20):  {rmse_user_fast:.4f}")
    if not np.isnan(rmse_user_manual):
        print(f"   RMSE User-User (Manual k=20): {rmse_user_manual:.4f}")
    
    print(f"   RMSE Item-Item (Fast k=20):  {rmse_item_fast:.4f}")
    if not np.isnan(rmse_item_manual):
        print(f"   RMSE Item-Item (Manual k=20): {rmse_item_manual:.4f}")
        
    print(f"   RMSE Content-Based:          {rmse_content:.4f}")
    print(f"   RMSE Funk-SVD (f=20):        {rmse_funk:.4f}")

    # 4. Preparar Dades per al Gràfic
    print("\n   -> Generant gràfic comparatiu...")
    comp_names = []
    comp_rmse = []
    comp_colors = []

    # User
    comp_names.append("User (Fast)")
    comp_rmse.append(rmse_user_fast)
    comp_colors.append('skyblue')
    if not np.isnan(rmse_user_manual):
        comp_names.append("User (Manual)")
        comp_rmse.append(rmse_user_manual)
        comp_colors.append('darkblue')

    # Item
    comp_names.append("Item (Fast)")
    comp_rmse.append(rmse_item_fast)
    comp_colors.append('lightgreen')
    if not np.isnan(rmse_item_manual):
        comp_names.append("Item (Manual)")
        comp_rmse.append(rmse_item_manual)
        comp_colors.append('darkgreen')

    # Content & Funk
    comp_names.extend(["Content-Based", "Funk-SVD"])
    comp_rmse.extend([rmse_content, rmse_funk])
    comp_colors.extend(['orange', 'salmon'])

    # 5. Dibuixar
    plt.figure(figsize=(10, 6))
    plt.title("RMSE (k=20): Similitud Ràpida (Llibreria) vs Manual")
    bars = plt.bar(comp_names, comp_rmse, color=comp_colors)
    plt.ylabel("RMSE")
    plt.xticks(rotation=45, ha='right')
    
    # Etiquetes a sobre de les barres
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 4), ha='center', va='bottom')
    
    plt.tight_layout()
    save_plot_to_disk("comparativa_rmse")

def plot_filter_bias(data):
    ratings_raw = data["ratings_raw"]
    movies = data["movies"]
    current_dataset = data.get("dataset_name", "small")

    print("\n📊 --- COMPARATIVA DE FILTRATGE (Bias Shift) ---")
    
    # Definim els escenaris a comparar
    if current_dataset == 'small':
        scenarios = {
            "Raw (k=0)": 0,
            "Moderat (k=20)": 20,
            "Estricte (k=40)": 40
        }
    else:
        scenarios = {
            "Raw (k=0)": 0,
            "Moderat (k=300)": 300,
            "Estricte (k=600)": 600
        }

    # Preparem les dades per a cada escenari
    data_store = {}
    print("   -> Aplicant filtres per generar comparatives...")
    for label, k in scenarios.items():
        if k == 0:
            print(f"   Movies: {len(movies)} | Ratings: {len(ratings_raw)}")
            df = ratings_raw.copy()
        else:
            df = filter_ratings_by_min_counts(ratings_raw, k)
        data_store[label] = df
        print(f"      - {label}: {len(df)} valoracions restants.")

    # Configuració del plot
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Impacte del Filtratge en el Biaix del Dataset', fontsize=16)

    # 1. DISTRIBUCIÓ DE NOTES
    plot_data_list = []
    
    for label, df in data_store.items():
        counts = df['rating'].value_counts(normalize=True).sort_index() * 100
        for rating, pct in counts.items():
            plot_data_list.append({
                'Nota': rating,
                'Percentatge': pct,
                'Filtre': label
            })
            
    df_plot = pd.DataFrame(plot_data_list)
    
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

    # 2. LONG TAIL
    ax = axes[0, 1]
    for label, df in data_store.items():
        item_counts = df.groupby('movieId').size().sort_values(ascending=False).values
        x_norm = np.linspace(0, 100, len(item_counts))
        ax.plot(x_norm, item_counts, label=label, linewidth=2)
    
    ax.set_title("Long Tail (Popularitat)")
    ax.set_xlabel("% del Catàleg de Pel·lícules (ordenat)")
    ax.set_ylabel("Vots (Log Scale)")
    ax.set_yscale('log')
    ax.legend()

    # 3. MITJANES D'USUARI
    ax = axes[1, 0]
    for label, df in data_store.items():
        user_means = df.groupby('userId')['rating'].mean()
        sns.kdeplot(user_means, ax=ax, label=label, fill=True, alpha=0.1)
    
    ax.set_title("Canvi en la Mitjana dels Usuaris")
    ax.set_xlabel("Nota Mitjana")
    ax.legend()

    # 4. ACTIVITAT USUARI
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
    save_plot_to_disk("comparativa_filtre_bias")
    print("✅ Gràfics comparatius generats.")

def plot_rmse_vs_k_neighbors(data):
    # Sweep de k
    print("\n   -> Sweep de k (veïns) per User-User i Item-Item (mode Fast i Manual)...")
    k_list = [5, 10, 20, 40, 60, 80]

    rmse_user_k_fast = [evaluate_model(data, "user",k=k) for k in k_list]
    rmse_item_k_fast = [evaluate_model(data, "item",k=k) for k in k_list]

    if data["sim_user_df_manual"] is not None and data["sim_item_df_manual"] is not None:
        rmse_user_k_manual = [evaluate_model(data, "user", k=k, override_sim_df=data["sim_user_df_manual"]) for k in k_list]
        rmse_item_k_manual = [evaluate_model(data, "item", k=k, override_sim_df=data["sim_item_df_manual"]) for k in k_list]
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
    
    save_plot_to_disk("hiperparam_k_neighbors")

def plot_rmse_vs_factors_svd(data):
    print("   -> Entrenant SVD i avaluant hiperparàmetres (n_factors, lr(lambda), reg, epochs)...")
    matrix = data["matrix"]

    # =========================
    # 1) Sweep n_factors
    # =========================
    print("   -> Sweep: n_factors (factors latents)")
    factors = [10, 20, 50, 100, 200, 400]
    rmses_factors = []

    for f in factors:
        print(f"      ... n_factors={f}")
        temp_model = train_funk_svd(matrix, n_factors=f, n_epochs=20, lr=0.01, reg=0.1, verbose=False)
        err = evaluate_model(data, 'funk', override_model=temp_model)
        rmses_factors.append(err)

    plt.figure(figsize=(10, 6))
    plt.plot(factors, rmses_factors, marker='D', color='salmon', linestyle='-')
    plt.title("Influència de Factors Latents (SVD) en el RMSE")
    plt.xlabel("Nombre de Factors (n_factors)")
    plt.ylabel("RMSE")
    plt.grid(True)
    save_plot_to_disk("hiperparam_svd_factors")

    # =========================
    # 2) Sweep lr (lambda)
    # =========================
    print("   -> Sweep: lr (lambda)")
    lr_values = [0.0005, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.1]
    rmses_lr = []

    base_factors = 50
    base_reg = 0.1
    base_epochs = 20

    for lr in lr_values:
        print(f"      ... lr={lr}")
        temp_model = train_funk_svd(matrix, n_factors=base_factors, n_epochs=base_epochs, lr=lr, reg=base_reg, verbose=False)
        err = evaluate_model(data, 'funk', override_model=temp_model)
        rmses_lr.append(err)

    plt.figure(figsize=(10, 6))
    plt.plot(lr_values, rmses_lr, marker='o', color='salmon', linestyle='-')
    plt.title("Influència del Learning Rate (lr = lambda) en el RMSE (SVD)")
    plt.xlabel("lr (lambda)")
    plt.ylabel("RMSE")
    plt.grid(True)
    save_plot_to_disk("hiperparam_svd_lr_lambda")

    # =========================
    # 3) Sweep reg (regularitzador)
    # =========================
    print("   -> Sweep: reg (regularitzador)")
    reg_values = [0.0, 0.0025, 0.005, 0.02, 0.1, 0.2, 0.4]
    rmses_reg = []

    base_factors = 50
    base_lr = 0.01
    base_epochs = 20

    for reg in reg_values:
        print(f"      ... reg={reg}")
        temp_model = train_funk_svd(matrix, n_factors=base_factors, n_epochs=base_epochs, lr=base_lr, reg=reg, verbose=False)
        err = evaluate_model(data, 'funk', override_model=temp_model)
        rmses_reg.append(err)

    plt.figure(figsize=(10, 6))
    plt.plot(reg_values, rmses_reg, marker='s', color='salmon', linestyle='-')
    plt.title("Influència de la Regularització (reg) en el RMSE (SVD)")
    plt.xlabel("reg (regularitzador)")
    plt.ylabel("RMSE")
    plt.grid(True)
    save_plot_to_disk("hiperparam_svd_reg")

    # =========================
    # 4) Sweep n_epochs
    # =========================
    # Això és el que et permet veure molt clar quan comença l’overfitting (si el RMSE de test empitjora).
    print("   -> Sweep: n_epochs (epochs)")
    epochs_values = [1, 3, 10, 20, 30, 50, 70]
    rmses_epochs = []

    base_factors = 50
    base_lr = 0.01
    base_reg = 0.1

    for ep in epochs_values:
        print(f"      ... n_epochs={ep}")
        temp_model = train_funk_svd(matrix, n_factors=base_factors, n_epochs=ep, lr=base_lr, reg=base_reg, verbose=False)
        err = evaluate_model(data, 'funk', override_model=temp_model)
        rmses_epochs.append(err)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_values, rmses_epochs, marker='^', color='salmon', linestyle='-')
    plt.title("Influència del Nombre d'Epochs (SVD) en el RMSE")
    plt.xlabel("n_epochs")
    plt.ylabel("RMSE")
    plt.grid(True)
    save_plot_to_disk("hiperparam_svd_epochs")



def menu_graphics(data):
    ensure_plots_dir()
    while True:
        print("\n   [ M E N Ú   G R À F I C S ]")
        print("   1. Comparativa RMSE (User vs Item vs Content vs Funk)")
        print("   2. Anàlisi de Biaix (Dades Crues vs Filtrades)")
        print("   3. Hiperparàmetre: RMSE vs K Veïns (KNN)")
        print("   4. Hiperparàmetre: RMSE vs Factors Latents (SVD)")
        print("   5. Tornar al menú principal")
        
        op = input("   Selecciona gràfica: ")
        
        if op == '1': plot_rmse_comparison(data)
        elif op == '2': plot_filter_bias(data)
        elif op == '3': plot_rmse_vs_k_neighbors(data)
        elif op == '4': plot_rmse_vs_factors_svd(data)
        elif op == '5': break
        else: print("Opció no vàlida.")

# =========================================================
# 9. GESTIÓ DE DATASETS I MAIN
# =========================================================

def compute_all_data(dataset_key, k_filter):
    """Carrega, filtra, entrena i retorna el diccionari de dades complet."""
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
    funk_model = train_funk_svd(matrix, n_epochs=20, n_factors=50, lr=0.01, reg=0.1, verbose=True)

    genre_matrix = build_genre_matrix(movies)
    user_profiles = build_user_profiles(train, genre_matrix)
    
    return {
        "dataset_name": dataset_key, # Important per al plot de biaix
        "movies": movies, "ratings_raw": ratings_raw,
        "train": train, "test": test, "matrix": matrix, "user_means": user_means,
        "sim_user_df_fast": sim_u_fast, "sim_item_df_fast": sim_i_fast,
        "sim_user_df_manual": None, "sim_item_df_manual": None, # Manual es calcula a demanda
        "funk_model": funk_model,
        "genre_matrix": genre_matrix, "user_profiles": user_profiles
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
                print(f"⭐ Predicció: {p:.4f}")
            except: print("Error d'input.")

        elif op == "3" and data:
            try:
                u = int(input("User ID: "))
                alg = input("Algorisme (user/item/content/funk): ").lower()
                recs = recommend(u, data["matrix"], data["movies"], alg, 'library', data=data)
                for i, r in enumerate(recs, 1): print(f"{i}. {r[1]} ({r[2]:.2f})")
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
                # Actualitzem el cache
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