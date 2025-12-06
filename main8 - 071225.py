import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import time

# --- Configuració ---
CACHE_FILE = "recsys_data_final.pkl"
PATH_DATA = "./ml_latest_small"

# =========================================================
# 1. FUNCIONS DE BASE I DATA WRANGLING
# =========================================================

# (Les funcions load_datasets, filter_ratings_by_min_counts,
# leave_one_out_split, i build_matrix es mantenen igual)

def load_datasets(base_path):
    movies = pd.read_csv(os.path.join(base_path, "movies.csv"))
    ratings = pd.read_csv(os.path.join(base_path, "ratings.csv"))
    ratings = ratings.dropna()
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    ratings = ratings[(ratings["rating"] >= 1.0) & (ratings["rating"] <= 5.0)]
    return movies, ratings


def filter_ratings_by_min_counts(ratings, k_user=5, k_item=5):
    filtered = ratings.copy()
    while True:
        before = len(filtered)
        valid_users = filtered.groupby("userId").size()[lambda x: x >= k_user].index
        valid_items = filtered.groupby("movieId").size()[lambda x: x >= k_item].index
        filtered = filtered[filtered["userId"].isin(valid_users) & filtered["movieId"].isin(valid_items)]
        if len(filtered) == before: break
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
    """Pearson manual, només sobre ítems comuns."""
    mask = (vec_a > 0) & (vec_b > 0)
    if mask.sum() == 0: return 0.0

    ratings_a, ratings_b = vec_a[mask], vec_b[mask]
    centered_a = ratings_a - mitjana_a
    centered_b = ratings_b - mitjana_b

    numerator = np.dot(centered_a, centered_b)
    denominator = np.linalg.norm(centered_a) * np.linalg.norm(centered_b)

    return float(numerator / denominator) if denominator != 0 else 0.0


def adjusted_cosine_manual(item_a_ratings, item_b_ratings, user_means_series):
    """Adjusted Cosine manual, sobre usuaris comuns."""
    mask = (item_a_ratings > 0) & (item_b_ratings > 0)
    if sum(mask) == 0: return 0.0

    ratings_a = item_a_ratings.loc[mask]
    ratings_b = item_b_ratings.loc[mask]
    means = user_means_series.loc[mask]

    centered_a = ratings_a - means
    centered_b = ratings_b - means

    numerator = np.dot(centered_a.values, centered_b.values)
    denominator = np.linalg.norm(centered_a.values) * np.linalg.norm(centered_b.values)

    return float(numerator / denominator) if denominator != 0 else 0.0

# =========================================================
# 3. FUNCIONS DE PRE-CÀLCUL (OPTIMITZADES PER LLIBRERIA)
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
    """
    Calcula les matrius de similitud de forma manual mitjançant bucles for (MOLT LENT).
    Utilitza les funcions pearson_similarity_manual i adjusted_cosine_manual.
    """
    print("\n   ... INICIANT CÀLCUL MANUAL (Pot trigar MINUTS)...")
    start_time = time.time()

    user_ids = matrix.index
    movie_ids = matrix.columns

    # --- 1. User-User (Pearson) Manual ---
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

            # Només calculem la meitat superior, ja que és simètrica
            similarity = pearson_similarity_manual(ratings_i, ratings_j, mean_i, mean_j)
            sim_user_matrix[i, j] = similarity
            sim_user_matrix[j, i] = similarity  # Copiem a l'altra meitat

    sim_user_df = pd.DataFrame(sim_user_matrix, index=user_ids, columns=user_ids)

    # --- 2. Item-Item (Adjusted Cosine) Manual ---
    print("   ... calculant matriu Item-Item manualment...")
    sim_item_matrix = np.zeros((len(movie_ids), len(movie_ids)))

    for i in range(len(movie_ids)):
        m_i = movie_ids[i]
        ratings_i = matrix[m_i]

        for j in range(i, len(movie_ids)):
            m_j = movie_ids[j]
            ratings_j = matrix[m_j]

            # Només calculem la meitat superior, ja que és simètrica
            similarity = adjusted_cosine_manual(ratings_i, ratings_j, user_means_series)
            sim_item_matrix[i, j] = similarity
            sim_item_matrix[j, i] = similarity  # Copiem a l'altra meitat

    sim_item_df = pd.DataFrame(sim_item_matrix, index=movie_ids, columns=movie_ids)

    print(f"   ... CÀLCUL MANUAL FINALITZAT en {time.time() - start_time:.2f} segons.")
    return sim_user_df, sim_item_df

# =========================================================
# 4. FUNCIONS DE PREDICCIÓ FAST (MODE LLIBRERIA)
# =========================================================

def predict_user_user_fast(user, item, matrix, user_means, sim_user_df, k=10):
    if item not in matrix.columns or user not in sim_user_df.index: return user_means.get(user, 3.0)

    sim_series = sim_user_df.loc[user]
    rated_users_mask = matrix[item] > 0

    valid_sims = sim_series[rated_users_mask & (sim_series.index != user)]
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
    top_k = sims.nlargest(k)

    if top_k.empty: return user_means.get(user, 3.0)

    num = 0.0
    den = 0.0

    for rated_item_id, sim_val in top_k.items():
        rating = matrix.loc[user, rated_item_id]
        num += sim_val * rating
        den += abs(sim_val)

    pred = num / den if den != 0 else user_means.get(user, 3.0)
    return max(1.0, min(5.0, pred))


# Content Based
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
# 5. AVALUACIÓ I RECOMANACIÓ GENERAL
# =========================================================

def evaluate(test, matrix, mode, user_means, sim_df):
    """Utilitza sempre la funció FAST per a l'avaluació."""
    real, pred = [], []
    for _, row in test.iterrows():
        user, item = int(row.userId), int(row.movieId)
        if user not in matrix.index or item not in matrix.columns: continue

        if mode == "user":
            p = predict_user_user_fast(user, item, matrix, user_means, sim_df)
        elif mode == "item":
            p = predict_item_item_fast(user, item, matrix, user_means, sim_df)

        real.append(float(row.rating));
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


def recommend(user, matrix, movies, mode, sim_mode, k=10, n_recs=5, data=None):
    if user not in matrix.index: return []

    user_means = data["user_means"]
    user_row = matrix.loc[user]
    seen_items = set(user_row[user_row > 0].index)
    candidates = [m for m in matrix.columns if m not in seen_items]
    id_to_title = movies.set_index("movieId")["title"].to_dict()
    recs = []

    # ------------------------------------------------------------------
    # GESTIÓ DELS MODES DE SIMILITUD COL·LABORATIUS (User/Item)
    # ------------------------------------------------------------------

    # Determinem si el mode manual ha estat triat i si la matriu lenta existeix a la cache.
    is_manual_mode = sim_mode == 'manual' and mode in ['user', 'item']
    manual_stored_exists = is_manual_mode and (
        data.get("sim_user_df_manual") is not None if mode == 'user' else data.get("sim_item_df_manual") is not None)

    # RUTA 1: CÀLCUL MANUAL 'ON-THE-FLY' (MÉS LENT: sense cache ni matriu guardada)
    # Aquesta ruta s'executa si l'usuari tria 'Manual' PERÒ la matriu lenta no s'ha precalculat i guardat.
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
                        # Ús de la funció manual per a cada veí potencial
                        sim_val = pearson_similarity_manual(target_vec, other_user_ratings, mean_user, mean_other)
                        num += sim_val * (other_user_ratings.loc[item] - mean_other)
                        den += abs(sim_val)
                p = mean_user + num / den if den != 0 else mean_user

            elif mode == 'item':
                rated_items = user_row[user_row > 0].index
                item_ratings = matrix[item]

                for rated_item in rated_items:
                    rated_item_ratings = matrix[rated_item]
                    # Ús de la funció manual per a cada ítem valorat
                    sim_val = adjusted_cosine_manual(item_ratings, rated_item_ratings, user_means)

                    num += sim_val * user_row.loc[rated_item]
                    den += abs(sim_val)
                p = num / den if den != 0 else user_means.get(user, 3.0)

            recs.append((item, id_to_title.get(item, "Títol desconegut"), max(1.0, min(5.0, float(p)))))

        print(f"⏱️ Temps de càlcul manual ON-THE-FLY: {time.time() - start_time:.2f} segons.")
        recs.sort(key=lambda x: x[2], reverse=True)
        return recs[:n_recs]

    # ------------------------------------------------------------------
    # RUTA 2 & 3: MODES RÀPIDS (Llibreria FAST o Manual STORED) / Content Based
    # ------------------------------------------------------------------
    else:

        # 2a. Mode MANUAL STORED (Matriu guardada, ús ràpid)
        if manual_stored_exists:
            sim_user_df = data["sim_user_df_manual"]
            sim_item_df = data["sim_item_df_manual"]
            print(f"\n[Mode Manual STORED] Utilitzant matriu calculada lentament. (Ús RÀPID).")

        # 2b. Mode LLIBRERIA FAST (Matriu guardada amb llibreries, ús ràpid)
        else:
            sim_user_df = data.get("sim_user_df_fast")
            sim_item_df = data.get("sim_item_df_fast")

        # El mode Content Based no utilitza cap matriu de similitud col·laborativa
        user_profiles = data.get("user_profiles")
        genre_matrix = data.get("genre_matrix")

        start_time = time.time()

        for m in candidates:
            if mode == "user":
                p = predict_user_user_fast(user, m, matrix, user_means, sim_user_df, k=k)
            elif mode == "item":
                p = predict_item_item_fast(user, m, matrix, user_means, sim_item_df, k=k)
            elif mode == "content":
                p = predict_content_based(user, m, user_profiles, genre_matrix, user_means)
            recs.append((m, id_to_title.get(m, "Títol desconegut"), p))

        # El temps es mostra només si no estem en mode Content Based (el Content Based és ràpid però no usa sim_df)
        if mode != 'content':
            print(f"⏱️ Temps de càlcul RÀPID: {time.time() - start_time:.4f} segons.")

        recs.sort(key=lambda x: x[2], reverse=True)
        return recs[:n_recs]


# =========================================================
# 6. FUNCIÓ MAIN I MENÚ (Fragment Clau Modificat)
# =========================================================

def select_algorithm(ask_for_sim_mode=False):
    """Permet seleccionar l'algorisme i, opcionalment, el mode de similitud."""
    while True:
        print("\n   [ Algorisme de Recomanació ]")
        print("   u. User-User (Pearson)")
        print("   i. Item-Item (Adjusted Cosine)")
        print("   c. Content-Based (Gèneres)")
        alg_choice = input("   Selecciona (u/i/c): ").lower().strip()

        if alg_choice in ['u', 'i', 'c']:
            alg_map = {'u': 'user', 'i': 'item', 'c': 'content'}
            mode_alg = alg_map[alg_choice]
            mode_sim = 'library'  # Default Ràpid/Library

            # Només preguntem pel mode de similitud si se'ns demana explícitament (a l'Opció 2)
            if ask_for_sim_mode and alg_choice in ['u', 'i']:
                print("\n   [ Mode de Càlcul de Similitud ]")
                print("   L. Llibreries / Cache (RÀPID, recomanat)")
                print("   M. Manual (MOLT LENT, només per demostració)")
                sim_choice = input("   Selecciona (L/M): ").lower().strip()
                if sim_choice == 'm':
                    mode_sim = 'manual'

            return mode_alg, mode_sim
        print("⚠️ Opció d'algorisme no vàlida.")


def compute_all_data(base_path):
    print("⚙️  Processant dades: Lectura, filtratge i càlcul de matrius...")
    movies, ratings = load_datasets(base_path)
    ratings_filtered = filter_ratings_by_min_counts(ratings, k_user=5, k_item=5)
    train, test = leave_one_out_split(ratings_filtered)

    matrix = build_matrix(train)
    matrix_nan = matrix.replace(0, np.nan)
    user_means = matrix_nan.mean(axis=1).fillna(3.0)

    # 1. CÀLCUL RÀPID (SEMPRE REQUERIT)
    print("\n   [Calculant matrius RÀPIDES (Llibreria)]")
    sim_user_df_fast, sim_item_df_fast = compute_similarity_matrices(matrix)

    # 2. OPCIÓ DE CÀLCUL MANUAL (PER DEMOSTRACIÓ)
    print("\n   [ Mode de Càlcul LENT ]")
    print("   M. Voleu calcular i guardar la matriu MANUAL? (MOLT LENT, NOMÉS UNA VEGADA)")
    calc_manual = input("   Calcula la matriu manual (S/N): ").lower().strip()

    if calc_manual == 's':
        sim_user_df_manual, sim_item_df_manual = compute_similarity_matrices_manual(matrix, user_means)
    else:
        # Utilitzem None si no es calcula
        sim_user_df_manual, sim_item_df_manual = None, None

    # (Content-Based es manté igual)
    genre_matrix = build_genre_matrix(movies)
    user_profiles = build_user_profiles(train, genre_matrix)

    data = {
        "movies": movies, "train": train, "test": test,
        "matrix": matrix, "user_means": user_means,

        # Guardem amb claus explícites
        "sim_user_df_fast": sim_user_df_fast,
        "sim_item_df_fast": sim_item_df_fast,
        "sim_user_df_manual": sim_user_df_manual,  # Pot ser DataFrame o None
        "sim_item_df_manual": sim_item_df_manual,  # Pot ser DataFrame o None

        "genre_matrix": genre_matrix, "user_profiles": user_profiles
    }
    return data


def main_menu():
    if not os.path.exists(PATH_DATA):
        print(f"❌ Error: El directori de dades '{PATH_DATA}' no existeix.")
        print("Assegura't que tens la carpeta 'ml_latest_small' a la mateixa ubicació.")
        return

    # --- Càrrega intel·ligent (Cache) ---
    data = None
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            print(f"✅ Cache '{CACHE_FILE}' carregat! Sistema a punt.")
        except:
            print(f"⚠️ Error llegint el cache. Recalculant dades...")

    if data is None:
        data = compute_all_data(PATH_DATA)
        try:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(data, f)
            print("✅ Dades processades i guardades a la cache.")
        except Exception as e:
            print(f"❌ No s'ha pogut guardar la cache: {e}")

    # Desempaquetem variables clau
    matrix, movies, user_means, test, sim_user_df, sim_item_df, train = \
        data["matrix"], data["movies"], data["user_means"], data["test"], \
            data["sim_user_df_fast"], data["sim_item_df_fast"], data["train"]

    while True:
        print("\n" + "=" * 45)
        print(" SISTEMA DE RECOMANACIÓ - 3 VIES (v2.0)")
        print("=" * 45)
        print("1. Predir una valoració (Ús mode ràpid)")
        print("2. Recomanar rànquing (Selecció Mode)")
        print("3. Avaluar model (RMSE)")
        print("4. Recalcular i regenerar cache")
        print("5. Sortir")
        print("-" * 45)

        op = input("Selecciona una opció: ").strip()

        if op == "1":
            try:
                u = int(input("  User ID: "))
                m = int(input("  Movie ID: "))
                if u not in matrix.index: print("❌ User no existeix"); continue

                # CRIDA AJUSTADA: No demanem el mode de similitud (ask_for_sim_mode=False)
                mode_alg, mode_sim = select_algorithm(ask_for_sim_mode=False)

                p = 0
                if mode_alg == 'user':
                    if mode_sim == 'library':
                        p = predict_user_user_fast(u, m, matrix, user_means, sim_user_df)
                    # else:
                        # p =
                elif mode_alg == 'item':
                    p = predict_item_item_fast(u, m, matrix, user_means, sim_item_df)
                elif mode_alg == 'content':
                    p = predict_content_based(u, m, data["user_profiles"], data["genre_matrix"], user_means)

                print(f"⭐ Predicció (Ràpida): {p:.4f}")
            except ValueError:
                print("Error d'entrada.")
            except IndexError:
                print("❌ Error: Dades Content Based incompletes.")

        elif op == "2":
            try:
                # 1. Demanar l'ID d'usuari
                u = int(input("  User ID: "))

                # 2. Validar l'existència de l'usuari
                if u not in matrix.index:
                    print("❌ User no existeix")
                    continue

                # 3. Seleccionar l'Algorisme i el Mode de Similitud (Ràpid/Manual)
                mode_alg, mode_sim = select_algorithm(ask_for_sim_mode=True)

                try:
                    n = int(input("   Quantes pel·lícules vols veure? (per defecte 10): ") or 10)
                except:
                    n = 10

                print(f"\n🔎 Buscant les millors {n} recomanacions...")

                # 4. Cridar la funció de recomanació
                recs = recommend(u, matrix, movies, mode=mode_alg, sim_mode=mode_sim, k=10, n_recs=10, data=data)

                # 5. Imprimir els resultats
                print(f"\n🎬 TOP {n} RECOMANACIONS:")
                for i, x in enumerate(recs, 1):
                    # x conté (movie_id, title, score)
                    print(f"{i}. [{x[2]:.2f}] {x[1]}")

            except ValueError:
                print("Error: Has d'introduir un número per a l'User ID.")
            except IndexError:
                print("❌ Error: L'algorisme Content Based no té dades (potser la pel·lícula no té gèneres).")

        elif op == "3":
            print("\n📊 AVALUANT MODELS (Utilitza sempre mode ràpid)...")
            print(f"   RMSE User-User:     {evaluate(test, matrix, 'user', user_means, sim_user_df):.4f}")
            print(f"   RMSE Item-Item:     {evaluate(test, matrix, 'item', user_means, sim_item_df):.4f}")
            print(f"   RMSE Content-Based: {evaluate_content_based(movies, train, test):.4f}")

        elif op == "4":
            data = compute_all_data(PATH_DATA)
            try:
                with open(CACHE_FILE, "wb") as f:
                    pickle.dump(data, f)
            except:
                pass
            print("✅ Dades actualitzades i cache regenerada.")

        elif op == "5":
            print("Adéu! 👋")
            break

        else:
            print("⚠️ Opció no reconeguda.")


if __name__ == "__main__":
    main_menu()
