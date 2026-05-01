from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
RATINGS_PATH = PROCESSED_DIR / "cleaned_ratings.csv"
TRUSTS_PATH = RAW_DIR / "trusts.txt"


def load_ratings():
    ratings = pd.read_csv(RATINGS_PATH)
    return ratings[["userId", "movieId", "rating"]].drop_duplicates()


def leave_one_positive_out(ratings, seed=42):
    positive = ratings[ratings["rating"] >= 4].copy()
    eligible_users = positive.groupby("userId").size()
    eligible_users = eligible_users[eligible_users >= 2].index
    positive = positive[positive["userId"].isin(eligible_users)]

    test = positive.groupby("userId", group_keys=False).sample(n=1, random_state=seed)
    test_keys = set(zip(test["userId"], test["movieId"], test["rating"]))

    mask = ratings.apply(
        lambda row: (row["userId"], row["movieId"], row["rating"]) not in test_keys,
        axis=1,
    )
    train = ratings[mask].copy()
    return train, test


def build_item_vectors(train):
    user_codes, users = pd.factorize(train["userId"], sort=True)
    item_codes, items = pd.factorize(train["movieId"], sort=True)
    values = train["rating"].astype(float).to_numpy()

    item_user = csr_matrix((values, (item_codes, user_codes)), shape=(len(items), len(users)))
    item_vectors = normalize(item_user, norm="l2", axis=1)

    item_lookup = {movie_id: idx for idx, movie_id in enumerate(items)}
    user_history = (
        pd.DataFrame(
            {
                "userId": train["userId"].to_numpy(),
                "movieId": train["movieId"].to_numpy(),
                "item_idx": item_codes,
                "rating": values,
            }
        )
        .groupby("userId")[["movieId", "item_idx", "rating"]]
        .apply(
            lambda x: {
                "movie_ids": set(x["movieId"]),
                "item_idx": x["item_idx"].to_numpy(),
                "ratings": x["rating"].to_numpy(),
            }
        )
        .to_dict()
    )
    return item_vectors, item_lookup, user_history


def item_cf_scores(candidates, user_id, item_vectors, item_lookup, user_history, global_mean):
    history = user_history.get(user_id)
    scores = np.full(len(candidates), global_mean, dtype=float)

    if history is None:
        return scores

    candidate_indices = [item_lookup.get(movie_id) for movie_id in candidates]
    known_positions = [i for i, idx in enumerate(candidate_indices) if idx is not None]
    if not known_positions:
        return scores

    known_candidate_indices = [candidate_indices[i] for i in known_positions]
    sims = item_vectors[known_candidate_indices].dot(item_vectors[history["item_idx"]].T).toarray()
    denom = np.abs(sims).sum(axis=1)
    centered_ratings = history["ratings"] - global_mean
    weighted_scores = global_mean + (sims @ centered_ratings) / np.where(denom == 0, 1, denom)

    for pos, score, weight in zip(known_positions, weighted_scores, denom):
        if weight != 0:
            scores[pos] = score

    return np.clip(scores, 1, 5)


def precision_recall_at_k(ranked_items, true_item, k):
    recommended = ranked_items[:k]
    hit = int(true_item in recommended)
    return hit / k, float(hit)


def evaluate_topk(train, test, negative_sample_size=100, seed=42):
    rng = np.random.default_rng(seed)
    all_movies = np.array(sorted(train["movieId"].unique()))
    global_mean = train["rating"].mean()
    item_vectors, item_lookup, user_history = build_item_vectors(train)

    metrics = {
        "Global Average": {5: [], 10: []},
        "Item-Based CF": {5: [], 10: []},
    }

    for row in test.itertuples(index=False):
        history = user_history.get(row.userId)
        seen_movies = history["movie_ids"] if history else set()
        negative_pool = np.array([movie for movie in all_movies if movie not in seen_movies and movie != row.movieId])
        if len(negative_pool) == 0:
            continue

        sample_size = min(negative_sample_size, len(negative_pool))
        negatives = rng.choice(negative_pool, size=sample_size, replace=False)
        candidates = np.concatenate(([row.movieId], negatives))
        candidates = rng.permutation(candidates)

        global_scores = np.full(len(candidates), global_mean)
        item_scores = item_cf_scores(candidates, row.userId, item_vectors, item_lookup, user_history, global_mean)

        rankings = {
            "Global Average": candidates[np.argsort(-global_scores)],
            "Item-Based CF": candidates[np.argsort(-item_scores)],
        }

        for model_name, ranked_items in rankings.items():
            for k in (5, 10):
                metrics[model_name][k].append(precision_recall_at_k(ranked_items, row.movieId, k))

    rows = []
    for model_name, k_values in metrics.items():
        for k, values in k_values.items():
            precision = np.mean([value[0] for value in values])
            recall = np.mean([value[1] for value in values])
            rows.append(
                {
                    "model": model_name,
                    "k": k,
                    "precision_at_k": round(float(precision), 4),
                    "recall_at_k": round(float(recall), 4),
                    "evaluated_users": len(values),
                    "negative_sample_size": negative_sample_size,
                }
            )
    return pd.DataFrame(rows)


def summarize_trust_network():
    trusts = pd.read_csv(TRUSTS_PATH, names=["trustorId", "trusteeId", "trustRating"])
    unique_users = pd.unique(trusts[["trustorId", "trusteeId"]].values.ravel())
    possible_links = len(unique_users) * (len(unique_users) - 1)

    summary = pd.DataFrame(
        [
            {
                "trust_links": len(trusts),
                "unique_trustors": trusts["trustorId"].nunique(),
                "unique_trustees": trusts["trusteeId"].nunique(),
                "unique_network_users": len(unique_users),
                "average_trust_rating": round(float(trusts["trustRating"].mean()), 4),
                "duplicate_links": int(trusts.duplicated().sum()),
                "self_trust_links": int((trusts["trustorId"] == trusts["trusteeId"]).sum()),
                "network_density": round(len(trusts) / possible_links, 6),
            }
        ]
    )
    return summary


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    ratings = load_ratings()
    train, test = leave_one_positive_out(ratings)

    topk_metrics = evaluate_topk(train, test)
    trust_summary = summarize_trust_network()

    topk_path = RESULTS_DIR / "topk_bonus_metrics.csv"
    trust_path = RESULTS_DIR / "trust_network_bonus_summary.csv"
    topk_metrics.to_csv(topk_path, index=False)
    trust_summary.to_csv(trust_path, index=False)

    print("Top-K Bonus Metrics")
    print(topk_metrics.to_string(index=False))
    print("\nTrust Network Bonus Summary")
    print(trust_summary.to_string(index=False))
    print(f"\nSaved: {topk_path}")
    print(f"Saved: {trust_path}")


if __name__ == "__main__":
    main()
