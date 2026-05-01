from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
RATINGS_PATH = ROOT / "data" / "processed" / "cleaned_ratings.csv"


def load_ratings():
    df = pd.read_csv(RATINGS_PATH)
    return df[["userId", "movieId", "rating"]].drop_duplicates()


def build_item_user_matrix(train):
    user_codes, users = pd.factorize(train["userId"], sort=True)
    item_codes, items = pd.factorize(train["movieId"], sort=True)
    ratings = train["rating"].astype(float).to_numpy()

    item_user = csr_matrix(
        (ratings, (item_codes, user_codes)),
        shape=(len(items), len(users)),
    )
    normalized = normalize(item_user, norm="l2", axis=1)

    user_lookup = {user_id: idx for idx, user_id in enumerate(users)}
    item_lookup = {movie_id: idx for idx, movie_id in enumerate(items)}
    user_items = (
        pd.DataFrame(
            {
                "userId": train["userId"].to_numpy(),
                "item_idx": item_codes,
                "rating": ratings,
            }
        )
        .groupby("userId")[["item_idx", "rating"]]
        .apply(lambda x: (x["item_idx"].to_numpy(), x["rating"].to_numpy()))
        .to_dict()
    )

    return normalized, user_lookup, item_lookup, user_items


def predict_item_cf(test, item_vectors, item_lookup, user_items, global_mean):
    predictions = []

    for row in test.itertuples(index=False):
        movie_idx = item_lookup.get(row.movieId)
        history = user_items.get(row.userId)

        if movie_idx is None or history is None:
            predictions.append(global_mean)
            continue

        rated_item_indices, user_ratings = history
        sims = item_vectors[movie_idx].dot(item_vectors[rated_item_indices].T).toarray().ravel()
        denom = np.abs(sims).sum()

        if denom == 0:
            predictions.append(global_mean)
        else:
            centered = user_ratings - global_mean
            predictions.append(global_mean + float(np.dot(sims, centered) / denom))

    return np.clip(np.array(predictions), 1, 5)


def main():
    df = load_ratings()
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    global_mean = train["rating"].mean()

    item_vectors, _, item_lookup, user_items = build_item_user_matrix(train)
    predictions = predict_item_cf(test, item_vectors, item_lookup, user_items, global_mean)

    rmse = np.sqrt(mean_squared_error(test["rating"], predictions))
    mae = mean_absolute_error(test["rating"], predictions)

    print("Item-Based Collaborative Filtering Performance")
    print("RMSE:", round(rmse, 4))
    print("MAE:", round(mae, 4))


if __name__ == "__main__":
    main()
