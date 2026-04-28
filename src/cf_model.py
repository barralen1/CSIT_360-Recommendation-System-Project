import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

df = pd.read_csv("data/processed/cleaned_ratings.csv")

df = df[["userId", "movieId", "rating"]]

train, test = train_test_split(df, test_size=0.2, random_state=42)

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

global_avg = train["rating"].mean()

print("\nGlobal Average Rating:", round(global_avg, 3))

predictions = np.full(len(test), global_avg)

rmse = np.sqrt(mean_squared_error(test["rating"], predictions))
mae = mean_absolute_error(test["rating"], predictions)

print("\nBaseline Model Performance")
print("RMSE:", round(rmse, 4))
print("MAE:", round(mae, 4))