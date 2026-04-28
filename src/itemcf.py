import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
!pip install scikit-surprise
!pip install "numpy<2.0"
import numpy as np

column_names = [
    "userId",
    "movieId",
    "categoryId",
    "reviewId",
    "rating",
    "reviewDate"
]

df = pd.read_csv(
    "movie-ratings.txt",
    sep=",",
    names=column_names,
    header=None
)

print("Original Shape:", df.shape)

df = df.drop_duplicates()

print("After Removing Duplicates:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

df.to_csv("cleaned_ratings.csv", index=False)

print("\nCleaned dataset saved successfully.")

from surprise import Dataset, Reader
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import KNNBasic, KNNWithMeans
from surprise import accuracy

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

print("Data loaded into Surprise format.")

from surprise.model_selection import train_test_split

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

print(f"Training set size: {trainset.n_ratings}")
print(f"Test set size: {len(testset)}")

param_grid_knn_basic = {
    'k': [20, 40, 60],
    'sim_options': {
        'name': ['msd', 'cosine', 'pearson'],
        'user_based': [False]  # Item-based similarity
    }
}
gs_knn_basic = GridSearchCV(KNNBasic, param_grid_knn_basic, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
gs_knn_basic.fit(data)

print("KNNBasic - Best RMSE score:", gs_knn_basic.best_score['rmse'])
print("KNNBasic - Best parameters:", gs_knn_basic.best_params['rmse'])

print("KNNBasic - Best MAE score:", gs_knn_basic.best_score['mae'])
print("KNNBasic - Best parameters:", gs_knn_basic.best_params['mae'])
