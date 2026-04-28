import pandas as pd

column_names = [
    "userId",
    "movieId",
    "categoryId",
    "reviewId",
    "rating",
    "reviewDate"
]

df = pd.read_csv(
    "data/raw/movie-ratings.txt",
    sep=",",
    names=column_names,
    header=None
)

print("Original Shape:", df.shape)

df = df.drop_duplicates()

print("After Removing Duplicates:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

df.to_csv("data/processed/cleaned_ratings.csv", index=False)

print("\nCleaned dataset saved successfully.")