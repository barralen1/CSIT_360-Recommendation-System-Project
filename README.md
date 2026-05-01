# Recommendation System Project

## Team Members
- Nicholas Barrale
- Paramjit Kaur

---

## Dataset Description

The dataset used in this project is sourced from **CiaoDVD**, crawled from the DVD 
category of [dvd.ciao.co.uk](http://dvd.ciao.co.uk) in December 2013.

**File used:** `movie-ratings.txt`

The dataset contains the following six features:

| Feature           | Description                        |
|-------------------|------------------------------------|
| `userId`          | Unique identifier for each user    |
| `movieId`         | Unique identifier for each movie   |
| `movie-categoryId`| Category ID of the movie           |
| `reviewId`        | Unique identifier for each review  |
| `rating`          | Movie rating on a scale of 1 to 5  |
| `date`            | Date of the review                 |

### Key Dataset Statistics
- **Total Ratings:** 72,665
- **Total Users:** 17,615
- **Total Movies:** 16,121
- **Sparsity:** 99.97%
- **Missing Values:** None
- **Duplicate Rows:** None
- **Rating Distribution:** Skewed toward higher ratings (4-star and 5-star dominant)
- **Long-tail Pattern:** Most users rate only a few movies; most movies receive only a few ratings

---

## How to Install Dependencies

Ensure you have **Python 3.x** installed. Then install the required libraries using pip:
pip install -r requirements.txt

## How to Run the Code

1. **Clone the repository:**
git clone https://github.com/barralen1/CSIT_360-Recommendation-System-Project.git   

2. **Dataset files:** This repository includes `data/raw/movie-ratings.txt` for the required models and `data/raw/trusts.txt` for the optional trust-network bonus. The cleaned ratings file is saved as `data/processed/cleaned_ratings.csv`.

3. **Run the main scripts:**
   - `python src/preprocessing.py`
   - `python src/global_avg.py`
   - `python src/itemcf.py`
   - `python src/bonus_topk_trust.py`
   - `jupyter notebook notebooks/01_eda.ipynb`
   - `jupyter notebook notebooks/Model_Comparison.ipynb`
   
## Summary of Results

Two recommendation algorithms were implemented and evaluated using **RMSE** 
(Root Mean Square Error) and **MAE** (Mean Absolute Error):

| Model              | RMSE  | MAE   |
|--------------------|-------|-------|
| Global Average     | 1.0821| 0.8351|
| Item-Based CF (KNN)| 1.1004| 0.8268|

### Extra Credit Results

The project also includes both optional CSIT 360 bonus opportunities:

1. **Top-K Evaluation:** Precision@K and Recall@K were calculated using a leave-one-positive-out setup. For each evaluated user, one rating of 4 or 5 was held out as the relevant item, and 100 unrated movies were sampled as ranking candidates.

| Model | K | Precision@K | Recall@K |
|-------|---|-------------|----------|
| Global Average | 5 | 0.0100 | 0.0502 |
| Global Average | 10 | 0.0100 | 0.1001 |
| Item-Based CF | 5 | 0.0275 | 0.1375 |
| Item-Based CF | 10 | 0.0237 | 0.2367 |

2. **Trust Network Summary:** The `trusts.txt` file was analyzed as an optional trust-network component. It contains 40,133 trust links, 1,438 unique trustors, 4,299 unique trustees, and 4,658 unique users in the trust network. The average trust rating is 1.0, with no duplicate trust links and no self-trust links.

The bonus output files are saved in:
- `results/topk_bonus_metrics.csv`
- `results/trust_network_bonus_summary.csv`

### Algorithm Descriptions

- **Global Average:** A baseline predictor that calculates the mean of all ratings in 
  the training set and uses it to predict any unknown rating.

- **Item-Based Collaborative Filtering (KNN):** Identifies movies that are rated 
  similarly to one another and uses those similarities to predict unknown ratings for 
  a given user.

### Key Observations

- Both models performed similarly overall.
- The **imbalanced rating distribution** (most ratings being 4 or 5 stars) likely 
  contributed to the similar performance between models.
- The **high sparsity (99.97%)** of the dataset made it difficult for Item-Based CF 
  to find sufficient similar items, limiting its advantage over the simpler baseline.
- Despite its simplicity, the **Global Average model achieved a lower RMSE**, 
  suggesting that the dataset's sparsity reduces the effectiveness of more complex 
  collaborative filtering approaches.
