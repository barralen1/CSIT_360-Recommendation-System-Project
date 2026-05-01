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



## Summary of Results

Two recommendation algorithms were implemented and evaluated using **RMSE** 
(Root Mean Square Error) and **MAE** (Mean Absolute Error):

| Model              | RMSE  | MAE   |
|--------------------|-------|-------|
| Global Average     | Lower | Higher|
| Item-Based CF (KNN)| Higher| Lower |

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
