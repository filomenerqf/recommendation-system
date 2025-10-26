# Movie Recommendation System

This group project aims to predict missing entries in a movie-user ratings matrix from the [MovieLens](https://grouplens.org/datasets/movielens/) dataset. This repository implements and compares several **matrix completion algorithms** for rating prediction on the dataset. The codebase is organized into modular components for data preprocessing, matrix completion algorithms, evaluation metrics, and utilities.


<p align="center"><img src="docs/ratings_matrix.png" alt="Ratings" title="Ratings Matrix" width="500"/></p>



## Documents
For context and overview of the work achieved one can refer to these resources :
- Short report presenting our main ideas and results ([report.pdf](docs/report.pdf))
- Slides for the project presentation ([slides.pdf](docs/slides.pdf))

---

## Structure

src/  
├── matrix_completion_methods/ # PCA, Kernel PCA, Factorization, Baseline  
├── preprocessing/ # Data preprocessing utilities  
├── metrics/ # RMSE and accuracy computation  
├── parsing/ # CLI argument parsing  
└── tuning.py # Hyperparameter tuning  
sandbox/ # Experimental scripts  
docs/ # Report, slides, and figures  
notebooks/ # Exploratory analysis  
generate.py # Main experiment launcher  
requirements.txt  

---

## Preprocessing

Minimal preprocessing utilities are provided in `src/preprocessing/data_preprocessor.py`.  

The `DataPreprocessor` supports splitting/fusing rating matrices, filtering sparse users/items, and per-user (or per-item) normalization/denormalization.  
Use the `--filter_tables` and `--normalize` arguments in `generate.py` to enable these steps.

---

## Implemented Methods

**Matrix completion algorithms** implementing different matric completion approaches :

-  **AverageCompletion**: Simple baseline method using row/column averages to fill missing values.

- **MatrixFactorisation**: Advanced method using matrix factorization with two different algorithms (Alternating Least Squares and Gradient-based optimization)

- **IterativePCA**: Iterative PCA-based imputation that alternates between estimating missing entries and computing a low-rank PCA reconstruction until convergence.

- **IterativeKernelPCA**: Extension of the previous method, using kernel techniques to try to capture nonlinear behavior in the data.


All methods inherit from a shared `MatrixCompletionMethod` base class with a unified API:
```
fit(X_train, mask)
complete(X_train, mask)
```

---

## Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run experiments

#### Matrix Factorization:

```bash
python generate.py \
    --method MatrixFactorisation \
    --fitting_algorithm gd \
    --k 20 \
    --n_iter 50 \
    --lambda_reg 0.1 \
    --mu_reg 0.1 \
    --learning_rate_U 0.005 \
    --learning_rate_I 0.005 \
    --filter_tables \
    --min_ratings_user 10 \
    --min_ratings_movies 10 \
    --test_size 0.2 \
    --normalize True \
    --verbose True
```

#### Iterative PCA:

```bash
python generate.py \
    --method IterativePCA \
    --k 20 \
    --n_iter 30 \
    --filter_tables \
    --min_ratings_user 10 \
    --min_ratings_movies 10 \
    --test_size 0.2 \
    --normalize True \
    --verbose True
```

#### Iterative Kernel PCA:

```bash
python generate.py \
    --method IterativeKernelPCA \
    --k 20 \
    --gamma 0.1 \
    --alpha 0.1 \
    --n_iter 30 \
    --filter_tables \
    --min_ratings_user 10 \
    --min_ratings_movies 10 \
    --test_size 0.2 \
    --normalize True \
    --verbose True
```

The script trains on `ratings_train.npy`, saves the completed rating matrix to `output.npy` in the working directory and prints RMSE / accuracy on the provided `ratings_test.npy`.

---

### Tuning

The file `tuning.py`is a script for hyper-parameter selection and model validation. It provides basic cross validation and tuning tools such as K-folds and grid search.