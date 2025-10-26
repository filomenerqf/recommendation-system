import numpy as np
from preprocessing import DataPreprocessor
from matrix_completion_methods import MatrixFactorisation
from metrics.metrics_utils import MetricsUtils
from sklearn.model_selection import KFold
from itertools import product
import os, csv

# Import all methods
from matrix_completion_methods import (
    MatrixFactorisation,
    IterativePCA,
    IterativeKernelPCA,
)

# ---------- Utilities ----------

def matrix_kfold(ratings, n_splits=5, seed=42):
    rng = np.random.default_rng(seed)
    mask = np.argwhere(~np.isnan(ratings))
    rng.shuffle(mask)

    kf = KFold(n_splits=n_splits, shuffle=False)

    for train_idx, val_idx in kf.split(mask):
        R_train = ratings.copy()
        R_val = ratings.copy()
        for i, j in mask[val_idx]:
            R_train[i, j] = np.nan
        for i, j in mask[train_idx]:
            R_val[i, j] = np.nan
        yield R_train, R_val


def _params_slug(algo: str, params: dict) -> str:
    def norm(v):
        if isinstance(v, float):
            return f"{v:.6g}".replace(".", "p")
        return str(v)

    keys = sorted(k for k in params.keys() if k not in ["method", "fitting_algorithm"])
    parts = [f"method={algo}"] + [f"{k}={norm(params[k])}" for k in keys]
    return "__".join(parts)


def _save_historic_csv(log_dir: str, algo: str, params: dict, fold_id: int, historic: list[dict]) -> None:
    os.makedirs(log_dir, exist_ok=True)
    slug = _params_slug(algo, params)
    path = os.path.join(log_dir, f"{slug}.csv")  # append all folds into one file
    write_header = not os.path.exists(path)
    hp_keys = sorted(k for k in params.keys() if k not in ["method"])

    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(
                [
                    "fold",
                    "epoch",
                    "train_loss",
                    "train_rmse",
                    "train_acc",
                    "val_loss",
                    "val_rmse",
                    "val_acc",
                ] + [f"hp_{k}" for k in hp_keys]
            )
        for e in historic:
            train = e.get("train", {})
            val = e.get("val", {})
            row = [
                fold_id,
                e.get("epoch"),
                train.get("loss"),
                train.get("rmse"),
                train.get("acc"),
                val.get("loss"),
                val.get("rmse"),
                val.get("acc"),
            ] + [params[k] for k in hp_keys]
            w.writerow(row)


# ---------- Cross-validation ----------

def cross_val_rmse(
    matrix,
    model_class,
    model_params,
    metric=MetricsUtils.rmse,
    n_splits=5,
    seed=42,
    verbose=False,
    log_dir="outputs/tuning",
):
    """Evaluate a model class via KFold cross-validation."""
    scores = []

    for fold_id, (R_train, R_val) in enumerate(matrix_kfold(matrix, n_splits=n_splits, seed=seed), start=1):
        model = model_class(**model_params)
        model.fit(train_matrix=R_train, validation_matrix=R_val, normalize=True, verbose=False)

        _save_historic_csv(
            log_dir=log_dir,
            algo=model_class.__name__,
            params=model_params,
            fold_id=fold_id,
            historic=model.historic,
        )

        R_pred = model.complete()
        score = metric(R_pred, R_val)
        scores.append(score)
        if verbose:
            print(f"  Fold {fold_id}: RMSE = {score:.4f}")

    return float(np.mean(scores)), float(np.std(scores))


# ---------- Grid search ----------

def grid_search_matrix_completion(
    matrix,
    methods_param_grid: dict,
    metric=MetricsUtils.rmse,
    n_splits=3,
    seed=42,
    verbose=True,
):
    """
    Generic grid search for multiple matrix completion methods.

    methods_param_grid = {
        "MatrixFactorisation": {
            "class": MatrixFactorisation,
            "param_grid": {
                "fitting_algorithm": ["gd"],
                "k": [10, 20],
                "lambda_reg": [0.1],
                "mu_reg": [0.1],
                "n_iter": [50],
            },
        },
        "IterativePCA": {
            "class": IterativePCA,
            "param_grid": {
                "k": [10, 20],
                "n_iter": [20],
            },
        },
        "IterativeKernelPCA": {
            "class": IterativeKernelPCA,
            "param_grid": {
                "k": [10, 20],
                "gamma": [0.1, 0.01],
                "alpha": [0.1],
                "n_iter": [20],
            },
        },
    }
    """
    results = []
    best = None

    for method_name, method_info in methods_param_grid.items():
        model_class = method_info["class"]
        grid = method_info["param_grid"]

        keys = list(grid.keys())
        for values in product(*(grid[k] for k in keys)):
            params = dict(zip(keys, values))
            params["seed"] = seed

            if verbose:
                print(f"\nTesting {method_name} with params: {params}")

            mean_rmse, std_rmse = cross_val_rmse(
                matrix=matrix,
                model_class=model_class,
                model_params=params,
                metric=metric,
                n_splits=n_splits,
                seed=seed,
                verbose=False,
            )
            results.append((method_name, params, mean_rmse, std_rmse))

            if verbose:
                print(f"→ Mean RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")

            if best is None or mean_rmse < best[2]:
                best = (method_name, params, mean_rmse)

    if best is None:
        if verbose:
            print("\nNo configuration was evaluated; returning defaults.")
        return None, {}, float("inf"), results

    best_method, best_params, best_rmse = best
    if verbose:
        print("\nBest configuration:")
        print(f"  method: {best_method}")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"  RMSE = {best_rmse:.4f}")

    return best_method, best_params, best_rmse, results


# ---------- Main script ----------

if __name__ == "__main__":
    print("Ratings loading...")
    ratings_train = np.load("data/ratings_train.npy")
    ratings_test = np.load("data/ratings_test.npy")
    print("Ratings Loaded.")

    seed = 42
    n_splits = 3

    data_preprocessor = DataPreprocessor(method="user_mean")
    ratings = data_preprocessor.fusion(train_matrix=ratings_train, test_matrix=ratings_test)

    # Define grid for all models
    methods_param_grid = {
        "MatrixFactorisation": {
            "class": MatrixFactorisation,
            "param_grid": {
                "fitting_algorithm": ["gd"],
                "k": [10, 20],
                "lambda_reg": [0.1],
                "mu_reg": [0.1],
                "n_iter": [50],
                "learning_rate_I": [0.001],
                "learning_rate_U": [0.001],
            },
        },
        "IterativePCA": {
            "class": IterativePCA,
            "param_grid": {
                "k": [10, 20],
                "n_iter": [20],
            },
        },
        "IterativeKernelPCA": {
            "class": IterativeKernelPCA,
            "param_grid": {
                "k": [10, 20],
                "gamma": [0.1, 0.01],
                "alpha": [0.1],
                "n_iter": [20],
            },
        },
    }

    best_method, best_params, best_rmse, all_results = grid_search_matrix_completion(
        matrix=ratings,
        methods_param_grid=methods_param_grid,
        metric=MetricsUtils.rmse,
        n_splits=n_splits,
        seed=seed,
        verbose=True,
    )

    print("\nBest params returned:", {"method": best_method, **best_params})
