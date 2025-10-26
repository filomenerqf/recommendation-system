"""Main file of the project (generic version)"""

import argparse
import inspect
import time
import numpy as np
import sys
import importlib

from src.preprocessing import DataPreprocessor
from src.metrics import MetricsUtils
from src.parsing import ArgParser


if __name__ == "__main__":
    args = ArgParser.parse_args()
    all_args = vars(args)

    print("Ratings loading...")
    ratings_train = np.load("data/ratings_train.npy")
    ratings_test = np.load("data/ratings_test.npy")
    print("Ratings Loaded.")

    start_time = time.time()

    data_preprocessor = DataPreprocessor(method="user_mean")

    if args.filter_tables:
        ratings = data_preprocessor.fusion(train_matrix=ratings_train, test_matrix=ratings_test)
        ratings, _, _ = data_preprocessor.filter_by_threshold(
            matrix=ratings,
            min_ratings_user=args.min_ratings_user,
            min_ratings_movies=args.min_ratings_movies,
        )
        ratings_train, ratings_test = data_preprocessor.split(
            ratings=ratings, test_size=args.test_size
        )

    # Normalization (used by all models)
    ratings_train_normalized = data_preprocessor.normalize(matrix=ratings_train)
    ratings_test_normalized = data_preprocessor.normalize(matrix=ratings_test)

    mcm_module = importlib.import_module("src.matrix_completion_methods")
    available_methods = {
        name: cls for name, cls in vars(mcm_module).items() if inspect.isclass(cls)
    }

    if args.method not in available_methods:
        print(f"Error: Unknown method '{args.method}'. Available methods are:")
        for m in sorted(available_methods.keys()):
            print(f"  - {m}")
        sys.exit(1)

    method_class = available_methods[args.method]

    sig_init = inspect.signature(method_class.__init__)
    valid_init_keys = sig_init.parameters.keys()
    filtered_args = {k: v for k, v in all_args.items() if k in valid_init_keys}

    print(f"\nUsing method: {args.method}")
    print("With parameters:")
    for k, v in filtered_args.items():
        print(f"  {k}: {v}")

    method = method_class(**filtered_args)

    fit_sig = inspect.signature(method.fit)
    fit_kwargs = {}

    # All possible fit args coming from parser
    potential_fit_args = {
        "train_matrix": ratings_train_normalized,
        "validation_matrix": ratings_test_normalized,
        "normalize": args.normalize,
        "verbose": args.verbose,
        "n_iter": args.n_iter,
        "init_from_pca": (
            None if not args.init_from_pca else "TODO"
        ),  # can later handle actual PCA init
    }

    # Only keep what the method.fit() actually accepts
    fit_kwargs = {k: v for k, v in potential_fit_args.items() if k in fit_sig.parameters}

    print("\nTraining model...")
    method.fit(**fit_kwargs)
    print("Training complete.")

    completed_table_normalized = method.complete()

    completed_table = data_preprocessor.denormalize(matrix_standardized=completed_table_normalized)

    end_time = time.time()
    total_time = end_time - start_time

    test_rmse = MetricsUtils.rmse(pred_matrix=completed_table, true_sparse_matrix=ratings_test)
    test_acc = MetricsUtils.accuracy(pred_matrix=completed_table, true_sparse_matrix=ratings_test)

    print(f"\nAccuracy exact: {test_acc * 100:.2f}%")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Total time: {total_time:.2f} s")

    np.save("output.npy", completed_table)
    print("\nCompleted table saved to 'output.npy'.")
