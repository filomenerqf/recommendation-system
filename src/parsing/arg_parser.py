"""arg_parser.py — Argument parser for matrix completion main script"""

import argparse


class ArgParser:
    """Handles CLI argument parsing for all matrix completion methods."""

    @staticmethod
    def get_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Generate a completed ratings table using a chosen matrix completion method."
        )

        # --- Data options ---
        parser.add_argument(
            "--filter_tables", action="store_true",
            help="Filter users/movies with too few ratings before applying the method."
        )
        parser.add_argument("--min_ratings_movies", type=int, default=5,
                            help="Minimum number of ratings required per movie.")
        parser.add_argument("--min_ratings_user", type=int, default=5,
                            help="Minimum number of ratings required per user.")
        parser.add_argument("--test_size", type=float, default=0.3,
                            help="Proportion of ratings to use for testing.")

        # --- Method selection ---
        parser.add_argument("--method", type=str, default="MatrixFactorisation",
                            help="Matrix completion method to use (class name).")
        parser.add_argument("--k", type=int, default=20,
                            help="Latent dimension or embedding size (if applicable).")
        parser.add_argument("--n_iter", type=int, default=20,
                            help="Number of iterations for the completion algorithm.")

        # --- MatrixFactorisation-specific ---
        parser.add_argument("--fitting_algorithm", type=str, default="gd",
                            help="Optimization algorithm (for MatrixFactorisation).")
        parser.add_argument("--lambda_reg", type=float, default=0.1,
                            help="Regularization term λ (MatrixFactorisation).")
        parser.add_argument("--mu_reg", type=float, default=0.1,
                            help="Regularization term μ (MatrixFactorisation).")
        parser.add_argument("--learning_rate_U", type=float, default=0.001,
                            help="Learning rate for user factors (MatrixFactorisation).")
        parser.add_argument("--learning_rate_I", type=float, default=0.001,
                            help="Learning rate for item factors (MatrixFactorisation).")

        # --- Kernel / PCA methods ---
        parser.add_argument("--gamma", type=float, default=0.1,
                            help="Kernel parameter (for kernel PCA methods).")
        parser.add_argument("--alpha", type=float, default=0.1,
                            help="Regularization term α (for kernel PCA methods).")

        # --- Fit behavior ---
        parser.add_argument("--normalize", type=bool, default=True,
                            help="Whether to normalize inside fit().")
        parser.add_argument("--verbose", type=bool, default=True,
                            help="Whether to print training info.")
        parser.add_argument("--init_from_pca", type=bool, default=False,
                            help="Whether to initialize with PCA for MatrixFactorisation.")

        return parser

    @staticmethod
    def parse_args():
        """Returns argparse.Namespace"""
        return ArgParser.get_parser().parse_args()
