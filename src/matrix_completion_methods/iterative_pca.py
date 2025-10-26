import numpy as np

from .abstract_method import MatrixCompletionMethod
from src.preprocessing import DataPreprocessor
from src.metrics.metrics_utils import MetricsUtils
from typing import Optional


class IterativePCA(MatrixCompletionMethod):
    """
    Implements an iterative Principal Component Analysis (PCA) for matrix completion.

    Handles missing data by iteratively imputing values and applying SVD.
    Refactored to match the structure of IterativeKernelPCA.
    """

    def __init__(
        self,
        k: int = 20,
        **kwargs,
    ):
        """
        Args:
            k (int): The number of principal components (rank) to keep.
        """
        super().__init__(**kwargs)

        self.data_preprocessor = DataPreprocessor()

        self.k = k
        self.historic: list[dict] = []
        self.X_hat: Optional[np.ndarray] = None

    def fit(
        self,
        train_matrix: np.ndarray,
        validation_matrix: np.ndarray,
        normalize: bool = False,
        n_iter: int = 20,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        Fits the Iterative PCA model to the training data.

        Args:
            train_matrix (np.ndarray): The raw training matrix with NaN for missing values.
            validation_matrix (np.ndarray): The raw validation matrix with NaN.
            n_iter (int): Number of iterations for the imputation loop.
            verbose (bool): If True, print metrics during training.
            **kwargs: Additional parameters (unused for this method)
        """

        self.normalize = normalize

        # Store data and masks
        self.ratings_train = train_matrix
        self.ratings_val = validation_matrix
        self.train_mask = ~np.isnan(train_matrix)
        self.valid_mask = ~np.isnan(validation_matrix)
        
        # Normalize the raw training data at the beginning
        # This normalized matrix contains NaNs where original data was missing
        if self.normalize:
            self.train_matrix = self.data_preprocessor.normalize(train_matrix)

        # Create the initial dense matrix for iteration by filling NaNs with 0
        # (assuming normalization includes centering, so 0 is the mean)
        X = self._compute_iterative_pca_initial_matrix(train_matrix=self.train_matrix)

        self.historic = []

        for i in range(n_iter):
            # Step 1: Perform SVD on the current dense matrix X
            U, s, Vt = self._perform_svd(X)
            self.X_hat = self._perform_low_rank_approx(U, s, Vt)

            reconstruction = self.complete()

            # Compute and print metrics
            train_rmse, train_acc, val_rmse, val_acc = MetricsUtils.print_metrics(
                prefix=f"Iter {i+1}/{n_iter}",
                pred_matrix=reconstruction,
                validation_matrix=self.ratings_val,
                train_matrix=self.ratings_train,
                verbose=verbose
            )
            
            # Compute losses
            train_loss = self._compute_loss(
                rating_matrix=self.ratings_train, 
                reconstruction=reconstruction, 
                mask=self.train_mask
            )
            val_loss = self._compute_loss(
                rating_matrix=self.ratings_val, 
                reconstruction=reconstruction, 
                mask=self.valid_mask
            )

            # Log metrics
            self.historic.append(
                {
                    "epoch": i + 1,
                    "train": {"loss": train_loss, "rmse": train_rmse, "acc": train_acc},
                    "val": {"loss": val_loss, "rmse": val_rmse, "acc": val_acc},
                }
            )

        self.completed_normalized_matrix = X
        self.is_fitted = True

    def _compute_loss(self, rating_matrix: np.ndarray, reconstruction: np.ndarray, mask: np.ndarray) -> float:
        """
        Compute the loss (sum of squared errors) on observed entries.

        Args:
            rating_matrix: Rating matrix (may contain NaN)
            reconstruction: Reconstructed rating matrix
            mask: Boolean mask, True where rating is observed

        Returns:
            Sum of squared errors on observed entries
        """
        # Residual only for observed entries
        diff = mask * (rating_matrix - reconstruction)

        # Squared error on observed entries
        loss = float(np.sum(diff**2))

        return loss

    def complete(self) -> np.ndarray:
        """
        Returns the final completed matrix after denormalization and rounding.
        """
        assert self.X_hat is not None, "You must run fit() before calling complete()."
        assert self.train_matrix is not None, "Training matrix not found."

        reconstruction = self._update_known_values(self.X_hat, train_matrix=self.train_matrix)
        if self.normalize:
            reconstruction = self.data_preprocessor.denormalize(reconstruction)
        return np.rint(
            reconstruction
        )

    def _compute_iterative_pca_initial_matrix(
        self, train_matrix: np.ndarray
    ) -> np.ndarray:
        X_init = train_matrix.copy()
        missing_values_mask = np.isnan(train_matrix)
        X_init[missing_values_mask] = 0
        return X_init

    def _perform_svd(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        return U, s, Vt

    def _perform_low_rank_approx(
        self, U: np.ndarray, s: np.ndarray, Vt: np.ndarray
    ) -> np.ndarray:
        X_hat = U[:, : self.k] @ np.diag(s[: self.k]) @ Vt[: self.k, :]
        return X_hat

    def _update_known_values(
        self,
        X_hat: np.ndarray,
        train_matrix: np.ndarray,
    ) -> np.ndarray:

        known_values_mask = ~np.isnan(train_matrix)
        X_next = X_hat.copy()
        X_next[known_values_mask] = train_matrix[known_values_mask]

        return X_next
