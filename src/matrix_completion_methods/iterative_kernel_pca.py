import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import Ridge

from .abstract_method import MatrixCompletionMethod
from src.preprocessing import DataPreprocessor
from src.metrics.metrics_utils import MetricsUtils
from typing import Optional


class IterativeKernelPCA(MatrixCompletionMethod):
    """
    Implements an iterative Kernel PCA for matrix completion.

    Handles missing data by iteratively imputing values and applying
    Kernel PCA + Ridge regression for pre-image reconstruction.
    """

    def __init__(
        self,
        k: int = 20,
        gamma: float = 0.1,
        alpha: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            k (int): Number of principal components to keep.
            gamma (float): RBF kernel gamma parameter.
            alpha (float): Ridge regression regularization strength.
        """
        super().__init__(**kwargs)

        self.data_preprocessor = DataPreprocessor()
        self.k = k
        self.gamma = gamma
        self.alpha = alpha

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
        Fits the Iterative Kernel PCA model to the training data.

        Args:
            train_matrix (np.ndarray): Raw training matrix with NaN for missing values.
            validation_matrix (np.ndarray): Raw validation matrix with NaN.
            normalize (bool): Whether to normalize input data.
            n_iter (int): Number of iterations for the imputation loop.
            verbose (bool): If True, print metrics during training.
        """
        self.normalize = normalize
        self.ratings_train = train_matrix
        self.ratings_val = validation_matrix
        self.train_mask = ~np.isnan(train_matrix)
        self.valid_mask = ~np.isnan(validation_matrix)

        # Normalize the raw training data if required
        if self.normalize:
            self.train_matrix = self.data_preprocessor.normalize(train_matrix)
        else:
            self.train_matrix = train_matrix.copy()

        # Initialize matrix by replacing NaNs with 0
        X = self._compute_iterative_kpca_initial_matrix(self.train_matrix)
        self.historic = []

        for i in range(n_iter):
            # Step 1: Kernel PCA projection
            X_kpca_features = self._perform_kpca(X)

            # Step 2: Pre-image reconstruction via Ridge regression
            self.X_hat = self._perform_preimage_reconstruction(X_kpca_features, X)

            # Step 3: Update known values
            X = self._update_known_values(self.X_hat, self.train_matrix)

            # Step 4: Compute reconstructed matrix
            reconstruction = self.complete()

            # Step 5: Compute metrics
            train_rmse, train_acc, val_rmse, val_acc = MetricsUtils.print_metrics(
                prefix=f"Iter {i+1}/{n_iter}",
                pred_matrix=reconstruction,
                validation_matrix=self.ratings_val,
                train_matrix=self.ratings_train,
                verbose=verbose
            )

            # Step 6: Compute losses
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

            # Step 7: Log metrics
            self.historic.append(
                {
                    "epoch": i + 1,
                    "train": {"loss": train_loss, "rmse": train_rmse, "acc": train_acc},
                    "val": {"loss": val_loss, "rmse": val_rmse, "acc": val_acc},
                }
            )

        self.completed_normalized_matrix = X
        self.is_fitted = True

    def complete(self) -> np.ndarray:
        """
        Returns the final completed matrix after denormalization and rounding.
        """
        assert self.X_hat is not None, "You must run fit() before calling complete()."
        assert self.train_matrix is not None, "Training matrix not found."

        reconstruction = self._update_known_values(self.X_hat, self.train_matrix)
        if self.normalize:
            reconstruction = self.data_preprocessor.denormalize(reconstruction)
        return np.rint(reconstruction)

    def _compute_loss(
        self, rating_matrix: np.ndarray, reconstruction: np.ndarray, mask: np.ndarray
    ) -> float:
        """
        Compute the loss (sum of squared errors) on observed entries.
        """
        diff = mask * (rating_matrix - reconstruction)
        loss = float(np.sum(diff**2))
        return loss

    def _compute_iterative_kpca_initial_matrix(
        self, train_matrix: np.ndarray
    ) -> np.ndarray:
        X_init = train_matrix.copy()
        missing_values_mask = np.isnan(train_matrix)
        X_init[missing_values_mask] = 0
        return X_init

    def _perform_kpca(self, X: np.ndarray) -> np.ndarray:
        kpca = KernelPCA(
            n_components=self.k,
            kernel="rbf",
            gamma=self.gamma,
        )
        return kpca.fit_transform(X)

    def _perform_preimage_reconstruction(
        self, X_kpca_features: np.ndarray, X: np.ndarray
    ) -> np.ndarray:
        pre_image_model = Ridge(alpha=self.alpha)
        pre_image_model.fit(X_kpca_features, X)
        X_hat = pre_image_model.predict(X_kpca_features)
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
