"""Matrix factorisation method using Alternating Least Squares (ALS) or Gradient Descent."""

from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from tqdm import tqdm

from src.preprocessing import DataPreprocessor
from src.metrics.metrics_utils import MetricsUtils
from .abstract_method import MatrixCompletionMethod


class MatrixFactorisation(MatrixCompletionMethod):
    """
    Matrix factorisation method using either Alternating Least Squares (ALS) or Gradient Descent.
    
    This class implements collaborative filtering by decomposing the rating matrix R into
    user and item latent factor matrices U and I such that R ≈ U @ I.T.
    """

    def __init__(
        self,
        fitting_algorithm: str = "gd",
        init_method: str = "user_mean",
        k: int = 20,
        lambda_reg: float = 0.1,
        mu_reg: float = 0.1,
        n_iter: int = 20,
        learning_rate_I: float = 0.01,
        learning_rate_U: float = 0.01,
        seed: int = 42,
        **kwargs,
    ):
        """
        Initialize the Matrix Factorisation method.

        Args:
            fitting_algorithm: Method to use ('als' for Alternating Least Squares or 'gd' for Gradient Descent)
            init_method: Method for data preprocessing/initialization
            k: Latent dimension (number of factors)
            lambda_reg: Regularization parameter for item factors
            mu_reg: Regularization parameter for user factors
            n_iter: Number of iterations
            learning_rate_I: Learning rate for item factors (used only for 'gd')
            learning_rate_U: Learning rate for user factors (used only for 'gd')
            seed: Random seed for initialization
            **kwargs: Additional parameters including 'acc_tolerance'
        
        Raises:
            ValueError: If fitting_algorithm is not 'als' or 'gd'
        """
        super().__init__(**kwargs)
        if fitting_algorithm not in ["als", "gd"]:
            raise ValueError("fitting_algorithm must be either 'als' or 'gd'")
        
        self.fitting_algorithm = fitting_algorithm
        self.k = k
        self.lambda_reg = lambda_reg
        self.mu_reg = mu_reg
        self.n_iter = n_iter
        self.learning_rate_I = learning_rate_I
        self.learning_rate_U = learning_rate_U
        self.seed = seed
        self.acc_tolerance: float = kwargs.pop("acc_tolerance", 0.5)
        
        # Matrix dimensions
        self.n_items: int = 0
        self.n_users: int = 0
        
        # Latent factor matrices
        self.U: Optional[np.ndarray] = None  # User factors (n_users x k)
        self.I: Optional[np.ndarray] = None  # Item factors (n_items x k)
        
        # Data storage
        self.train_matrix: Optional[np.ndarray] = None
        self.ratings_train: Optional[np.ndarray] = None
        self.ratings_val: Optional[np.ndarray] = None
        self.train_mask: Optional[np.ndarray] = None  # Mask for observed train entries
        self.valid_mask: Optional[np.ndarray] = None  # Mask for validation entries
        self.normalize: bool = False
        
        # Preprocessing and metrics
        self.data_preprocessor = DataPreprocessor(method=init_method)
        self.historic: list[dict] = []

    def fit(
        self,
        train_matrix: np.ndarray,
        validation_matrix: np.ndarray,
        normalize: bool = False,
        init_from_pca: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        Fit the matrix factorisation method using the specified algorithm.

        Args:
            train_matrix: Input matrix with NaN values for missing entries
            validation_matrix: Validation matrix for tracking performance
            normalize: Whether to normalize the input matrix
            init_from_pca: Optional tuple of (U, s, Vt) from SVD for initialization
            verbose: If True, print metrics during training
            **kwargs: Additional parameters (unused for this method)
        """
        self.n_users = train_matrix.shape[0]
        self.n_items = train_matrix.shape[1]

        self.normalize = normalize

        self.ratings_train = train_matrix.copy()
        self.ratings_val = validation_matrix.copy()

        self.train_matrix = train_matrix.copy()
        if normalize:
            self.train_matrix = self.data_preprocessor.normalize(self.ratings_train)

        self.train_mask = ~np.isnan(train_matrix)
        self.valid_mask = (
            (~np.isnan(validation_matrix))
        )

        # Initialize factors
        self.U, self.I = self._initialize_factors(init_from_pca=init_from_pca)
        
        # Choose algorithm based on method
        if self.fitting_algorithm == "als":
            self._fit_als(train_matrix=self.train_matrix, verbose=verbose)
        elif self.fitting_algorithm == "gd":
            self._fit_gd(train_matrix=self.train_matrix, verbose=verbose)
        else:
            raise ValueError(f"Unknown fitting_algorithm: {self.fitting_algorithm}")

        self.is_fitted = True

    def _initialize_factors(
        self, init_from_pca: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialize user and item factor matrices.

        Args:
            init_from_pca: Optional tuple of (U, s, Vt) from SVD for initialization

        Returns:
            Tuple of (U, I) factor matrices where:
                - U has shape (n_users, k)
                - I has shape (n_items, k)
        """
        if init_from_pca is not None:
            U_pca, s_pca, Vt_pca = init_from_pca
            k = self.k

            S_root = np.diag(np.sqrt(s_pca[:k]))
            U = U_pca[:, :k] @ S_root
            I = Vt_pca[:k, :].T @ S_root
            print(f"Initialized from PCA: U={U.shape}, I={I.shape}")
            return U, I
        else:
            rng = np.random.default_rng(self.seed)
            U = 0.01 * rng.standard_normal((self.n_users, self.k))
            I = 0.01 * rng.standard_normal((self.n_items, self.k))
            print(f"Random initialization: U={U.shape}, I={I.shape}")
            return U, I

    def _fit_als(self, train_matrix: np.ndarray, verbose: bool = True) -> None:
        """
        Fit using Alternating Least Squares (ALS).

        Args:
            train_matrix: Input matrix with NaN values for missing entries
            verbose: If True, print metrics during training
        """
        # Ensure factors are initialized
        assert self.U is not None and self.I is not None, "Factors must be initialized before fitting"
        assert self.train_mask is not None, "Training mask must be set"
        assert self.ratings_train is not None, "Training ratings must be set"
        assert self.ratings_val is not None, "Validation ratings must be set"
        assert self.valid_mask is not None, "Validation mask must be set"
        
        # Convert NaN to 0 for sparse matrix operations
        R_filled: np.ndarray = np.nan_to_num(train_matrix, nan=0.0)
        R_csr: sp.csr_matrix = sp.csr_matrix(R_filled)
        R_csc: sp.csc_matrix = sp.csc_matrix(R_filled)

        self.historic = []
        
        # Perform ALS iterations
        for it in tqdm(range(self.n_iter), desc="ALS iterations"):
            hold_I = self.I
            self.I = self._update_item_factors(R_csc, self.U)
            self.U = self._update_user_factors(R_csr, hold_I)

            reconstruction = self.complete()

            train_rmse, train_acc, val_rmse, val_acc = MetricsUtils.print_metrics(
                prefix=f"Iter {it+1}/{self.n_iter}",
                pred_matrix=reconstruction,
                validation_matrix=self.ratings_val,
                train_matrix=self.ratings_train,
                verbose=verbose
            )
            train_loss = self._compute_loss(rating_matrix=R_filled, reconstruction=reconstruction, mask=self.train_mask)
            val_loss = self._compute_loss(rating_matrix=self.ratings_val, reconstruction=reconstruction, mask=self.valid_mask)
            

            self.historic.append(
                {
                    "epoch": it + 1,
                    "train": {"loss": train_loss, "rmse": train_rmse, "acc": train_acc},
                    "val": {"loss": val_loss, "rmse": val_rmse, "acc": val_acc},
                }
            )
            
    def _update_item_factors(self, R_csc: sp.csc_matrix, U: np.ndarray) -> np.ndarray:
        """
        Update item matrix by solving independent ridge regressions for each item.

        Args:
            R_csc: Sparse item-by-user rating matrix
            U: Current user latent factors (shape (n_users, k))

        Returns:
            Updated item latent factors (shape (n_items, k))
        """
        I = np.zeros((self.n_items, self.k))
        lambdaI = self.lambda_reg * np.eye(self.k)

        for i in range(self.n_items):
            start, end = R_csc.indptr[i], R_csc.indptr[i + 1]
            if start == end:
                continue
            user_idx = R_csc.indices[start:end]
            ratings = R_csc.data[start:end]

            U_i = U[user_idx, :]
            A = U_i.T @ U_i + lambdaI
            b = U_i.T @ ratings
            I[i, :] = self._solve_ridge(A, b)

        return I

    def _update_user_factors(self, R_csr: sp.csr_matrix, I: np.ndarray) -> np.ndarray:
        """
        Update user latent factors by solving independent ridge regressions for each user.

        Args:
            R_csr: Sparse user-by-item rating matrix in CSR format
            I: Current item latent factors (shape (n_items, k))

        Returns:
            Updated user latent factors (shape (n_users, k))
        """
        muI = self.mu_reg * np.eye(self.k)
        U = np.zeros((self.n_users, self.k))

        for u in range(self.n_users):
            start, end = R_csr.indptr[u], R_csr.indptr[u + 1]
            if start == end:
                continue
            item_idx = R_csr.indices[start:end]
            ratings = R_csr.data[start:end]

            I_u = I[item_idx, :]
            A = I_u.T @ I_u + muI
            b = I_u.T @ ratings
            U[u, :] = self._solve_ridge(A, b)

        return U

    def _solve_ridge(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve the regularized least squares system A x = b
        using Cholesky decomposition (fast and stable).

        Args:
            A: Symmetric positive-definite system matrix
            b: Right-hand side vector

        Returns:
            Solution vector
        """
        facto = cho_factor(A)
        return cho_solve(facto, b)

    def _fit_gd(self, train_matrix: np.ndarray, verbose: bool = True) -> None:
        """
        Fit using Gradient Descent.

        Args:
            train_matrix: Input matrix with NaN values for missing entries
            verbose: If True, print metrics during training
        """
        # Ensure factors are initialized
        assert self.U is not None and self.I is not None, "Factors must be initialized before fitting"
        assert self.train_mask is not None, "Mask must be set before fitting"
        assert self.valid_mask is not None, "Validation mask must be set"
        assert self.ratings_train is not None, "Training ratings must be set"
        assert self.ratings_val is not None, "Validation ratings must be set"
        
        # Fill NaN with 0 for computation
        R_filled = np.nan_to_num(train_matrix, nan=0.0)

        self.historic = []

        # Perform gradient descent iterations
        for it in tqdm(range(self.n_iter), desc="Gradient Descent iterations"):
            # Compute residual only for observed entries
            E = self.train_mask * (R_filled - self.U @ self.I.T)

            # Gradients
            grad_U = -2 * (E @ self.I) + 2 * self.mu_reg * self.U
            grad_I = -2 * (E.T @ self.U) + 2 * self.lambda_reg * self.I

            # Update
            self.U -= self.learning_rate_U * grad_U
            self.I -= self.learning_rate_I * grad_I

            reconstruction = self.complete()

            train_rmse, train_acc, val_rmse, val_acc = MetricsUtils.print_metrics(
                prefix=f"Iter {it+1}/{self.n_iter}",
                pred_matrix=reconstruction,
                validation_matrix=self.ratings_val,
                train_matrix=self.ratings_train,
                verbose=verbose
            )
            train_loss = self._compute_loss(rating_matrix=R_filled, reconstruction=reconstruction, mask=self.train_mask)
            val_loss = self._compute_loss(rating_matrix=self.ratings_val, reconstruction=reconstruction, mask=self.valid_mask)
            

            self.historic.append(
                {
                    "epoch": it + 1,
                    "train": {"loss": train_loss, "rmse": train_rmse, "acc": train_acc},
                    "val": {"loss": val_loss, "rmse": val_rmse, "acc": val_acc},
                }
            )

    def _compute_loss(self, rating_matrix: np.ndarray, reconstruction: np.ndarray, mask: np.ndarray) -> float:
        """
        Compute the cost function:
            C(U, I) = ||R - U I^T||_S^2 + mu||U||_F^2 + lambda||I||_F^2
        where ||.||_S^2 is over observed entries only.

        Args:
            rating_matrix: Rating matrix (zeros for missing entries)
            reconstruction: Reconstructed matrix from U and I
            mask: Boolean mask, True where rating is observed

        Returns:
            Total loss (reconstruction error + regularization)
        """
        # Ensure all required attributes are set
        assert self.U is not None, "User factors must be set"
        assert self.I is not None, "Item factors must be set"

        # Residual only for observed entries
        diff = mask * (rating_matrix - reconstruction)

        # Squared error on observed entries
        mse = np.sum(diff**2)

        # Regularization
        reg = self.mu_reg * np.sum(self.U**2) + self.lambda_reg * np.sum(self.I**2)

        return float(mse + reg)

    def complete(self) -> np.ndarray:
        """
        Complete the matrix using the learned factorization.

        Returns:
            Completed matrix with no NaN values
            
        Raises:
            AssertionError: If factors are not initialized
        """
        # Ensure factors are initialized
        assert self.U is not None and self.I is not None, "Factors must be initialized"

        # Reconstruct the matrix using U @ I.T
        completed_matrix = self.U @ self.I.T
        if self.normalize:
            completed_matrix = self.data_preprocessor.denormalize(
                matrix_standardized=completed_matrix
            )
        return completed_matrix
