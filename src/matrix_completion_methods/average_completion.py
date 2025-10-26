"""Average matrix completion method"""

from .abstract_method import MatrixCompletionMethod
import numpy as np


class AverageCompletion(MatrixCompletionMethod):
    """
    Average completion method - replaces NaN values with global average.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.matrix_average = None
        self.train_matrix = None
        self.is_fitted = False

    def fit(self, train_matrix: np.ndarray, validation_matrix: np.ndarray, **kwargs) -> None:
        """
        Fit the average completion method.

        Args:
            train_matrix: Input matrix with NaN values
            validation_matrix: Validation matrix for tracking performance
            **kwargs: Additional parameters (unused for this method)

        Returns:
        """
        self.train_matrix = train_matrix
        self.matrix_average = np.nanmean(self.train_matrix)
        self.is_fitted = True

    def complete(self) -> np.ndarray:
        """
        Complete the matrix using average completion.

        Returns:
            np.ndarray: Completed matrix
        """
        if not self.is_fitted:
            raise ValueError("Method must be fitted before completing matrix")

        assert self.train_matrix is not None, "train_matrix is None"
        assert self.matrix_average is not None, "matrix_average is None"

        train = np.asarray(self.train_matrix)
        avg = float(self.matrix_average)

        return np.nan_to_num(train, nan=avg)
