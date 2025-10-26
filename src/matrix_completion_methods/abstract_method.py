"""Abstract class for methods"""

from abc import ABC, abstractmethod
import numpy as np


class MatrixCompletionMethod(ABC):
    """
    Abstract base class for matrix completion methods.

    This class provides a common interface for different matrix completion algorithms
    while ensuring consistency with the project's requirements.
    """

    def __init__(self, **kwargs):
        """
        Initialize the matrix completion method.

        Args:
            **kwargs: Method-specific parameters
        """
        self.is_fitted = False
        self.original_shape = None
        self.params = kwargs

    @abstractmethod
    def fit(self, train_matrix: np.ndarray, validation_matrix: np.ndarray, **kwargs) -> None:
        """
        Fit the matrix completion method to the data.

        Args:
            train_matrix: Input matrix with NaN values for missing entries
            validation_matrix: Validation matrix for tracking performance
            **kwargs: Method-specific parameters (e.g., n_iter, verbose, etc.)

        """
        pass

    @abstractmethod
    def complete(self) -> np.ndarray:
        """
        Complete the missing values in the matrix used for fitting.

        Returns:
            np.ndarray: Completed matrix with no NaN values
        """
        pass
