"""Static utility class for metrics computation."""

import numpy as np


class MetricsUtils:
    """
    Static utility class for computing and printing metrics.
    
    This class provides static methods for metric calculations and should not be instantiated.
    """

    @staticmethod
    def rmse(pred_matrix: np.ndarray, true_sparse_matrix: np.ndarray) -> float:
        """
        Compute RMSE between predicted and true rating matrices.

        Args:
            pred_matrix: Predicted rating matrix
            true_sparse_matrix: True rating matrix (may contain NaN)

        Returns:
            Root mean squared error over masked entries
        """
        mask = ~np.isnan(true_sparse_matrix)
        diff = pred_matrix[mask] - true_sparse_matrix[mask]
        return float(np.sqrt(np.mean(diff**2)))

    @staticmethod
    def accuracy(pred_matrix: np.ndarray, true_sparse_matrix: np.ndarray) -> float:
        """
        Compute accuracy between predicted and true rating matrices.

        Args:
            pred_matrix: Predicted rating matrix
            true_sparse_matrix: True rating matrix (may contain NaN)

        Returns:
            Accuracy (fraction of exact matches) over masked entries
        """
        mask = ~np.isnan(true_sparse_matrix)
        pred_values = pred_matrix[mask]
        true_values = true_sparse_matrix[mask]
        correct = np.sum(pred_values == true_values)
        return float(correct / len(true_values))
    
    @staticmethod
    def print_metrics(
    prefix: str,
    pred_matrix: np.ndarray,
    validation_matrix: np.ndarray,
    train_matrix: np.ndarray,
    verbose: bool = False,
    ) -> tuple[float, float, float, float]:
        """
        Print RMSE and Accuracy for training and validation sets.
        """
        train_rmse = MetricsUtils.rmse(pred_matrix, train_matrix)
        train_acc = MetricsUtils.accuracy(pred_matrix, train_matrix)

        valid_rmse = MetricsUtils.rmse(pred_matrix, validation_matrix)
        valid_acc = MetricsUtils.accuracy(pred_matrix, validation_matrix)

        if verbose:
            print(
                f"{prefix} | Train RMSE: {train_rmse:.4f} | Train Acc: {train_acc:.4f} | "
                f"Valid RMSE: {valid_rmse:.4f} | Valid Acc: {valid_acc:.4f}"
            )

        return train_rmse, train_acc, valid_rmse, valid_acc
        