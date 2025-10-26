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
    
    # @staticmethod
    # def compute_metrics(
    #     iteration: int,
    #     ratings_train: np.ndarray,
    #     train_mask: np.ndarray,
    #     ratings_val: np.ndarray,
    #     valid_mask: np.ndarray,
    #     reconstruction: np.ndarray,
    #     loss: float,
    #     verbose: bool = False
    # ) -> None:
    #     """
    #     Compute and log metrics for a given iteration.
        
    #     Args:
    #         iteration: Current iteration number
    #         verbose: If True, print metrics
    #     """

    #     # Train metrics on observed entries
    #     train_loss, train_rmse, train_acc = MetricsUtils.compute_split_metrics(
    #         ratings_train, train_mask, reconstruction, loss
    #     )

    #     # Validation metrics if provided
    #     val_loss = val_rmse = val_acc = None
    #     if ratings_val is not None and valid_mask is not None:
    #         val_loss, val_rmse, val_acc = MetricsUtils.compute_split_metrics(
    #             ratings_val, valid_mask, reconstruction, loss
    #         )

    #     # Log per-epoch metrics
    #     historic.append(
    #         {
    #             "epoch": iteration + 1,
    #             "train": {"loss": train_loss, "rmse": train_rmse, "acc": train_acc},
    #             "val": {"loss": val_loss, "rmse": val_rmse, "acc": val_acc},
    #         }
    #     )

    #     if verbose:
    #         tqdm.write(
    #             f"Iter {iteration + 1}/{n_iter} | "
    #             f"Train RMSE: {train_rmse:.4f}, Acc: {train_acc:.3f}, Loss: {train_loss:.2f}, Obj: {loss:.2f}"
    #             + (
    #                 f" | Val RMSE: {val_rmse:.4f}, Acc: {val_acc:.3f}, Loss: {val_loss:.2f}"
    #                 if val_rmse is not None
    #                 else ""
    #             )
    #         )

    # @staticmethod
    # def compute_split_metrics(
    #     rating_matrix: np.ndarray,
    #     mask: np.ndarray,
    #     reconstruction: np.ndarray,
    #     loss: float,
    #     verbose: bool = False,
    #     prefix: str = "",
    # ) -> tuple[float, float, float]:
    #     """
    #     Compute reconstruction loss, RMSE, and accuracy on a data split.

    #     Args:
    #         rating_matrix: Rating matrix (may contain NaN)
    #         mask: Boolean mask selecting valid observed entries
    #         reconstruction: Reconstructed rating matrix
    #         loss: Pre-computed loss value
    #         verbose: If True, print the computed metrics
    #         prefix: Prefix string for verbose output

    #     Returns:
    #         Tuple of (loss, rmse, accuracy)
    #     """
    #     if mask is None or not np.any(mask):
    #         return np.nan, np.nan, np.nan

    #     # Use existing methods
    #     rmse = MetricsUtils.rmse(reconstruction, rating_matrix)
    #     acc = MetricsUtils.accuracy(reconstruction, rating_matrix)

    #     if verbose and prefix:
    #         print(f"{prefix} | Loss: {loss:.2f} | RMSE: {rmse:.4f} | Acc: {acc:.4f}")
            
    #     return loss, rmse, acc

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
        
    # @staticmethod
    # def print_metrics_string(
    #     train_loss: float,
    #     train_rmse: float,
    #     train_acc: float,
    #     val_loss: float,
    #     val_rmse: float,
    #     val_acc: float,
    #     prefix: str = "",
    # ) -> str:
    #     """
    #     Print the provided metrics with an optional prefix.

    #     Args:
    #         loss: Loss value
    #         rmse: RMSE value
    #         acc: Accuracy value
    #         prefix: Prefix string for output
    #     """
    #     return(
    #         f"Train RMSE: {train_rmse:.4f}, Acc: {train_acc:.3f}, Loss: {train_loss:.2f}"
    #             + (
    #                 f" | Val RMSE: {val_rmse:.4f}, Acc: {val_acc:.3f}, Loss: {val_loss:.2f}"
    #                 if val_rmse is not None
    #                 else ""
    #             )
    #     )
