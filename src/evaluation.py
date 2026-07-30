import numpy as np
import pandas as pd

class Evaluator:
    """Evaluates predicted Z_alg matrices against true Z matrices."""
    
    def __init__(self, Z_true: np.ndarray, Z_alg: np.ndarray):
        self.Z_true = Z_true
        self.Z_alg = Z_alg

    def calculate_pairwise_jaccard(self) -> float:
        """
        Calculates pairwise co-clustering Jaccard on the upper triangle.

        This compares whether each pair of statements is placed together in
        both the true and predicted partitions, excluding the diagonal.
        """
        y_true = np.argmax(self.Z_true, axis=1)
        y_pred = np.argmax(self.Z_alg, axis=1)

        true_pairs = y_true[:, None] == y_true[None, :]
        pred_pairs = y_pred[:, None] == y_pred[None, :]
        mask = np.triu(np.ones_like(true_pairs, dtype=bool), k=1)

        intersection = np.logical_and(true_pairs, pred_pairs)[mask].sum()
        union = np.logical_or(true_pairs, pred_pairs)[mask].sum()

        if union == 0:
            return 1.0
        return float(intersection / union)

    def evaluate(self) -> dict:
        """Calculates pairwise co-clustering Jaccard."""
        return {
            "jaccard_pairwise": self.calculate_pairwise_jaccard()
        }

class ResultAggregator:
    """Handles the statistical aggregation of benchmark results (calculating means)."""

    @staticmethod
    def calculate_step_means(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the mean of all metrics for each difficulty step across its iterations.
        """
        group_cols = ["step", "st", "k", "mean_cii", "std_cii", "mean_cij", "std_cij", "std_e"]
        return df.groupby(group_cols)[["jaccard_pairwise"]].mean().reset_index()

    @staticmethod
    def calculate_overall_means(df: pd.DataFrame) -> dict:
        """
        Calculates the overall mean metrics across the entire experiment (all steps and iterations).
        """
        return {
            "overall_jaccard_pairwise": df["jaccard_pairwise"].mean()
        }
