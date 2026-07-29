import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, jaccard_score

class Evaluator:
    """Evaluates predicted Z_alg matrices against true Z matrices."""
    
    def __init__(self, Z_true: np.ndarray, Z_alg: np.ndarray):
        self.Z_true = Z_true
        self.Z_alg = Z_alg
        self.Z_alg_aligned = self._align_labels()

    def _align_labels(self) -> np.ndarray:
        """
        Solves the label permutation problem by maximizing the intersection 
        between true clusters and predicted clusters.
        """
        cost_matrix = - (self.Z_true.T @ self.Z_alg)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        Z_alg_aligned = np.zeros_like(self.Z_alg)
        for true_idx, alg_idx in zip(row_ind, col_ind):
            Z_alg_aligned[:, true_idx] = self.Z_alg[:, alg_idx]
            
        return Z_alg_aligned

    def get_labels(self, Z_matrix: np.ndarray) -> np.ndarray:
        """Converts a one-hot Z matrix into a 1D array of cluster labels."""
        return np.argmax(Z_matrix, axis=1)

    def evaluate(self) -> dict:
        """Calculates aligned accuracy and Jaccard metrics."""
        y_true = self.get_labels(self.Z_true)
        y_pred = self.get_labels(self.Z_alg_aligned)
        
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "jaccard_eqcluster": jaccard_score(y_true, y_pred, average="macro", zero_division=0),
            "jaccard_eqst": jaccard_score(y_true, y_pred, average="micro", zero_division=0)
        }

class ResultAggregator:
    """Handles the statistical aggregation of benchmark results (calculating means)."""

    @staticmethod
    def calculate_step_means(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the mean of all metrics for each difficulty step across its iterations.
        """
        # Updated to include ALL 7 parameters in the grouping
        group_cols = ["step", "st", "k", "mean_cii", "std_cii", "mean_cij", "std_cij", "std_e"]
        summary = df.groupby(group_cols)[["accuracy", "jaccard_eqcluster", "jaccard_eqst"]].mean().reset_index()
        return summary

    @staticmethod
    def calculate_overall_means(df: pd.DataFrame) -> dict:
        """
        Calculates the overall mean metrics across the entire experiment (all steps and iterations).
        """
        return {
            "overall_accuracy": df["accuracy"].mean(),
            "overall_jaccard_eqcluster": df["jaccard_eqcluster"].mean(),
            "overall_jaccard_eqst": df["jaccard_eqst"].mean()
        }
