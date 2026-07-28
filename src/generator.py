import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class GeneratorConfig:
    st: int = 100
    k: int = 10
    mean_cii: float = 1.0
    std_cii: float = 0.0
    mean_cij: float = 0.0
    std_cij: float = 0.0
    std_e: float = 0.0

class SyntheticDataGenerator:
    """Generates synthetic similarity matrices for concept mapping framework."""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.Z: Optional[np.ndarray] = None
        self.C: Optional[np.ndarray] = None
        self.S0: Optional[np.ndarray] = None
        self.E: Optional[np.ndarray] = None
        self.S: Optional[np.ndarray] = None

    def generate_Z(self) -> np.ndarray:
        """Generates a random binary statement-to-cluster matrix Z."""
        st, k = self.config.st, self.config.k
        self.Z = np.zeros((st, k), dtype=int)
        
        for i in range(st):
            cluster = np.random.randint(0, k)
            self.Z[i, cluster] = 1
            
        return self.Z

    def set_Z(self, Z_matrix: np.ndarray):
        """Allows manual injection of a predefined Z matrix."""
        if Z_matrix.shape != (self.config.st, self.config.k):
            raise ValueError(f"Z matrix must be shape ({self.config.st}, {self.config.k})")
        self.Z = Z_matrix

    def generate_matrices(self):
        """Generates the C, S0, E, and final S matrices based on Z and config."""
        if self.Z is None:
            self.generate_Z()
            
        c = self.config
        k, st = c.k, c.st

        # 1. Cluster Covariance Matrix (C)
        self.C = np.zeros((k, k))
        for i in range(k):
            for j in range(i, k):
                if i == j:
                    val = np.random.normal(c.mean_cii, c.std_cii)
                else:
                    val = np.random.normal(c.mean_cij, c.std_cij)
                self.C[i, j] = val
                self.C[j, i] = val
        self.C = np.round(self.C, 3)

        # 2. Base Similarity Matrix (S0)
        self.S0 = np.round(self.Z @ self.C @ self.Z.T, 3)

        # 3. Error Matrix (E)
        self.E = np.zeros((st, st))
        for i in range(st):
            for j in range(i + 1, st):
                noise = np.random.normal(0, c.std_e)
                self.E[i, j] = noise
                self.E[j, i] = noise 
        self.E = np.round(self.E, 3)

        # 4. Final Noisy Similarity Matrix (S)
        self.S = self.S0 + self.E
        np.fill_diagonal(self.S, 1.0)
        self.S = np.round(self.S, 3)