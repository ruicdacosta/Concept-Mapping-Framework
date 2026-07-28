from abc import ABC, abstractmethod
import numpy as np
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

class BaseCMAlgorithm(ABC):
    """Abstract base class for Concept Mapping clustering algorithms."""
    
    @abstractmethod
    def fit(self, S: np.ndarray, k: int) -> np.ndarray:
        """
        Fits the clustering algorithm to the similarity matrix S.
        
        Args:
            S: The (st x st) similarity matrix.
            k: The expected number of clusters.
            
        Returns:
            Z_alg: The predicted (st x k) binary assignment matrix.
        """
        pass

class RandomClusteringAlgorithm(BaseCMAlgorithm):
    """A dummy clustering algorithm for testing purposes."""
    
    def fit(self, S: np.ndarray, k: int) -> np.ndarray:
        st = S.shape[0]
        Z_alg = np.zeros((st, k), dtype=int)
        for i in range(st):
            cluster = np.random.randint(0, k)
            Z_alg[i, cluster] = 1
        return Z_alg

class TrochimMethodAlgorithm(BaseCMAlgorithm):
    """
    Implementation of the traditional Trochim Concept Mapping method.
    
    Pipeline:
    1. Convert similarity matrix S to dissimilarity matrix D.
    2. Perform Non-metric MDS (2 dimensions) on D.
    3. Perform Hierarchical Agglomerative Clustering (Ward's method) on MDS coordinates.
    """
    
    def fit(self, S: np.ndarray, k: int) -> np.ndarray:
        st = S.shape[0]
        
        # 1. Convert Similarity to Dissimilarity
        max_sim = np.max(S)
        D = max_sim - S
        np.fill_diagonal(D, 0)
        
        # 2. Non-metric MDS (Using updated 'metric' parameter to avoid scikit-learn deprecation warnings)
        mds = MDS(
            n_components=2, 
            metric=False,              # Replaces metric_mds=False
            dissimilarity='precomputed',# Legacy fallback compatibility if needed, but scikit-learn uses 'metric'
            random_state=42,
            init='random'
        )
        # Clean scikit-learn call using 'metric' keyword
        mds = MDS(
            n_components=2,
            metric='precomputed',     # 'precomputed' tells MDS the input is already a distance matrix
            n_init=4,
            random_state=42,
            init='random'
        )
        mds_coords = mds.fit_transform(D)
        
        # 3. Hierarchical Clustering (Ward's method)
        hc = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
        labels = hc.fit_predict(mds_coords)
        
        # 4. Convert cluster labels to a binary Z_alg matrix
        Z_alg = np.zeros((st, k), dtype=int)
        Z_alg[np.arange(st), labels] = 1
        
        return Z_alg

class PeladeauMethodAlgorithm(BaseCMAlgorithm):
    """
    Implementation of the Péladeau et al. (2017) Concept Mapping method.
    
    Pipeline:
    1. Convert similarity matrix S to dissimilarity matrix D.
    2. Perform Hierarchical Cluster Analysis (HCA) directly on the original 
       distance matrix D to avoid MDS dimensional distortion.
    3. Uses "weighted" average linkage as suggested in their experimental setup.
    """
    
    def fit(self, S: np.ndarray, k: int) -> np.ndarray:
        st = S.shape[0]
        
        # 1. Convert Similarity to Dissimilarity
        max_sim = np.max(S)
        D = max_sim - S
        np.fill_diagonal(D, 0)
        
        # scipy's linkage requires a condensed 1D distance matrix
        condensed_D = squareform(D)
        
        # 2. Hierarchical Clustering (Weighted average linkage)
        Z_linkage = linkage(condensed_D, method='weighted')
        
        # Extract flat cluster labels (fcluster returns 1-indexed arrays)
        labels = fcluster(Z_linkage, k, criterion='maxclust')
        labels = labels - 1 # Convert to 0-indexed for Python
        
        # 3. Convert cluster labels to a binary Z_alg matrix
        Z_alg = np.zeros((st, k), dtype=int)
        Z_alg[np.arange(st), labels] = 1
        
        return Z_alg