# Concept Mapping Algorithm Benchmarking Framework

## Overview and Context

Concept mapping combines qualitative and quantitative statistical analysis to help identify, prioritize, and relate the components of a given reality. This framework provides an objective, mathematical environment to test and benchmark clustering algorithms against a known ground truth using synthetic data.

---

## The Mathematical Generation Pipeline
 
To evaluate clustering performance, the framework generates synthetic, parameterized concept mapping data. The generation process relies on configurable input parameters and produces a final noisy similarity matrix, S, which mimics human sorting behavior.

### 1. The Ground Truth Matrix (Z)
The framework generates a binary statement-to-cluster assignment matrix Z of shape (st x k), where st is the number of statements and k is the number of clusters. A value of 1 indicates a statement belongs to a cluster.

### 2. The Cluster Covariance Matrix (C)
A covariance matrix C of shape (k x k) is generated to define the relationships between clusters. The diagonal values (within-cluster similarities) are drawn from a normal distribution defined by mean_cii and std_cii. The off-diagonal values (between-cluster similarities) are defined by mean_cij and std_cij.

### 3. The Base Similarity Matrix (S0)
The true, noise-free relationship between all statements is computed using matrix multiplication:

S0 = Z * C * Z^T

### 4. The Error Matrix (E)
To simulate human disagreement and sorting noise, an error matrix E of shape (st x st) is generated. Values are drawn from a normal distribution with a mean of 0 and a standard deviation of std_e.

### 5. The Final Similarity Matrix (S)
The algorithm receives the final noisy similarity matrix S:

S = S0 + E

*(Note: The diagonal of S is forced to 1.0 to reflect perfect self-similarity.)*

---

## Implementing Custom Algorithms

The framework relies on Object-Oriented Programming to provide a plug-and-play environment for researchers. You do not need to alter the data generation, the GUI, or the evaluation logic to add a new method.

### Steps for Integration
1. Open `src/algorithms.py`.
2. Create a new class that inherits from the `BaseCMAlgorithm` abstract base class.
3. Implement the `fit(self, S, k)` method.
    * **Inputs:** The method will receive the (st x st) noisy similarity matrix S and the expected number of clusters k.
    * **Outputs:** The method must return a binary prediction matrix Z_alg of shape (st x k).

**Example Template:**
```python
from src.algorithms import BaseCMAlgorithm
import numpy as np

class MyNovelAlgorithm(BaseCMAlgorithm):
    def fit(self, S: np.ndarray, k: int) -> np.ndarray:
        st = S.shape[0]
        
        # ... Insert custom clustering logic here ...
        
        Z_alg = np.zeros((st, k), dtype=int)
        # Populate Z_alg with 1s based on your clustering results
        return Z_alg
```

Once your class is added to `src/algorithms.py`, the GUI uses the Python `inspect` module to automatically discover it. It will immediately appear in the UI dropdown menu without requiring any changes to the core application files.


## Configuration and the experiment_settings.json

Default parameters for the batch experiments are decoupled from the code and stored in `experiment_settings.json`. 

* **What to change:** You can modify the `default_algorithm` string to automatically load your new algorithm upon startup. You can also alter the start and end bounds for the 7 evolution parameters (e.g., st, k, std_e).

---

## Running the Framework

Launch the framework via command line:
* **Batch Benchmark Mode:** `python main.py` or `python main.py --mode batch`
* **Mathematical Explorer Mode:** `python main.py --mode math`

---

## Evaluation Metrics and Exporting

Because clustering algorithms assign arbitrary labels to clusters, predicting exact column matches between the ground truth $Z$ and the predicted $Z_{alg}$ is a label permutation problem solved automatically via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`).

### Evaluated Metrics:
* **Accuracy:** Standard classification accuracy of statement assignments.
* **Jaccard Macro:** Average Jaccard Index across all clusters (giving equal weight to small and large clusters).
* **Jaccard Micro:** Global Jaccard Index across all pooled statements (favoring larger clusters).

### Exporting Results:
Once a benchmark run is completed in the GUI, you can click the **"⬇ Export JSON"** button to export a structured file named following the convention `CMAP-[AlgorithmName]-[YYYYMMDD].json`. The exported JSON file contains:
1. The selected algorithm name.
2. The overall mean performance metrics across the entire experiment.
3. The step-by-step averaged performance records.
4. The complete raw iteration logs.