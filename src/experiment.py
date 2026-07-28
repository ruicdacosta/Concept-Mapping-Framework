import pandas as pd
from typing import List, Dict
from src.generator import SyntheticDataGenerator, GeneratorConfig
from src.algorithms import BaseCMAlgorithm
from src.evaluation import Evaluator

class ExperimentRunner:
    """Executes parameter evolution experiments sequentially from Easy to Hard."""
    
    def __init__(self, algorithm: BaseCMAlgorithm, steps_config: List[Dict], iterations: int = 5):
        self.algorithm = algorithm
        self.steps_config = steps_config
        self.iterations = iterations

    def run(self) -> pd.DataFrame:
        results = []
        
        for step_idx, params in enumerate(self.steps_config):
            for iteration in range(self.iterations):
                config = GeneratorConfig(**params)
                gen = SyntheticDataGenerator(config)
                gen.generate_matrices()
                
                Z_alg = self.algorithm.fit(gen.S, config.k)
                
                evaluator = Evaluator(gen.Z, Z_alg)
                metrics = evaluator.evaluate()
                
                record = {
                    "step": step_idx + 1,
                    **params,
                    "iteration": iteration,
                    **metrics
                }
                results.append(record)
                
        return pd.DataFrame(results)