import numpy as np
from typing import Dict, List, Tuple

class BenchmarkGenerator:
    """Handles the interpolation and boundary rules for benchmark parameter evolution."""
    
    @staticmethod
    def generate_steps_config(bounds: Dict[str, Tuple[float, float]], steps: int) -> List[Dict]:
        steps_config = []
        interpolated = {}
        
        for key, (start_val, end_val) in bounds.items():
            interpolated[key] = np.linspace(start_val, end_val, steps)

        st_vals = np.clip(np.round(interpolated["st"]).astype(int), 3, 100)
        k_vals = np.round(interpolated["k"]).astype(int)

        for i in range(steps):
            current_st = int(st_vals[i])
            current_k = int(k_vals[i])
            
            # Enforce Rules: k > 1 (min 2) and k < st (max st - 1)
            current_k = max(2, min(current_k, current_st - 1))

            step_params = {
                "st": current_st,
                "k": current_k,
                "mean_cii": float(interpolated["mean_cii"][i]),
                "std_cii": float(interpolated["std_cii"][i]),
                "mean_cij": float(interpolated["mean_cij"][i]),
                "std_cij": float(interpolated["std_cij"][i]),
                "std_e": float(interpolated["std_e"][i])
            }
            steps_config.append(step_params)

        return steps_config