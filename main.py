import argparse
import tkinter as tk
from gui.app import ExperimentApp, MathModeApp

def launch_experiment_gui():
    """Starts the default Batch Experiment GUI with dynamic algorithm discovery."""
    root = tk.Tk()
    app = ExperimentApp(root)
    root.mainloop()

def launch_math_gui():
    """Starts the framework strictly in Mathematical/Matrix viewing mode."""
    root = tk.Tk()
    app = MathModeApp(root)
    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concept Mapping Framework CLI")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["batch", "math"], 
        default="batch",
        help="Default 'batch' runs the experiment runner. 'math' opens the matrix inspection view."
    )
    
    args = parser.parse_args()
    
    if args.mode == "math":
        launch_math_gui()
    else:
        launch_experiment_gui()