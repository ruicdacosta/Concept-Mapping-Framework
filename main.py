import argparse
import tkinter as tk

def launch_experiment_gui():
    """Starts the default experiment GUI with dynamic algorithm discovery."""
    from gui.app import ExperimentApp

    root = tk.Tk()
    app = ExperimentApp(root)
    root.mainloop()

def launch_math_gui():
    """Starts the framework strictly in Mathematical/Matrix viewing mode."""
    from gui.app import MathModeApp

    root = tk.Tk()
    app = MathModeApp(root)
    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concept Mapping Framework CLI")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["math"], 
        default=None,
        help="Use 'math' to open the matrix inspection view. Omit --mode to run the benchmark GUI."
    )
    
    args = parser.parse_args()
    
    if args.mode == "math":
        launch_math_gui()
    else:
        launch_experiment_gui()
