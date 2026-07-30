import argparse
import tkinter as tk

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
    root = tk.Tk()
    
    if args.mode == "math":
        from gui.app import MathModeApp
        app = MathModeApp(root)
    else:
        from gui.app import ExperimentApp
        app = ExperimentApp(root)

    root.mainloop()
