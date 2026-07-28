import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.manifold import MDS
import pandas as pd
import json
import os
import sys
import inspect
from datetime import datetime

import src.algorithms as alg_module
from src.algorithms import BaseCMAlgorithm
from src.generator import SyntheticDataGenerator, GeneratorConfig
from src.experiment import ExperimentRunner
from src.generation_benchmark import BenchmarkGenerator
from src.evaluation import ResultAggregator

class ExperimentApp:
    """Modern GUI for running parameter-evolution experiments with ranking metrics."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Concept Mapping - Algorithm Benchmark")
        self.root.geometry("1100x860")
        self.root.configure(bg="#f8fafc")
        
        self.settings = self.load_settings()
        self.available_algs = self.get_available_algorithms()
        
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.configure_styles()
        self.setup_ui()

    def load_settings(self) -> dict:
        settings_path = "experiment_settings.json"
        if os.path.exists(settings_path):
            try:
                with open(settings_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Warning: Invalid JSON format. Loading defaults.")
        return {}

    def get_available_algorithms(self) -> dict:
        alg_classes = {}
        for name, obj in inspect.getmembers(alg_module, inspect.isclass):
            if issubclass(obj, BaseCMAlgorithm) and obj is not BaseCMAlgorithm:
                alg_classes[name] = obj
        return alg_classes

    def configure_styles(self):
        self.style.configure(".", background="#f8fafc", font=("Segoe UI", 10))
        self.style.configure("Card.TFrame", background="#ffffff", relief="flat", borderwidth=1)
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), background="#f8fafc", foreground="#0f172a")
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 11, "bold"), background="#ffffff", foreground="#334155")
        self.style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), background="#2563eb", foreground="#ffffff")
        self.style.map("Primary.TButton", background=[("active", "#1d4ed8")])
        self.style.configure("Treeview", font=("Segoe UI", 9), rowheight=26, background="#ffffff", fieldbackground="#ffffff")
        self.style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), background="#e2e8f0", foreground="#1e293b")

    def setup_ui(self):
        main_container = ttk.Frame(self.root, padding=20)
        main_container.pack(fill=tk.BOTH, expand=True)

        # -----------------------------------------------------
        # 1. HEADER CARD
        # -----------------------------------------------------
        header_frame = ttk.Frame(main_container)
        header_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 15))
        ttk.Label(header_frame, text="Select Algorithm:", style="Header.TLabel").pack(side=tk.LEFT)
        
        self.alg_combo = ttk.Combobox(header_frame, values=list(self.available_algs.keys()), state="readonly", width=35, font=("Segoe UI", 11))
        self.alg_combo.pack(side=tk.LEFT, padx=15)
        
        default_alg = self.settings.get("default_algorithm", "")
        if default_alg in self.available_algs:
            self.alg_combo.set(default_alg)
        elif self.available_algs:
            self.alg_combo.current(0)

        # -----------------------------------------------------
        # 2. CONFIGURATION CARD
        # -----------------------------------------------------
        config_card = ttk.Frame(main_container, style="Card.TFrame", padding=15)
        config_card.pack(side=tk.TOP, fill=tk.X, pady=(0, 15))

        ttk.Label(config_card, text="Parameter Bounds (Start → End)", style="SubHeader.TLabel").grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 10))

        self.inputs = {}
        json_params = self.settings.get("parameters", {})
        
        # Tuple format: (Label, Key, Default Start, Default End)
        param_keys = [
            ("st (Statements ≤ 100)", "st", "20", "100"),
            ("k (Clusters)", "k", "3", "12"),
            ("mean_cii (within-cluster)", "mean_cii", "1.0", "0.6"),
            ("std_cii (within-cluster)", "std_cii", "0.0", "0.25"),
            ("mean_cij (between-clusters)", "mean_cij", "0.0", "0.4"),
            ("std_cij (between-clusters)", "std_cij", "0.0", "0.25"),
            ("std_e (error)", "std_e", "0.01", "0.5")
        ]

        ttk.Label(config_card, text="Parameter", font=("Segoe UI", 9, "bold"), background="#ffffff").grid(row=1, column=0, sticky="w")
        ttk.Label(config_card, text="Start", font=("Segoe UI", 9, "bold"), background="#ffffff").grid(row=1, column=1, sticky="w")
        ttk.Label(config_card, text="End", font=("Segoe UI", 9, "bold"), background="#ffffff").grid(row=1, column=2, sticky="w")

        for idx, (label, key, def_start, def_end) in enumerate(param_keys, start=2):
            ttk.Label(config_card, text=label, background="#ffffff").grid(row=idx, column=0, sticky="w", pady=2, padx=(0, 10))
            
            # Fetch from JSON if exists, otherwise use the realistic defaults defined above
            start_val = str(json_params.get(label, {}).get("start", def_start))
            end_val = str(json_params.get(label, {}).get("end", def_end))
            
            start_entry = ttk.Entry(config_card, width=8)
            start_entry.insert(0, start_val)
            start_entry.grid(row=idx, column=1, pady=2, padx=5)
            
            end_entry = ttk.Entry(config_card, width=8)
            end_entry.insert(0, end_val)
            end_entry.grid(row=idx, column=2, pady=2, padx=5)
            
            self.inputs[key] = (start_entry, end_entry)

        run_settings = self.settings.get("run_settings", {})
        
        ttk.Label(config_card, text="#Steps:", background="#ffffff", font=("Segoe UI", 9, "bold")).grid(row=1, column=4, sticky="w", padx=(30, 5))
        self.steps_entry = ttk.Entry(config_card, width=8)
        self.steps_entry.insert(0, str(run_settings.get("steps", 5)))
        self.steps_entry.grid(row=1, column=5, sticky="w")

        ttk.Label(config_card, text="#Runs per Step:", background="#ffffff", font=("Segoe UI", 9, "bold")).grid(row=2, column=4, sticky="w", padx=(30, 5))
        self.iters_entry = ttk.Entry(config_card, width=8)
        self.iters_entry.insert(0, str(run_settings.get("iterations_per_step", 5)))
        self.iters_entry.grid(row=2, column=5, sticky="w")

        run_btn = ttk.Button(config_card, text="▶ Run Benchmark", style="Primary.TButton", command=self.run_experiment)
        run_btn.grid(row=4, column=4, columnspan=2, sticky="ew", padx=(30, 0), pady=10)

        # -----------------------------------------------------
        # 3. SUMMARY CARD (Pinned to the BOTTOM)
        # -----------------------------------------------------
        summary_card = ttk.Frame(main_container, style="Card.TFrame", padding=15)
        summary_card.pack(side=tk.BOTTOM, fill=tk.X, pady=(15, 0))

        ttk.Label(summary_card, text="Overall Average Performance", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 5))
        
        bottom_row = tk.Frame(summary_card, bg="#ffffff")
        bottom_row.pack(fill=tk.X, expand=True)

        self.metrics_lbl = tk.Label(
            bottom_row, 
            text="Run the benchmark to calculate overall average metrics.",
            font=("Segoe UI", 11),
            bg="#ffffff",
            fg="#475569"
        )
        self.metrics_lbl.pack(side=tk.LEFT, anchor="w")

        self.export_btn = tk.Button(
            bottom_row, 
            text="⬇ Export JSON", 
            command=self.export_results,
            bg="#2563eb",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            state=tk.DISABLED,
            padx=15,
            pady=5
        )
        self.export_btn.pack(side=tk.RIGHT)

        # -----------------------------------------------------
        # 4. TABLE CARD (Takes remaining space)
        # -----------------------------------------------------
        table_card = ttk.Frame(main_container, style="Card.TFrame", padding=15)
        table_card.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        ttk.Label(table_card, text="Step Execution Results", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 10))

        tree_container = ttk.Frame(table_card)
        tree_container.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(tree_container, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        columns = ("step", "st", "k", "mean_cii", "std_cii", "mean_cij", "std_cij", "std_e", "accuracy", "jaccard_macro", "jaccard_micro")
        
        self.tree = ttk.Treeview(tree_container, columns=columns, show="headings", height=6, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.tree.yview)

        col_settings = {
            "step": ("Step", 50), "st": ("st", 50), "k": ("k", 50), 
            "mean_cii": ("m_cii", 70), "std_cii": ("s_cii", 70), 
            "mean_cij": ("m_cij", 70), "std_cij": ("s_cij", 70), "std_e": ("std_e", 70),
            "accuracy": ("Accuracy", 80), "jaccard_macro": ("Jacc.(Macro)", 100), "jaccard_micro": ("Jacc.(Micro)", 100)
        }
        
        for col, (head, width) in col_settings.items():
            self.tree.heading(col, text=head)
            self.tree.column(col, anchor="center", width=width)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def run_experiment(self):
        try:
            selected_alg_name = self.alg_combo.get()
            if not selected_alg_name:
                messagebox.showwarning("Warning", "Please select an algorithm.")
                return
                
            alg_class = self.available_algs[selected_alg_name]
            algorithm_instance = alg_class()

            steps = int(self.steps_entry.get())
            iters = int(self.iters_entry.get())
            
            bounds = {}
            for key, (start_entry, end_entry) in self.inputs.items():
                bounds[key] = (float(start_entry.get()), float(end_entry.get()))

            steps_config = BenchmarkGenerator.generate_steps_config(bounds, steps)

            runner = ExperimentRunner(algorithm=algorithm_instance, steps_config=steps_config, iterations=iters)
            df = runner.run()

            summary = ResultAggregator.calculate_step_means(df)
            overall = ResultAggregator.calculate_overall_means(df)

            for item in self.tree.get_children():
                self.tree.delete(item)

            for _, row in summary.iterrows():
                self.tree.insert("", tk.END, values=(
                    int(row["step"]),
                    int(row["st"]),
                    int(row["k"]),
                    f"{row['mean_cii']:.2f}",
                    f"{row['std_cii']:.2f}",
                    f"{row['mean_cij']:.2f}",
                    f"{row['std_cij']:.2f}",
                    f"{row['std_e']:.2f}",
                    f"{row['accuracy']:.3f}",
                    f"{row['jaccard_macro']:.3f}",
                    f"{row['jaccard_micro']:.3f}"
                ))

            summary_text = (
                f"Overall Mean Accuracy: {overall['overall_accuracy']:.3f}   |   "
                f"Overall Mean Jaccard (Macro): {overall['overall_jaccard_macro']:.3f}   |   "
                f"Overall Mean Jaccard (Micro): {overall['overall_jaccard_micro']:.3f}"
            )
            self.metrics_lbl.config(text=summary_text, fg="#0f172a", font=("Segoe UI", 11, "bold"))

            # Save state for the Export Button
            self.latest_df = df
            self.latest_summary = summary
            self.latest_overall = overall
            self.export_btn.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Execution Error", f"Failed to run benchmark: {str(e)}")

    def export_results(self):
        if not hasattr(self, 'latest_df') or self.latest_df is None:
            return
        
        # Format the default filename: CMAP-<Algorithm>-<YYYYMMDD>
        alg_name = self.alg_combo.get()
        date_str = datetime.now().strftime("%Y%m%d")
        default_filename = f"CMAP-{alg_name}-{date_str}.json"
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile=default_filename,
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            title="Export Benchmark Results"
        )
        
        if not filepath:
            return
            
        try:
            export_data = {
                "algorithm": alg_name,
                "overall_means": self.latest_overall,
                "step_averages": json.loads(self.latest_summary.to_json(orient="records")),
                "raw_iterations": json.loads(self.latest_df.to_json(orient="records"))
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=4)
                
            messagebox.showinfo("Success", f"Data exported successfully to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")


class MathModeApp:
    """Mathematical Mode GUI for inspecting matrix generation math directly."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Concept Mapping - Mathematical Mode")
        self.root.geometry("1400x800")
        self.generator = None
        self.setup_ui()

    def setup_ui(self):
        left_frame = tk.Frame(self.root, width=300, padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        middle_frame = tk.Frame(self.root, padx=10, pady=10)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = tk.Frame(self.root, width=400, padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(left_frame, text="1Configuration", font=("Arial", 12, "bold")).pack(pady=5)

        self.inputs = {}
        labels = ["st", "k", "mean_cii", "std_cii", "mean_cij", "std_cij", "std_e"]
        defaults = ["100", "10", "1.0", "0.0", "0.0", "0.0", "0.0"] 

        for label, default in zip(labels, defaults):
            frame = tk.Frame(left_frame)
            frame.pack(fill=tk.X, pady=2)
            tk.Label(frame, text=label, width=15, anchor="w").pack(side=tk.LEFT)
            entry = tk.Entry(frame, width=10)
            entry.insert(0, default)
            entry.pack(side=tk.RIGHT)
            self.inputs[label] = entry

        tk.Label(left_frame, text="Matrix Z (Manual or Auto):", font=("Arial", 10, "bold")).pack(pady=(15, 2))
        self.z_text = tk.Text(left_frame, height=8, width=30)
        self.z_text.pack()
        
        tk.Button(left_frame, text="Generate Random Z", command=self.generate_Z_ui, bg="#e0e0e0").pack(pady=5, fill=tk.X)
        
        tk.Label(left_frame, text="2. Process", font=("Arial", 12, "bold")).pack(pady=(20, 5))
        tk.Button(left_frame, text="Generate Matrices", command=self.generate_matrices_ui, bg="#cce5ff").pack(pady=5, fill=tk.X)

        tk.Label(left_frame, text="3. Visualization", font=("Arial", 12, "bold")).pack(pady=(20, 5))
        self.mds_target = tk.StringVar(value="S0")
        tk.Radiobutton(left_frame, text="Base Similarity (S0)", variable=self.mds_target, value="S0").pack(anchor="w")
        tk.Radiobutton(left_frame, text="Noisy Similarity (S)", variable=self.mds_target, value="S").pack(anchor="w")
        tk.Button(left_frame, text="Plot MDS", command=self.plot_mds, bg="#d4edda").pack(pady=5, fill=tk.X)

        tk.Label(middle_frame, text="Non-metric MDS Visualization", font=("Arial", 12, "bold")).pack(pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=middle_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.stress_label = tk.Label(middle_frame, text="Stress Level: N/A", font=("Arial", 10, "italic"), fg="blue")
        self.stress_label.pack(side=tk.BOTTOM, anchor="e", pady=5)

        tk.Label(right_frame, text="Data Explorer", font=("Arial", 12, "bold")).pack(pady=5)

        self.right_canvas = tk.Canvas(right_frame)
        scrollbar = tk.Scrollbar(right_frame, orient="vertical", command=self.right_canvas.yview)
        self.scrollable_frame = tk.Frame(self.right_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))
        )
        self.right_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.right_canvas.configure(yscrollcommand=scrollbar.set)
        self.right_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.text_displays = {}
        for mat_name in ["C", "S0", "E", "S"]:
            frame = tk.Frame(self.scrollable_frame)
            frame.pack(fill=tk.X, pady=(10, 0))
            tk.Label(frame, text=f"Matrix {mat_name}", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
            tk.Button(frame, text="🔍 Expand", font=("Arial", 8), command=lambda m=mat_name: self.expand_matrix(m)).pack(side=tk.RIGHT)

            txt = tk.Text(self.scrollable_frame, height=6, width=45, wrap=tk.NONE)
            txt.pack(fill=tk.X)
            self.text_displays[mat_name] = txt

    def get_config(self) -> GeneratorConfig:
        try:
            return GeneratorConfig(
                st=int(self.inputs["st"].get()),
                k=int(self.inputs["k"].get()),
                mean_cii=float(self.inputs["mean_cii"].get()),
                std_cii=float(self.inputs["std_cii"].get()),
                mean_cij=float(self.inputs["mean_cij"].get()),
                std_cij=float(self.inputs["std_cij"].get()),
                std_e=float(self.inputs["std_e"].get())
            )
        except ValueError:
            messagebox.showerror("Input Error", "Ensure all parameters are valid numbers.")
            return None

    def generate_Z_ui(self):
        config = self.get_config()
        if not config: return
        self.generator = SyntheticDataGenerator(config)
        Z = self.generator.generate_Z()
        self.z_text.delete(1.0, tk.END)
        for row in Z:
            self.z_text.insert(tk.END, " ".join(map(str, row)) + "\n")

    def parse_Z_from_ui(self) -> bool:
        text = self.z_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Matrix Z is empty.")
            return False
        try:
            rows = text.split("\n")
            Z = np.array([list(map(int, row.split())) for row in rows if row.strip()])
            config = self.get_config()
            if Z.shape != (config.st, config.k):
                messagebox.showerror("Error", f"Z shape mismatch. Expected ({config.st}, {config.k})")
                return False
            if not self.generator:
                self.generator = SyntheticDataGenerator(config)
            self.generator.set_Z(Z)
            return True
        except Exception:
            messagebox.showerror("Error", "Invalid Z format.")
            return False

    def generate_matrices_ui(self):
        if not self.get_config() or not self.parse_Z_from_ui(): return
        self.generator.generate_matrices()
        self.update_matrix_display()

    def update_matrix_display(self):
        np.set_printoptions(precision=3, suppress=True, threshold=100)
        matrices = {
            "C": self.generator.C, 
            "S0": self.generator.S0, 
            "E": self.generator.E, 
            "S": self.generator.S
        }
        for name, mat in matrices.items():
            self.text_displays[name].delete(1.0, tk.END)
            if mat is not None:
                self.text_displays[name].insert(tk.END, str(mat))

    def plot_mds(self):
        if not self.generator or self.generator.S is None:
            messagebox.showwarning("Warning", "Generate matrices first.")
            return

        target = self.generator.S0 if self.mds_target.get() == "S0" else self.generator.S
        dissimilarity = np.max(target) - target
        np.fill_diagonal(dissimilarity, 0) 

        mds = MDS(
            n_components=2, 
            metric_mds=False, 
            dissimilarity='precomputed', 
            random_state=42,
            init='random'
        )
        coords = mds.fit_transform(dissimilarity)

        self.stress_label.config(text=f"Stress Level: {mds.stress_:.4f}")

        clusters = np.argmax(self.generator.Z, axis=1)
        colors = plt.cm.tab10(clusters % 10)

        self.ax.clear()
        self.ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=100, edgecolors='k', zorder=2)
        for i in range(len(coords)):
            self.ax.annotate(f"{i}", (coords[i, 0], coords[i, 1]), xytext=(4,4), textcoords='offset points')

        self.ax.set_title(f"MDS of {self.mds_target.get()}")
        self.ax.grid(True, linestyle='--', alpha=0.6, zorder=1)
        self.canvas.draw()

    def expand_matrix(self, mat_name):
        matrices = {"C": self.generator.C, "S0": self.generator.S0, "E": self.generator.E, "S": self.generator.S}
        mat = matrices.get(mat_name)
        if mat is None: return

        top = tk.Toplevel(self.root)
        top.title(f"Full Matrix {mat_name}")
        txt = tk.Text(top, wrap=tk.NONE, font=("Courier", 10))
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        txt.insert(tk.END, np.array2string(mat, threshold=sys.maxsize, max_line_width=sys.maxsize))
        txt.config(state=tk.DISABLED)