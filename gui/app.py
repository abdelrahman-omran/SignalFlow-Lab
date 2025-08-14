import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

from processing.generation import generate_sine
from processing.arithmetic import multiply_signal
from processing.visualization import visualize_signal_mode
from processing.io import ensure_results_dir, save_result


class SignalProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Processor (Minimal)")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)

        # App state
        self.signals = []  # list of tuples: (indices, values)
        self.results_dir = ensure_results_dir(os.path.join(os.getcwd(), "results", "minimal"))

        # Layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self._create_toolbar()

        # Notebook (optional now, ready for future tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self._add_home_tab()

    def _create_toolbar(self):
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar.grid(row=0, column=0, sticky="ew")

        tk.Button(toolbar, text="Generate Sine", command=self.on_generate_sine).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(toolbar, text="Multiply ×C", command=self.on_multiply).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(toolbar, text="Plot (Continuous)", command=lambda: self.on_visualize("continuous")).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(toolbar, text="Plot (Discrete)", command=lambda: self.on_visualize("discrete")).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(toolbar, text="Clear Signals", command=self.on_clear).pack(side=tk.LEFT, padx=4, pady=4)

    def _add_home_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Home")
        lbl = ttk.Label(frame, text="Minimal refactor: Generate → Multiply → Visualize", anchor="center")
        lbl.pack(pady=20)

    # ===== Handlers =====

    def on_generate_sine(self):
        try:
            A = simpledialog.askfloat("Sine", "Amplitude (A):", minvalue=0.0)
            if A is None: return
            theta = simpledialog.askfloat("Sine", "Phase (θ in radians):", initialvalue=0.0)
            if theta is None: return
            f_analog = simpledialog.askfloat("Sine", "Analog frequency (Hz):", minvalue=0.0)
            if f_analog is None: return
            f_sampling = simpledialog.askfloat("Sine", "Sampling frequency (Hz):", minvalue=0.0)
            if f_sampling is None: return

            indices, values = generate_sine(A, theta, f_analog, f_sampling, duration=1.0)
            self.signals.append((indices, values))

            path = save_result("sine", indices, values, self.results_dir)
            messagebox.showinfo("Success", f"Sine generated and saved:\n{path}")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {e}")

    def on_multiply(self):
        if not self.signals:
            messagebox.showerror("Error", "No signal available. Generate a sine first.")
            return
        try:
            c = simpledialog.askfloat("Multiply", "Constant (C):")
            if c is None: return
            indices, values = multiply_signal(self.signals[-1], c)
            self.signals.append((indices, values))
            path = save_result("mul", indices, values, self.results_dir)
            messagebox.showinfo("Success", f"Signal multiplied and saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {e}")

    def on_visualize(self, mode):
        try:
            visualize_signal_mode(self.signals, mode=mode)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {e}")

    def on_clear(self):
        self.signals = []
        messagebox.showinfo("Cleared", "All signals cleared.")
