import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import importlib
import inspect
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from processing.io import *
# Example imports (replace with actual functions in each package)
from processing.generation import *
from processing.basic_ops import *

PACKAGE_FUNCTIONS = {
    "Generation": ["sine", "cosine"],
    "basic_ops": ["add", "subtract", "multiply", "shift", "reverse"],
    "signal_digitize": ["sampling", "quantization"],

}


class SignalProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Processor (Tabbed)")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)

        self.signals = []  # list of tuples: (indices, values)
        self.results_dir = ensure_results_dir(os.path.join(os.getcwd(), "results"))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self._create_tabs()

    def _create_tabs(self):
        # Function tabs
        for package_name, functions in PACKAGE_FUNCTIONS.items():
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=package_name)
            frame.columnconfigure(0, weight=1)

            for func_name in functions:
                btn = ttk.Button(frame, text=func_name.replace("_", " ").title(),
                                 command=lambda f=func_name, p=package_name: self._run_function(p, f))
                btn.pack(fill="x", padx=10, pady=5)

        # Visualization tab
        self._create_visualization_tab()

    def _run_function(self, package_name, func_name):
        try:
            module_name = f"processing.{package_name.lower().replace(' ', '_')}.{func_name}"
            module = importlib.import_module(module_name)

            # Get the first callable function in the module
            func = None
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj):
                    func = obj
                    break

            if not func:
                messagebox.showerror("Error", f"No callable found in {func_name}")
                return

            params = {}

            if package_name in ["basic_ops", "Discrete Ops"]:
                sig = inspect.signature(func)

                for param_name in sig.parameters.keys():
                    # Ask for input file for signals
                    file_path = filedialog.askopenfilename(
                        title=f"Select {param_name} file",
                        initialdir=self.results_dir,
                        filetypes=[("Text Files", "*.txt")]
                    )
                    if not file_path:
                        return

                    indices, values = [], []
                    with open(file_path, 'r') as f:
                        lines = f.readlines()[3:]  # skip metadata
                        for line in lines:
                            i, v = line.strip().split()
                            indices.append(float(i))
                            values.append(float(v))

                    params[param_name] = (indices, values)

                # Ask for extra numeric parameters if needed (like rate, step, etc.)
                for param_name, param in sig.parameters.items():
                    if param_name not in params:
                        value = simpledialog.askfloat("Input Required", f"Enter value for {param_name}:")
                        if value is None:
                            return
                        params[param_name] = value

            else:
                # for generation (still ask numbers, maybe a file if needed)
                sig = inspect.signature(func)
                for param_name, param in sig.parameters.items():
                    if "signal" in param_name:
                        file_path = filedialog.askopenfilename(
                            title=f"Select {param_name} file",
                            initialdir=self.results_dir,
                            filetypes=[("Text Files", "*.txt")]
                        )
                        if not file_path:
                            return

                        indices, values = [], []
                        with open(file_path, 'r') as f:
                            lines = f.readlines()[3:]
                            for line in lines:
                                i, v = line.strip().split()
                                indices.append(float(i))
                                values.append(float(v))

                        params[param_name] = (indices, values)
                    else:
                        value = simpledialog.askfloat("Input Required", f"Enter value for {param_name}:")
                        if value is None:
                            return
                        params[param_name] = value

            # run the function
            result = func(**params)

            if result:
                self.signals.append(result)
                path = save_result(func_name, result[0], result[1], self.results_dir)
                messagebox.showinfo("Success", f"{func_name} executed and saved:\n{path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to run {func_name}: {e}")

    # ---------------- VISUALIZATION ----------------
    def _create_visualization_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Visualization")

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)

        self.file_listbox = tk.Listbox(frame, height=15)
        self.file_listbox.grid(row=0, column=0, sticky="ns", padx=5, pady=5)

        button_frame = tk.Frame(frame)
        button_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        ttk.Button(button_frame, text="Refresh", command=self._refresh_file_list).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Plot", command=self._plot_selected_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Clear", command=self._clear_plot).pack(side=tk.LEFT, padx=2)

        # Matplotlib figure area
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Signal Plot")
        self.ax.set_xlabel("Index")
        self.ax.set_ylabel("Value")
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().grid(row=0, column=1, rowspan=2, sticky="nsew", padx=5, pady=5)

        self._refresh_file_list()

    def _refresh_file_list(self):
        self.file_listbox.delete(0, tk.END)
        if os.path.exists(self.results_dir):
            for file in os.listdir(self.results_dir):
                if file.endswith(".txt"):
                    self.file_listbox.insert(tk.END, file)

    def _load_signal_file(self, file_path):
        indices, values = [], []
        with open(file_path, 'r') as f:
            lines = f.readlines()[3:]  # Skip first 3 metadata lines
            for line in lines:
                i, v = line.strip().split()
                indices.append(float(i))
                values.append(float(v))
        return indices, values

    def _plot_selected_file(self):
        selection = self.file_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "No file selected.")
            return

        file_name = self.file_listbox.get(selection[0])
        file_path = os.path.join(self.results_dir, file_name)

        indices, values = self._load_signal_file(file_path)

        self.ax.clear()
        self.ax.plot(indices, values)
        self.ax.set_title(f"Signal: {file_name}")
        self.ax.set_xlabel("Index")
        self.ax.set_ylabel("Value")
        self.ax.grid(True)

        self.canvas.draw()

    def _clear_plot(self):
        self.ax.clear()
        self.ax.set_title("Signal Plot")
        self.ax.set_xlabel("Index")
        self.ax.set_ylabel("Value")
        self.canvas.draw()



if __name__ == "__main__":
    root = tk.Tk()
    app = SignalProcessorApp(root)
    root.mainloop()
