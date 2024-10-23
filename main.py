import os
import tkinter as tk
import numpy as np
from tkinter import ttk, filedialog, simpledialog
import matplotlib.pyplot as plt
from tkinter import messagebox

class SignalProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Processor")

        # Make app responsive
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # List to store signals
        self.signals = []

        # Create results directory if it doesn't exist
        self.results_dir = "./results/task2"
        os.makedirs(self.results_dir, exist_ok=True)

        # Create toolbar
        self.create_toolbar()

        # Create tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(sticky="nsew", row=1)  # Place tabs below the toolbar

        # Create the different tabs for tasks
        self.create_tabs()

    def create_toolbar(self):
        """Creates a toolbar for common operations."""
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar.grid(row=0, column=0, sticky="ew")

        # Toolbar buttons
        load_signal_btn = tk.Button(toolbar, text="Load Signal", command=self.load_signal)
        load_signal_btn.pack(side=tk.LEFT, padx=2, pady=2)

        tk.Button(toolbar, text="Visualize as Continuous", command=lambda: self.visualize_signal_mode("continuous")).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Visualize as Discrete", command=lambda: self.visualize_signal_mode("discrete")).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Clear Signals", command=self.clear_signals).pack(side=tk.LEFT, padx=2, pady=2)

    def create_tabs(self):
        # Tab 1: Task 1 with all operations
        task1_tab = ttk.Frame(self.notebook)
        self.notebook.add(task1_tab, text="Task 1")
        self.create_task1_tab(task1_tab)

        # Tab 2: Task 2 for signal generation
        task2_tab = ttk.Frame(self.notebook)
        self.notebook.add(task2_tab, text="Task 2")
        self.create_task2_tab(task2_tab)

    def create_task1_tab(self, tab):
        # Task 1 Tab Layout: Buttons for signal operations
        button_frame = ttk.Frame(tab)
        button_frame.pack(padx=10, pady=10, fill="both", expand=True)

        tk.Button(button_frame, text="Add Signals", command=self.add_signals).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(button_frame, text="Multiply Signal by Constant", command=self.multiply_signal).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(button_frame, text="Subtract Signals", command=self.subtract_signals).grid(row=1, column=0, padx=5, pady=5)
        tk.Button(button_frame, text="Shift Signal", command=self.shift_signal).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(button_frame, text="Reverse Signal", command=self.reverse_signal).grid(row=1, column=2, padx=5, pady=5)

    def create_task2_tab(self, tab):
        """Task 2 Tab Layout: Allows user to generate new signals."""
        button_frame = ttk.Frame(tab)
        button_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Add Signal Generation options in Task 2
        tk.Button(button_frame, text="Generate Sine Wave", command=lambda: self.generate_signal("sine")).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(button_frame, text="Generate Cosine Wave", command=lambda: self.generate_signal("cosine")).grid(row=0, column=1, padx=5, pady=5)

    def generate_signal(self, signal_type):
        """Generates a sinusoidal or cosinusoidal signal."""
        try:
            amplitude = float(simpledialog.askstring("Input", "Enter amplitude (A):"))
            theta = float(simpledialog.askstring("Input", "Enter phase shift (theta in radians):"))
            f_analog = float(simpledialog.askstring("Input", "Enter analog frequency (Hz):"))
            f_sampling = float(simpledialog.askstring("Input", "Enter sampling frequency (Hz):"))

            # Ensure that the sampling frequency obeys the sampling theorem
            if f_sampling < 2 * f_analog:
                messagebox.showerror("Error", "Sampling frequency must be at least 2 times the analog frequency!")
                return

            # Generate time points based on the sampling frequency
            t = np.arange(0, 1, 1 / f_sampling)  # 1 second duration

            # Generate the signal (either sine or cosine)
            if signal_type == "sine":
                signal = amplitude * np.sin(2 * np.pi * f_analog * t + theta)
            else:  # cosine
                signal = amplitude * np.cos(2 * np.pi * f_analog * t + theta)

            indices = list(range(len(signal)))  # Use the time indices for plotting
            self.signals.append((indices, signal))

            self.save_result(signal_type, indices, signal)
            messagebox.showinfo("Success", f"{signal_type.capitalize()} wave generated successfully!")
        
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter numeric values.")

    def load_signal(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                start_index = int(lines[1])
                N = int(lines[2])
                signal_data = [list(map(float, line.split())) for line in lines[3:3 + N]]

                indices = [item[0] for item in signal_data]
                signal = [item[1] for item in signal_data]

                self.signals.append((indices, signal))
                messagebox.showinfo("Success", f"Loaded signal with {N} samples.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load signal: {e}")

    def visualize_signal_mode(self, mode="continuous"):
        """Visualize the signal in either continuous or discrete mode"""
        if not self.signals:
            messagebox.showerror("Error", "No signal loaded!")
            return
        
        plt.figure()
        for idx, (indices, signal) in enumerate(self.signals):
            if mode == "discrete":
                plt.stem(indices, signal, label=f"Signal {idx + 1}")  # Removed use_line_collection=True
            else:
                plt.plot(indices, signal, label=f"Signal {idx + 1}")
        
        plt.title(f"Signal Visualization - {mode.capitalize()} Mode")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

    def add_signals(self):
        if len(self.signals) < 2:
            messagebox.showerror("Error", "At least two signals are required for addition!")
            return
        result_signal = {}
        for _, (indices, signal) in enumerate(self.signals):
            for i, val in zip(indices, signal):
                result_signal[i] = result_signal.get(i, 0) + val

        sorted_result = sorted(result_signal.items())
        indices, values = zip(*sorted_result) if sorted_result else ([], [])
        self.save_result("add", indices, values)
        messagebox.showinfo("Success", "Signals added successfully!")

    def multiply_signal(self):
        if not self.signals:
            messagebox.showerror("Error", "No signal loaded!")
            return
        try:
            constant = float(simpledialog.askstring("Input", "Enter constant to multiply:"))
            last_indices, last_signal = self.signals[-1]
            result_signal = [val * constant for val in last_signal]
            self.save_result("mul", last_indices, result_signal)
            messagebox.showinfo("Success", f"Signal multiplied by {constant} successfully!")
        except ValueError:
            messagebox.showerror("Error", "Invalid constant value!")

    def subtract_signals(self):
        if len(self.signals) < 2:
            messagebox.showerror("Error", "At least two signals are required for subtraction!")
            return
        result_signal = {}
        first_indices, first_signal = self.signals[0]
        second_indices, second_signal = self.signals[1]

        for i, val in zip(first_indices, first_signal):
            result_signal[i] = result_signal.get(i, 0) + val
        for i, val in zip(second_indices, second_signal):
            result_signal[i] = result_signal.get(i, 0) - val

        sorted_result = sorted(result_signal.items())
        indices, values = zip(*sorted_result) if sorted_result else ([], [])
        self.save_result("sub", indices, values)
        messagebox.showinfo("Success", "Signals subtracted successfully!")

    def shift_signal(self):
        if not self.signals:
            messagebox.showerror("Error", "No signal loaded!")
            return
        try:
            shift_amount = int(simpledialog.askstring("Input", "Enter shift amount (in samples):"))
            last_indices, last_signal = self.signals[-1]
            result_indices = [i + shift_amount for i in last_indices]
            self.save_result("shift", result_indices, last_signal)
            messagebox.showinfo("Success", f"Signal shifted by {shift_amount} samples successfully!")
        except ValueError:
            messagebox.showerror("Error", "Invalid shift amount!")

    def reverse_signal(self):
        if not self.signals:
            messagebox.showerror("Error", "No signal loaded!")
            return
        last_indices, last_signal = self.signals[-1]
        result_signal = list(reversed(last_signal))
        self.save_result("reverse", last_indices, result_signal)
        messagebox.showinfo("Success", "Signal reversed successfully!")

    def clear_signals(self):
        self.signals = []
        messagebox.showinfo("Success", "All signals cleared!")

    def save_result(self, operation, indices, values):
        result_file = os.path.join(self.results_dir, f"{operation}-result.txt")
        with open(result_file, 'w') as f:
            f.write("0\n")
            f.write("0\n")
            f.write(f"{len(indices)}\n")
            for idx, val in zip(indices, values):
                f.write(f"{idx} {val}\n")
        print(f"Saved result to {result_file}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SignalProcessorApp(root)
    root.mainloop()
