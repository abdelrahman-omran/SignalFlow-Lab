import os
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import messagebox

class SignalProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Processor")

        # List to store signals
        self.signals = []

        # Create results directory if it doesn't exist
        self.results_dir = "./results"
        os.makedirs(self.results_dir, exist_ok=True)

        # Create tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        # Create the different tabs for tasks
        self.create_tabs()

    def create_tabs(self):
        # Tab 1: Signal Operations
        signal_operations_tab = ttk.Frame(self.notebook)
        self.notebook.add(signal_operations_tab, text="Signal Operations")
        self.create_signal_operations_tab(signal_operations_tab)

        # Tab 2: Visualize Signal
        visualize_signal_tab = ttk.Frame(self.notebook)
        self.notebook.add(visualize_signal_tab, text="Visualize Signal")
        self.create_visualize_signal_tab(visualize_signal_tab)

    def create_signal_operations_tab(self, tab):
        # Add buttons for signal operations in this tab
        tk.Button(tab, text="Load Signal", command=self.load_signal).pack(pady=10)
        tk.Button(tab, text="Add Signals", command=self.add_signals).pack(pady=10)
        tk.Button(tab, text="Multiply Signal by Constant", command=self.multiply_signal).pack(pady=10)
        tk.Button(tab, text="Subtract Signals", command=self.subtract_signals).pack(pady=10)
        tk.Button(tab, text="Shift Signal", command=self.shift_signal).pack(pady=10)
        tk.Button(tab, text="Reverse Signal", command=self.reverse_signal).pack(pady=10)

    def create_visualize_signal_tab(self, tab):
        # Visualization tab
        tk.Button(tab, text="Visualize Signal", command=self.visualize_signal).pack(pady=10)

    def load_signal(self):
        # Load signal from a .txt file
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                # Reading the text file
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                start_index = int(lines[1])  # The second line is the start index
                N = int(lines[2])  # The third line is the number of samples
                signal_data = [list(map(int, line.split())) for line in lines[3:3 + N]]

                indices = [item[0] for item in signal_data]
                signal = [item[1] for item in signal_data]

                self.signals.append((indices, signal))
                messagebox.showinfo("Success", f"Loaded signal with {N} samples.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load signal: {e}")

    def visualize_signal(self):
        if not self.signals:
            messagebox.showerror("Error", "No signal loaded!")
            return
        plt.figure()
        for idx, (indices, signal) in enumerate(self.signals):
            plt.plot(indices, signal, label=f"Signal {idx + 1}")
        plt.title("Signal Visualization")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

    def add_signals(self):
        if len(self.signals) < 2:
            messagebox.showerror("Error", "At least two signals are required for addition!")
            return
        # Create a new signal array for the result
        result_signal = {}
        for idx, (indices, signal) in enumerate(self.signals):
            for i, val in zip(indices, signal):
                if i in result_signal:
                    result_signal[i] += val
                else:
                    result_signal[i] = val

        # Convert result_signal back to lists for saving and display
        sorted_result = sorted(result_signal.items())
        indices, values = zip(*sorted_result) if sorted_result else ([], [])

        self.save_result("addition", indices, values)
        messagebox.showinfo("Success", "Signals added successfully!")

    def multiply_signal(self):
        if not self.signals:
            messagebox.showerror("Error", "No signal loaded!")
            return

        try:
            constant = float(simpledialog.askstring("Input", "Enter constant to multiply:"))
            last_indices, last_signal = self.signals[-1]

            result_signal = [val * constant for val in last_signal]

            self.save_result("multiplication", last_indices, result_signal)
            messagebox.showinfo("Success", f"Signal multiplied by {constant} successfully!")
        except ValueError:
            messagebox.showerror("Error", "Invalid constant value!")

    def subtract_signals(self):
        if len(self.signals) < 2:
            messagebox.showerror("Error", "At least two signals are required for subtraction!")
            return

        # Create a new signal array for the result
        result_signal = {}
        first_indices, first_signal = self.signals[0]
        second_indices, second_signal = self.signals[1]

        for i, val in zip(first_indices, first_signal):
            result_signal[i] = val

        for i, val in zip(second_indices, second_signal):
            if i in result_signal:
                result_signal[i] -= val
            else:
                result_signal[i] = -val

        # Convert result_signal back to lists for saving and display
        sorted_result = sorted(result_signal.items())
        indices, values = zip(*sorted_result) if sorted_result else ([], [])

        self.save_result("subtraction", indices, values)
        messagebox.showinfo("Success", "Signals subtracted successfully!")

    def shift_signal(self):
        if not self.signals:
            messagebox.showerror("Error", "No signal loaded!")
            return
        try:
            k = int(simpledialog.askstring("Input", "Enter k (shift amount):"))
            last_indices, last_signal = self.signals[-1]

            # Create the shifted signal
            shifted_indices = [index + k for index in last_indices]
            self.save_result("shift", shifted_indices, last_signal)
            messagebox.showinfo("Success", f"Signal shifted by {k} steps.")
        except ValueError:
            messagebox.showerror("Error", "Invalid shift value!")

    def reverse_signal(self):
        if not self.signals:
            messagebox.showerror("Error", "No signal loaded!")
            return
        # Reverse signal (fold)
        last_indices, last_signal = self.signals[-1]
        reversed_signal = last_signal[::-1]
        reversed_indices = [-i for i in last_indices]

        self.save_result("reverse", reversed_indices, reversed_signal)
        messagebox.showinfo("Success", "Signal reversed successfully!")

    def save_result(self, operation, indices, signal):

        # Save the result to a file
        prefix = 0
        result_file_path = os.path.join(self.results_dir, f"{operation}-result.txt")
        with open(result_file_path, 'w') as f:
            f.write(f"{prefix}\n")
            f.write(f"{prefix}\n")
            f.write(f"{len(signal)}\n")  # Write the number of samples
            for index, value in zip(indices, signal):
                f.write(f"{int(index)} {int(value)}\n")  # Write index and value
        messagebox.showinfo("Success", f"Result saved to {result_file_path}")


# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = SignalProcessorApp(root)
    root.mainloop()