import os
import tkinter as tk
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
        self.results_dir = "./results/task1"
        os.makedirs(self.results_dir, exist_ok=True)

        # Create tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(sticky="nsew")

        # Create the different tabs for tasks
        self.create_tabs()

    def create_tabs(self):
        # Tab 1: Task 1 with all operations
        task1_tab = ttk.Frame(self.notebook)
        self.notebook.add(task1_tab, text="Task 1")
        self.create_task1_tab(task1_tab)

        # Tab 2: Task 2 for signal visualization (continuous/discrete)
        task2_tab = ttk.Frame(self.notebook)
        self.notebook.add(task2_tab, text="Task 2")
        self.create_task2_tab(task2_tab)

    def create_task1_tab(self, tab):
        # Task 1 Tab Layout: Buttons for signal operations and visualization
        button_frame = ttk.Frame(tab)
        button_frame.pack(padx=10, pady=10, fill="both", expand=True)

        tk.Button(button_frame, text="Load Signal", command=self.load_signal).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(button_frame, text="Add Signals", command=self.add_signals).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(button_frame, text="Multiply Signal by Constant", command=self.multiply_signal).grid(row=0, column=2, padx=5, pady=5)
        tk.Button(button_frame, text="Subtract Signals", command=self.subtract_signals).grid(row=1, column=0, padx=5, pady=5)
        tk.Button(button_frame, text="Shift Signal", command=self.shift_signal).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(button_frame, text="Reverse Signal", command=self.reverse_signal).grid(row=1, column=2, padx=5, pady=5)
        tk.Button(button_frame, text="Visualize Signal", command=self.visualize_signal).grid(row=2, column=0, columnspan=3, padx=5, pady=5)
        tk.Button(button_frame, text="Clear Signals", command=self.clear_signals).grid(row=3, column=0, columnspan=3, padx=5, pady=5)

    def create_task2_tab(self, tab):
        """Task 2 Tab Layout: Allows user to display signals in continuous or discrete representation"""
        button_frame = ttk.Frame(tab)
        button_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Buttons for continuous and discrete visualization
        tk.Button(button_frame, text="Visualize as Continuous", command=lambda: self.visualize_signal_mode("continuous")).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(button_frame, text="Visualize as Discrete", command=lambda: self.visualize_signal_mode("discrete")).grid(row=0, column=1, padx=5, pady=5)

    def load_signal(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                start_index = int(lines[1])
                N = int(lines[2])
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
        result_signal = {i: v for i, v in zip(first_indices, first_signal)}
        for i, v in zip(second_indices, second_signal):
            result_signal[i] = result_signal.get(i, 0) - v
        sorted_result = sorted(result_signal.items())
        indices, values = zip(*sorted_result) if sorted_result else ([], [])
        self.save_result("sub", indices, values)
        messagebox.showinfo("Success", "Signals subtracted successfully!")

    def shift_signal(self):
        if not self.signals:
            messagebox.showerror("Error", "No signal loaded!")
            return
        try:
            k = int(simpledialog.askstring("Input", "Enter k (shift amount):"))
            last_indices, last_signal = self.signals[-1]
            shifted_indices = [index - k for index in last_indices]
            self.save_result("shift", shifted_indices, last_signal)
            messagebox.showinfo("Success", f"Signal shifted by {k} steps.")
        except ValueError:
            messagebox.showerror("Error", "Invalid shift value!")

    def reverse_signal(self):
        if not self.signals:
            messagebox.showerror("Error", "No signal loaded!")
            return
        last_indices, last_signal = self.signals[-1]
    
        # Find the minimum and maximum indices
        min_index = min(last_indices)
        max_index = max(last_indices)    
        # Reverse both indices and signal values
        reversed_indices = [-(max_index - (i - min_index)) for i in last_indices]
        reversed_signal = last_signal[::-1]

        self.save_result("rev", reversed_indices, reversed_signal)
        messagebox.showinfo("Success", "Signal reversed successfully!")
    
    def clear_signals(self):
        """Clears all loaded signals from memory."""
        self.signals.clear()
        messagebox.showinfo("Success", "All signals cleared!")

    def save_result(self, operation, indices, signal):
        result_file_path = os.path.join(self.results_dir, f"{operation}_result.txt")
        with open(result_file_path, 'w') as file:
            file.write(f"{operation.capitalize()} Result\n")
            file.write("Index Amplitude\n")
            for i, amp in zip(indices, signal):
                file.write(f"{i} {amp}\n")
        messagebox.showinfo("Success", f"Result saved to {result_file_path}!")

if __name__ == "__main__":
    root = tk.Tk()
    app = SignalProcessorApp(root)
    root.mainloop()
