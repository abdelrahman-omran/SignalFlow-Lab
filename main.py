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
        self.N = 0
        # Create results directory if it doesn't exist
        self.results_dir = "./results/task5"
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

        # Tab 3: Task 3 for signal quantization
        task3_tab = ttk.Frame(self.notebook)
        self.notebook.add(task3_tab, text="Task 3")
        self.create_task3_tab(task3_tab)

        # Tab 4: Task 4 for signal quantization
        task4_tab = ttk.Frame(self.notebook)
        self.notebook.add(task4_tab, text="Task 4")
        self.create_task3_tab(task4_tab)

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

    def create_task3_tab(self, tab):
        """Task 3 Tab Layout: Allows user to quantize a signal."""
        button_frame = ttk.Frame(tab)
        button_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Quantize Signal Button
        tk.Button(button_frame, text="Quantize Signal", command=self.quantize_signal).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(button_frame, text="display_quantization_results", command=self.display_quantization_results).grid(row=0, column=1, padx=5, pady=5)

    def create_task3_tab(self, tab):
        """Task 3 Tab Layout: Allows user to quantize a signal."""
        button_frame = ttk.Frame(tab)
        button_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Quantize Signal Button
        tk.Button(button_frame, text="", command=self.quantize_signal).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(button_frame, text="Get Signal Derivative", command=self.signal_derivative).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(button_frame, text="Convolute two signals", command=self.convolve).grid(row=0, column=2, padx=5, pady=5)



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
                self.N = int(lines[2])
                signal_data = [list(map(float, line.split())) for line in lines[3:3 + self.N]]

                indices = [item[0] for item in signal_data]
                signal = [item[1] for item in signal_data]

                self.signals.append((indices, signal))
                messagebox.showinfo("Success", f"Loaded signal with {self.N} samples.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load signal: {e}")

    def visualize_signal_mode(self, mode="continuous"):
        """Visualize the signal in either continuous or discrete mode"""
        if not self.signals:
            messagebox.showerror("Error", "No signal loaded!")
            return
        
        plt.figure()
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # A list of colors to cycle through

        for idx, (indices, signal) in enumerate(self.signals):
            color = colors[idx % len(colors)]  # Cycle through the color list
            if mode == "discrete":
                plt.stem(indices, signal, label=f"Signal {idx + 1}", linefmt=color, markerfmt=color+'o', basefmt=" ")
            else:
                plt.plot(indices, signal, label=f"Signal {idx + 1}")

        plt.title(f"Signal Visualization - {mode.capitalize()} Mode")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()

    def display_quantization_results(self):
        self.load_signal()
        self.load_signal()
        plt.figure()
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # A list of colors to cycle through
        indices = np.arange(0,len(self.signals[-1][0]))
        for idx, (index, signal) in enumerate(self.signals):
            color = colors[idx % len(colors)]  # Cycle through the color list
            plt.plot(indices, signal, label=f"Signal {idx + 1}")
        self.load_signal()
                # Define the x-axis values (indices)
        x_values = indices  # Assuming sequential indices, e.g., 0, 1, 2, ...
        y_values = self.signals[-1][1]
        # Plotting
        plt.step(x_values, y_values, where='post', label='Encoded Signal', color='b', linewidth=2)
        plt.scatter(x_values, y_values, color='red')  # Mark each value for clarity

        # Labels and title
        plt.xlabel('Index')
        plt.ylabel('Encoded Value')
        plt.title('Digital Signal Visualization')

        # Display encoded binary values at each point
        for i, txt in enumerate(self.signals[-1][0]):
            plt.text(x_values[i], y_values[i] + 0.02, txt, ha='center', fontsize=10)

        plt.grid(True)
        plt.legend()
        plt.show()
        messagebox.showinfo("Quantization Results")

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
        for idx, (indices, signal) in enumerate(self.signals):
            for i, val in zip(indices, signal):
                if idx == 0:
                    result_signal[i] = val
                else:
                    result_signal[i] = result_signal.get(i, 0) - val

        sorted_result = sorted(result_signal.items())
        indices, values = zip(*sorted_result) if sorted_result else ([], [])
        self.save_result("subtract", indices, values)
        messagebox.showinfo("Success", "Signals subtracted successfully!")

    def shift_signal(self):
        if not self.signals:
            messagebox.showerror("Error", "No signal loaded!")
            return
        try:
            shift_amount = int(simpledialog.askstring("Input", "Enter shift amount:"))
            last_indices, last_signal = self.signals[-1]
            shifted_indices = [i + shift_amount for i in last_indices]
            self.save_result("shift", shifted_indices, last_signal)
            messagebox.showinfo("Success", f"Signal shifted by {shift_amount} samples successfully!")
        except ValueError:
            messagebox.showerror("Error", "Invalid shift amount!")

    def reverse_signal(self):
        if not self.signals:
            messagebox.showerror("Error", "No signal loaded!")
            return
        last_indices, last_signal = self.signals[-1]
        reversed_signal = last_signal[::-1]
        self.save_result("reverse", last_indices, reversed_signal)
        messagebox.showinfo("Success", "Signal reversed successfully!")

    def quantize_signal(self):
        # Convert last_signal to a NumPy array if it isn't already
        if not self.signals:
            messagebox.showerror("Error", "No signal loaded!")
            return

        method = simpledialog.askstring("Input", "Enter 'bits' for bits input or 'levels' for levels input:")
        if method not in ['bits', 'levels']:
            messagebox.showerror("Error", "Invalid input method. Please enter 'bits' or 'levels'.")
            return

        last_indices, last_signal = self.signals[-1]
        indices = last_indices
        max_value = np.max(last_signal)
        min_value = np.min(last_signal)

        # Calculate number of bits or levels
        if method == 'bits':
            num_bits = int(simpledialog.askstring("Input", "Enter number of bits:"))
            num_levels = 2 ** num_bits
        else:  # 'levels'
            num_levels = int(simpledialog.askstring("Input", "Enter number of levels:"))
            num_bits = int(np.ceil(np.log2(num_levels)))

        # Calculate step size
        step_size = (max_value - min_value) / num_levels

        # Create intervals based on step_size
        intervals = np.arange(min_value, max_value + step_size, step_size)
        # Format intervals to two decimal places
        formatted_intervals = [f"{value:.2f}" for value in intervals]
        print("Intervals:", formatted_intervals)

        # Calculate midpoints of each interval
        midpoints = (intervals[:-1] + intervals[1:]) / 2
        # Format midpoints to two decimal places
        formatted_midpoints = [f"{value:.2f}" for value in midpoints]
        print("Midpoints:", formatted_midpoints)

        # Quantize the signal
        quantized_signal = np.zeros_like(last_signal)

        # Encode interval indices in binary and associate with midpoints
        encoded_indices = [f"{i:0{num_bits}b}" for i in range(len(midpoints))]

        # Initialize a list to store errors
        errors = np.zeros_like(last_signal)
        sig_errors = np.zeros_like(last_signal)
        # Iterate through each signal value
        for i, value in enumerate(last_signal):
            # Find the index of the interval that the value falls into
            index = np.searchsorted(intervals, value, side='right') - 1
            # Map to the midpoint
            if 0 <= index < len(midpoints):  # Ensure the index is valid
                quantized_signal[i] = midpoints[index]
                last_indices[i] = encoded_indices[index]
                # Calculate the error at the current index
                sig_errors[i] = abs(value - quantized_signal[i])
                errors[i] = abs(value - quantized_signal[i])**2

        # Calculate mean error
        error = np.mean(errors)

        # Format quantized_signal to two decimal places
        quantized_signal = [f"{value:.2f}" for value in quantized_signal]

        # Save results to a file
        output_path = "./results/task3/quantization_errors.txt"  # Change this path as needed
        with open(output_path, "w") as f:
            f.write(f"Mean Error: {error:.2f}\n")

        self.save_result("quantized", last_indices, quantized_signal)
        self.save_result("quantized_error", indices, sig_errors)

        messagebox.showinfo("Success", f"Signal quantized successfully! Errors saved to {output_path}")

    def signal_derivative(self):

        self.load_signal()

        first_derivative = []
        last_indices, last_signal = self.signals[-1]
        dev1_indices = []
        dev2_indices = []

        for i in range(1, self.N):
            first_derivative.append(last_signal[i] - last_signal[i - 1])
            dev1_indices.append(i-1)

        # Compute second derivative: Y(n) = x(n+1) - 2x(n) + x(n-1)
        second_derivative = []  # Initialize with zero for n = 0
        for i in range(1, self.N - 1):
            second_derivative.append(last_signal[i + 1] - 2 * last_signal[i] + last_signal[i - 1])
            dev2_indices.append(i-1)

            #second_derivative.append(0)  # Append zero for n = N-1 (boundary condition)

        self.save_result("first_derivative", dev1_indices, first_derivative)
        self.save_result("second_derivative", dev2_indices, second_derivative)

    def convolve(self):
    # Length of the resulting signal will be len(x) + len(h) - 1
        self.load_signal()
        last_indices, x = self.signals[-1]
        idx1 = last_indices[0]

        self.load_signal()
        last_indices, h = self.signals[-1]
        idx2 = last_indices[0]
        
        start_idx = min(idx1, idx2)
        y = [0] * (len(x) + len(h) - 1)
        y_indices  = []
    # Perform the convolution operation manually
        for n in range(len(y)):
            y_indices.append(start_idx)
            start_idx = start_idx + 1
            for k in range(len(h)):
                if n - k >= 0 and n - k < len(x):
                    y[n] += x[n - k] * h[k]
                    

        self.save_result("Signals Convolution",y_indices, y)

    def clear_signals(self):
        """Clear all loaded signals."""
        self.signals = []
        messagebox.showinfo("Success", "All signals cleared!")

    def save_result(self, operation, indices, values):
            result_file = os.path.join(self.results_dir, f"{operation}-result.txt")
            with open(result_file, 'w') as f:
                f.write("0\n")
                f.write("0\n")
                f.write(f"{len(indices)}\n")
                for idx, val in zip(indices, values):
                    f.write(f"{int(idx)} {int(val)}\n")
            print(f"Saved result to {result_file}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    root = tk.Tk()
    app = SignalProcessorApp(root)
    root.mainloop()
