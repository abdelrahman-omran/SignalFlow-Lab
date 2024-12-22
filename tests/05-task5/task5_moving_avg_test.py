#!/usr/bin/env python3

results_path = "./results/task5/moving_average-result.txt"
test_signal_path = "./tests/task5/testcases/Moving Average testcases/MovingAvg_out1.txt"
def ReadSignalFile(file_name):
    """Reads the signal file and extracts indices and values."""
    indices = []
    values = []

    with open(file_name, 'r') as f:
        # Skip the first three lines (header info)
        f.readline()
        f.readline()
        f.readline()
        line = f.readline()

        while line:
            L = line.strip()
            if len(L.split(' ')) == 2:
                parts = L.split(' ')
                index = int(parts[0])
                value = float(parts[1])
                indices.append(index)
                values.append(value)
                line = f.readline()
            else:
                break

    return indices, values

def MovingAverageTest(file_name, your_indices, your_values):
    """Compares the computed moving average result with the expected result."""
    expected_indices = []
    expected_values = []

    with open(file_name, 'r') as f:
        # Skip the first three lines (header info)
        f.readline()
        f.readline()
        f.readline()
        line = f.readline()

        while line:
            L = line.strip()
            if len(L.split(' ')) == 2:
                parts = L.split(' ')
                index = int(parts[0])
                value = float(parts[1])
                expected_indices.append(index)
                expected_values.append(value)
                line = f.readline()
            else:
                break

    if len(your_indices) != len(expected_indices) or len(your_values) != len(expected_values):
        print("Moving Average Test failed: Signal length mismatch.")
        return

    for i in range(len(your_indices)):
        if your_indices[i] != expected_indices[i]:
            print(f"Moving Average Test failed: Index mismatch at position {i}.")
            return

    for i in range(len(expected_values)):
        if abs(your_values[i] - expected_values[i]) >= 0.01:  # Allow small floating-point tolerance
            print(f"Moving Average Test failed: Value mismatch at position {i}.")
            return

    print("Moving Average Test passed successfully.")

if __name__ == "__main__":
    # Read the generated result
    your_indices, your_values = ReadSignalFile(results_path)

    # Compare with the expected result
    MovingAverageTest(test_signal_path, your_indices, your_values)
