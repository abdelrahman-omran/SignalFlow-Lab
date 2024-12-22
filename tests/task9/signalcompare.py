#!/usr/bin/env python3
import os

PROJECT_ROOT_DIR = os.getcwd()

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

def Compare_Signals(file_name,Your_indices,Your_samples):      
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one") 
            return
    print("Test case passed successfully")

if __name__ == "__main__":

    print("|| Filter Test ||")
    result_path = PROJECT_ROOT_DIR + "/results/Task9/Signals Convolution-result.txt"
    result_indices, result_values = ReadSignalFile(result_path)

    output_path = PROJECT_ROOT_DIR + "/tests/task9/FIR test cases/Testcase 8/ecg_band_stop_filtered.txt"
    #output_indices, output_values = ReadSignalFile(output_path)
    Compare_Signals(output_path, result_indices, result_values)
