#!/usr/bin/env python3

import math

#Use to test the Amplitude of DFT and IDFT
def SignalComapreAmplitude(SignalInput = [] ,SignalOutput= []):
    if len(SignalInput) != len(SignalOutput):
        return False
    else:
        for i in range(len(SignalInput)):
            if abs(SignalInput[i]-SignalOutput[i])>0.001:
                return False
            elif SignalInput[i]!=SignalOutput[i]:
                return False
        return True

def RoundPhaseShift(P):
    while P<0:
        p+=2*math.pi
    return float(P%(2*math.pi))

#Use to test the PhaseShift of DFT
def SignalComaprePhaseShift(SignalInput = [] ,SignalOutput= []):
    if len(SignalInput) != len(SignalOutput):
        return False
    else:
        for i in range(len(SignalInput)):
            A=round(SignalInput[i])
            B=round(SignalOutput[i])
            if abs(A-B)>0.0001:
                return False
            elif A!=B:
                return False
        return True


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
                index = float(parts[0])
                value = float(parts[1])
                indices.append(index)
                values.append(value)
                line = f.readline()
            else:
                break

    return indices, values

if __name__ == "__main__":
    # Read the generated result
    result_path = r"D:\GItHub Reops\DSP-tasks\results\task7\DFT-result.txt"
    result_amp, result_phase = ReadSignalFile(result_path)

    output_path = r"D:\GItHub Reops\DSP-tasks\tests\task7\DFT\Outout_Signal_DFT.txt"
    output_amp, output_phase = ReadSignalFile(output_path)

    amp_result = SignalComapreAmplitude(result_amp, output_amp)
    print("Amplitude comparing is done")
    if(amp_result):
        print("result: Succeeded \n")
    else:
        print("result: Failed \n")

    phase_result = SignalComaprePhaseShift(result_phase, output_phase)
    print("Phase Shift comparing is done")
    if(phase_result):
        print("result: Succeeded \n")
    else:
        print("result: Failed \n")
