#!/usr/bin/env python3

results_path = r"D:\GItHub Reops\DSP-tasks\results\task5\Signals Convolution-result.txt"
#tests_path = "./tests/task3/quantization_result"
def ReadSignalFile(file_name):
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
                V1=str(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    return expected_indices,expected_samples

def DerivativeTest1(file_name, Your_EncodedValues,Your_QuantizedValues):
    expectedEncodedValues = []
    expectedQuantizedValues= []
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
                V2=str(L[0])
                V3=float(L[1])
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                line = f.readline()
            else:
                break
    if( (len(Your_EncodedValues)!=len(expectedEncodedValues)) or (len(Your_QuantizedValues)!=len(expectedQuantizedValues))):
        print("Convolution Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_EncodedValues)):
        if(Your_EncodedValues[i]!=expectedEncodedValues[i]):
            print("Convolution Test case failed, your indices have different indicies from the expected one") 
            return
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            print("Convolution Test case failed, your values have different values from the expected one") 
            return
    
    print("Convolution Test case passed successfully")


if __name__ == "__main__":

    Your_indices, Your_Values = ReadSignalFile(results_path)
    DerivativeTest1("./testcases/Convolution testcases/Conv_output.txt", Your_indices, Your_Values)