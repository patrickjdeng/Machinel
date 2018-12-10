import sys

def read_file(filename):
    infile = open(filename)
    array = []
    for line in infile:
        array.append(line.strip('\n'))
    return array

def main():
    predictions_array = read_file(sys.argv[1])
    actual_array = read_file(sys.argv[2])
    correct = 0
    total = len(predictions_array)
    for i in range(len(predictions_array)):
        if predictions_array[i] == actual_array[i]:
            correct += 1
    percentage = float(correct)/float(total)*100
    print("Total Accuracy is: " + str(percentage))
    
main()