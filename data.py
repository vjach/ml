import csv
import numpy as np

def load(filename):
    X = []
    Y = []
    
    with open(filename, 'rb') as data_file:
        data_reader = csv.reader(data_file, delimiter=',')
        for row in data_reader:
            X.append([ float(val) for val in row[:-1] ])
            Y.append(int(row[-1]))

    return np.array(X), np.array(Y)
