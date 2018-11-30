import numpy as np
import matplotlib.pyplot as plt

def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    number_lines = len(arrayOlines)
    returnMat = np.zeros((number_lines,3))
    classvector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listLine = line.split('\t')
        returnMat[index,:] = listLine[:3]
        if listLine[-1] == 'didntLike':
            classvector.append(1)
        if listLine[-1] == 'didntLike':
            classvector.append(1)
        if listLine[-1] == 'didntLike':
            classvector.append(1)
