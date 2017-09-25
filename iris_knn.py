# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 13:54:35 2017

@author: Shawn
"""

import csv
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib.ticker import FuncFormatter
from sklearn.neighbors import KNeighborsClassifier

training_set_x = []
training_set_y = []
testing_set_x = []
testing_set_y = []
type_map = dict()
type_num = 0
X = []
Y = []


def read_and_split_data(file_path):
    global type_num
    with open(file_path, newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        row_num = 0
        for row in spamreader:
            row_num += 1
            if not row[len(row)-1] in type_map:
                type_map[row[len(row)-1]] = type_num
                type_num += 1
            y = type_map[row[len(row)-1]]
            row_x = row[0:len(row)-1] 
            row_x = [float(i) for i in row_x]
            row_mod = row_num % 5
            if row_mod != 0:
                training_set_x.append(row_x)
                training_set_y.append(y)
            else:
                testing_set_x.append(row_x)
                testing_set_y.append(y)
                
                
def print_2d_matrix(a_list, y):
    for count, row in enumerate(a_list):
        print(row, y[count], sep=',')


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'
    
    
def to_int(x, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = int(x)
    return s

def average(alist):
    asum = sum(alist)
    return asum/len(alist)
    

if __name__ == '__main__':
    read_and_split_data('iris_dataset/iris.data')
    print("----------- training set ------------")
    print_2d_matrix(training_set_x, training_set_y)
    print("----------- training ------------")
    accuracy_list = list()
    layer_accuracy = list()
    start = 1
    end = 17
    step_size = 1
    for neighbors in range(start, end, step_size):
        for randomx in range(0, 1):
            clf = KNeighborsClassifier(n_neighbors=neighbors)
            clf.fit(training_set_x, training_set_y)
            predictions = clf.predict(testing_set_x)
            print(predictions)
            accuracy = accuracy_score(testing_set_y, predictions)
            print(randomx, "accuracy = {:.1%}".format(accuracy))
            layer_accuracy.append(accuracy)
        accuracy_list.append(average(layer_accuracy))
        layer_accuracy = list()
    print('average accuracy =', average(accuracy_list))
    plt.plot([x for x in range(start, end, step_size)], accuracy_list)
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    xformatter = FuncFormatter(to_int)
    plt.gca().xaxis.set_major_formatter(xformatter)
    #vals = plt.get_yticks()
    #vals.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])
    plt.ylabel('average accuracy')
    #plt.xlabel('random state')
    plt.xlabel('Neighbors in Classifier')
    plt.title('Iris KNN')
    plt.grid(True)
    plt.show()
    
#print(row_num, 'v: ', ', '.join(str(v) for v in row))