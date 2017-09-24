# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 20:24:46 2017

@author: Shawn
"""

import csv
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib.ticker import FuncFormatter
from sklearn import svm

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
    start = 0
    end = 1
    step_size = 1
    kernel_list = ['linear','rbf','poly','sigmoid']
    for estimator in kernel_list:
        for randomx in range(0, 1):
            clf = svm.SVC(kernel='linear', cache_size=1000, random_state=randomx)
            clf.fit(training_set_x, training_set_y)
            predictions = clf.predict(testing_set_x)
            print(predictions)
            accuracy = accuracy_score(testing_set_y, predictions)
            print(randomx, "accuracy = {:.1%}".format(accuracy))
            layer_accuracy.append(accuracy)
        accuracy_list.append(average(layer_accuracy))
        layer_accuracy = list()
        print('trial =', trial)
    print('average accuracy =', average(accuracy_list))
    y_pos = np.arange(len(kernel_list))
    plt.bar(y_pos, accuracy_list, align='center', alpha=0.5)
    #plt.bar([x for x in range(start, end , step_size)], accuracy_list)
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.xticks(y_pos, kernel_list)
    plt.ylabel('average accuracy')
    #plt.xlabel('random state')
    plt.xlabel('trial#')
    plt.title('Iris linear svm')
    plt.grid(True)
    plt.show()
    
#print(row_num, 'v: ', ', '.join(str(v) for v in row))