# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 23:52:46 2017

@author: Shawn
"""

import csv
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import numbers
import matplotlib
from matplotlib.ticker import FuncFormatter

type_map = dict()
type_num = 0
X = []
Y = []


def read_data(file_path):
    with open(file_path, newline='\n') as csvfile:
        filereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        all_data = list()
        for row in filereader:
            filter_row = list(filter(None, row))
            all_data.append(filter_row)
        return all_data


def array_to_int(arr):
    arr2 = list()
    for x in arr:
        arr2.append(int(x[0]))
    return arr2


def array_to_float(arr):
    arr2 = list()
    for x in arr:
        arr2.append(float(x))
    return arr2
    
    
def array_to_float_2d(arr):
    arr2 = list()
    for x in arr:
        arr2.append(array_to_float(x))
    return arr2

            
def print_2d_matrix(a_list, y):
    for count, row in enumerate(a_list):
        print(row, y[count], sep=',')
        
def show_not_int(array):
    for x in array:
        if not isinstance(x,int):
            print(x, 'is not int')
            
def show_not_float(array):
    for x in array:
        for y in x:
            if not isinstance(y,numbers.Real):
                print(y, 'is not float')
            
            
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
    x_train = read_data('uci_har_dataset/train/X_train.txt')
    y_train = read_data('uci_har_dataset/train/Y_train.txt')
    x_test = read_data('uci_har_dataset/test/X_test.txt')
    y_test = read_data('uci_har_dataset/test/Y_test.txt')
    #print(len(x_test[0]), len(x_test))
    #print_2d_matrix(x_train, y_train)
    print('converting x_train')
    x_train = array_to_float_2d(x_train)
    #show_not_float(x_train)
    print('converting x_train')
    y_train = array_to_int(y_train)
    #show_not_int(y_train)
    print('converting y_train')
    x_test = array_to_float_2d(x_test)
    #show_not_float(x_test)
    print('converting x_test')
    y_test = array_to_int(y_test)
    print('training')
    accuracy_list = list()
    layer_accuracy = list()
    highest = 0
    lowest = 1
    step_size = 200
    start = 100
    end = 4000
    lrate = 2.0
    for estimators in range(start, end,step_size):
        for randomx in range(0,1):
            start_time = time.time() * 1000
            clf = AdaBoostClassifier(n_estimators=estimators, learning_rate=lrate)
            print("--training--")
            clf.fit(x_train, y_train)
            print("--predicting--")
            predictions = clf.predict(x_test)
            print(predictions)
            accuracy = accuracy_score(y_test, predictions)
            if(accuracy > highest):
                highest = accuracy
                print('highest =', accuracy, 'estimators = ', estimators, 'random = ', randomx)
            if(accuracy < lowest):
                lowest = accuracy
                print('lowest =', accuracy, 'estimators = ', estimators, 'random = ', randomx)
            layer_accuracy.append(accuracy)
            stop_time = time.time() * 1000
            print('took', (int(round(stop_time - start_time))), 'milliseconds' )
        aver = average(layer_accuracy)
        print('average =', aver)
        accuracy_list.append(aver)
        layer_accuracy = list()
        print('estimators =', estimators)
#        
#        clf = tree.DecisionTreeClassifier(max_depth=max_tree_depth)
#        clf = clf.fit(x_train, y_train)
#        print("----------- testing ------------")
#        print('max depth =', max_tree_depth)
#        predictions = clf.predict(x_test)
#        stop_time = time.time() * 1000
#        print('took', (int(round(stop_time - start_time))), 'milliseconds' )
    print("average accuracy = {:.1%}".format(average(accuracy_list)))
    plt.plot([x for x in range(start, end , step_size)], accuracy_list)
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    xformatter = FuncFormatter(to_int)
    plt.gca().xaxis.set_major_formatter(xformatter)
    plt.ylabel('Average Accuracy')
    plt.xlabel('Number of Estimators')
    plt.title('Human Activity AdaBoostClassifier (learn rate = {:.1f})'.format(lrate))
    plt.grid(True)
    plt.show()