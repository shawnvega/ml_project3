# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 21:12:05 2017

@author: Shawn
"""

import csv
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import numbers

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
    clf = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(5), random_state=1)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    print(predictions)
    accuracy = accuracy_score(y_test, predictions)
    print("accuracy = {:.1%}".format(accuracy))
#    for max_tree_depth in range(1,19):
#        start_time = time.time() * 1000
#        clf = tree.DecisionTreeClassifier(max_depth=max_tree_depth)
#        clf = clf.fit(x_train, y_train)
#        print("----------- testing ------------")
#        print('max depth =', max_tree_depth)
#        predictions = clf.predict(x_test)
#        stop_time = time.time() * 1000
#        print('took', (int(round(stop_time - start_time))), 'milliseconds' )
#        accuracy = accuracy_score(y_test, predictions)
#        print("accuracy = {:.1%}".format(accuracy))
#        accuracy_list.append(accuracy)
#    plt.plot([x for x in range(1, len(accuracy_list) + 1)], accuracy_list)
#    plt.ylabel('accuracy')
#    plt.xlabel('max_depth')
#    plt.title('Human Activity')
#    plt.grid(True)
#    plt.show()