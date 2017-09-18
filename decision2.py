# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 16:14:54 2017

@author: Shawn
"""

import csv
from sklearn import tree
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

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
                
def print_2d_matrix(a_list, y):
    for count, row in enumerate(a_list):
        print(row, y[count], sep=',')


if __name__ == '__main__':
    x_train = read_data('../uci_har_dataset/train/X_train.txt')
    y_train = read_data('../uci_har_dataset/train/Y_train.txt')
    x_test = read_data('../uci_har_dataset/test/X_test.txt')
    y_test = read_data('../uci_har_dataset/test/Y_test.txt')
    print(len(x_test[0]), len(x_test))
    accuracy_list = list()
    for max_tree_depth in range(1,19):
        start_time = time.time() * 1000
        clf = tree.DecisionTreeClassifier(max_depth=max_tree_depth)
        clf = clf.fit(x_train, y_train)
        print("----------- testing ------------")
        print('max depth =', max_tree_depth)
        predictions = clf.predict(x_test)
        stop_time = time.time() * 1000
        print('took', (int(round(stop_time - start_time))), 'milliseconds' )
        accuracy = accuracy_score(y_test, predictions)
        print("accuracy = {:.1%}".format(accuracy))
        accuracy_list.append(accuracy)
    plt.plot([x for x in range(1, len(accuracy_list) + 1)], accuracy_list)
    plt.ylabel('accuracy')
    plt.xlabel('max_depth')
    plt.title('Human Activity')
    plt.grid(True)
    plt.show()
    
#print(row_num, 'v: ', ', '.join(str(v) for v in row))