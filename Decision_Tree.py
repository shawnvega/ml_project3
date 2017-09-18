import csv
from sklearn import tree
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    read_and_split_data('iris.data')
    print("----------- training set ------------")
    print_2d_matrix(training_set_x, training_set_y)
    print("----------- training ------------")
    accuracy_list = list()
    for max_tree_depth in range(1,13):
        clf = tree.DecisionTreeClassifier(max_depth=max_tree_depth)
        clf = clf.fit(training_set_x, training_set_y)
        print("----------- testing ------------")
        print('max depth =', max_tree_depth)
        predictions = clf.predict(testing_set_x)
        print(predictions)
        accuracy = accuracy_score(testing_set_y, predictions)
        print("accuracy = {:.1%}".format(accuracy))
        accuracy_list.append(accuracy)
    plt.plot([x for x in range(1, len(accuracy_list) + 1)], accuracy_list)
    plt.ylabel('accuracy')
    plt.xlabel('max_depth')
    plt.grid(True)
    plt.show()
    
#print(row_num, 'v: ', ', '.join(str(v) for v in row))