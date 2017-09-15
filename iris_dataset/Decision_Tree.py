import csv
from sklearn import tree

training_set = []
validation_set = []
type_map = dict()
type_num = 0
X = []
Y = []


def read_and_split_data():
    with open('iris.data', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        row_num = 0
        for row in spamreader:
            row_num += 1
            if not row[len(row)-1] in type_map:
                type_map[row[len(row)-1]] = type_num
                type_num += 1
            if row_num % 2 == 0:
                print(row_num, 't: ', ', '.join(row))
                training_set.append(row)
            else:
                print(row_num, 'v: ', ', '.join(row))
                validation_set.append(row)


def train():
    pass

if __name__ == '__main__':
    read_and_split_data()