import csv

training_set = []
validation_set = []


with open('iris.data', newline='\n') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    row_num = 0
    for row in spamreader:
        row_num += 1
        if row_num % 2 == 0:
            print(row_num, 't: ', ', '.join(row))
            training_set.append(row)
        else:
            print(row_num, 'v: ', ', '.join(row))
            validation_set.append(row)
