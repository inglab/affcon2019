import os
import random

#read lines from labeled data file

filelist = os.listdir("./labeled")
for filename in filelist :
    print("processing"+filename)
    count=0
    with open("./labeled/"+filename) as data_file:
        lines = data_file.readlines()
#remove first line
lines=lines[1:]

#shuffle and split 9:1 to train,test
random.shuffle(lines)
train_len=(int)(0.9*len(lines))
train=lines[:train_len]
test=lines[train_len:]

#write to file
train_file=open('./labeled/labeled_train.csv', 'w')
test_file=open('./labeled/labeled_test.csv', 'w')

for line in train:
    train_file.write(line)

for line in test:
    test_file.write(line)

train_file.close()
test_file.close()
