from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
import csv
import numpy as np

#read from dictionary file
dictionary_file=open("./dict.txt",'r')
my_dict=dictionary_file.read().split('\n')[:-1]
dictionary_file.close()
word_count=len(my_dict)

data=[]

#change file name for different files
#labeled_train / labeled_test
data_file=open("./labeled/labeled_test.csv",'r')
processed_file=open("./processed/labeled_test.csv", 'w')

#one hot encoding for 'moment' using dictionary
#512 one hot + agency label (0,1) + social label(0,1)
reader=csv.reader(data_file,delimiter=',')
for row in reader:
    item=[]
    #1 moment one hot
    count1=[0]*word_count
    #print(row[1])
    tokens=word_tokenize(row[1])
    tokens=[w.lower() for w in tokens]
    #remove punctuation
    table=str.maketrans('','',string.punctuation)
    stripped=[w.translate(table) for w in tokens]
    #remove not alphabetic tokens
    words=[word for word in stripped if word.isalpha()]
    #filter out stop words
    stop_words=set(stopwords.words('english'))
    words=[w for w in words if not w in stop_words]
    #one hot encoding
    for word in words:
        #print(word)
        if word in my_dict:
            #print(word)
            count1[my_dict.index(word)]=1
    for i in range(0,word_count):
        item.append(str(count1[i]))

    #append label at the end
    if row[3]=='yes': #agency
        item.append(1)
    else:
        item.append(0)
    if row[4]=='yes': #social
        item.append(1)
    else:
        item.append(0)
    data.append(item)

#write to processed file
wr = csv.writer(processed_file, dialect='excel')
for x in data:
    wr.writerow(x)
processed_file.close()
