from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
import csv
import numpy as np

dict={}

with open("./labeled/labeled_train.csv") as csvfile:
    reader=csv.reader(csvfile,delimiter=',')
    for row in reader:
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
        #count number of encounters
        for w in words:
            if w in dict:
                dict[w]+=1
            else:
                dict[w]=1

dictionary_file=open("./dict.txt","w")
wds=[[],[]]
for wd,cnt in dict.items():
    wds[0].append(wd)
    wds[1].append(cnt)

#pick out 'dict_size' number of most frequent words
dict_size=512
top_votes=np.argsort(wds[1])[-(dict_size):]
#write to dictionary file
for x in top_votes:
    dictionary_file.write(wds[0][x]+"\n")
dictionary_file.close()
