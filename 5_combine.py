import csv
from numpy import genfromtxt
import numpy as np
def combiner(label_type):

    p_file_y=genfromtxt('./picked/label_'+str(label_type)+'_'+str(1)+'.csv', delimiter=',')
    p_file_n=genfromtxt('./picked/label_'+str(label_type)+'_'+str(0)+'.csv', delimiter=',')
    original_file=genfromtxt('./processed/labeled_train.csv', delimiter=',')
    #p_file_y=open('./picked/label_'+str(label_type)+'_'+str(1)+'.csv','r')
    #p_file_n=open('./picked/label_'+str(label_type)+'_'+str(0)+'.csv','r')
    #original_file=open('./processed/labeled_train.csv','r')
    combined_file=open('./picked/combined_'+str(label_type)+'.csv','w')

    indices=list(range(0, 512))
    indices.append(512+label_type)
    original_file=original_file[:,indices]

    wr = csv.writer(combined_file, dialect='excel')
    for r in p_file_y:
        wr.writerow(np.append(r,1))
    for r in p_file_n:
        wr.writerow(np.append(r,0))
    for r in original_file:
        wr.writerow(r)
    combined_file.close()
combiner(1)#social
#combiner(0)#agency
