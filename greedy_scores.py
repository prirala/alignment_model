# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 23:26:53 2016

@author: iralp
"""
#Calculates the precision,recall,fscores for the alignments found

import pandas as pd
from copy import deepcopy
import sklearn.metrics as mtrc

#read the csv into a dataframe
into_frame = pd.read_csv('C:/Personal_Docs/Course_Work/NLP/Simple_Standard_Wiki_ALignment/Data/Data_format/Greedy_align/greedy_results_per_file.csv',encoding = "utf8")

#group by article to obtain the count
#print(into_frame.groupby(['article']).size())

#get the annotations into a list
y_predicted = into_frame['annotation']

y_predicted_good = deepcopy(y_predicted)
y_predicted_good_partial = deepcopy(y_predicted)

#replace all 2s,1s with 0s
y_predicted_good = [0 if a == 2 else a for a in y_predicted_good]
y_predicted_good = [0 if a == 1 else a for a in y_predicted_good]
y_predicted_good = [1 if a == 3 else a for a in y_predicted_good]

print('y_predicted_good:',y_predicted_good)

len_ = len(y_predicted)

#create another list with default '3'-GOOD annotations of same size

y_output = [1]*len_

#now calculate the precision,recall and fscore
print('GOOD : ',mtrc.precision_recall_fscore_support(y_predicted_good,y_output,average='binary'))
#print('GOOD_PARTIAL : ',mtrc.precision_recall_fscore_support(y_predicted_good_partial,y_output,average='micro'))
