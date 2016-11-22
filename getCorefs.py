# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 00:46:57 2016

@author: iralp
"""

import json
from copy import deepcopy
#get the list of dependencies

print("executing")
#open json file
json_data = open('C:/Personal_Docs/Course_Work/NLP/Simple_Standard_Wiki_ALignment/Data/Formatted_data/A.C.F._Fiorentina_simple_corefs').read()

new_string = str.replace(json_data,"\\","")

new_json = json.loads(new_string)
jsonData = new_json['corefs']


#new_string = str.replace(jsonData,"'","\"")
sent_dict = {}
count = 5
#dicJson = json.loads(new_string)

print("I am here")

#now map the individual sentences to corresponding sentences in the article
with open('C:/Personal_Docs/Course_Work/NLP/Simple_Standard_Wiki_ALignment/Data/Formatted_data/A.C.F._Fiorentina_simple') as fd:
	lines = fd.readlines()

mapping = {}
indx = 0
while indx < (len(lines)):
    line = lines[indx]
    nos = line.split(".")
    print('indx : ',indx)
    print("line : ",line," nos : ",len(nos))
    print(nos)
    if len(nos) == 2:
        mapping[indx + 1] = indx + 1
        indx = indx + 1
    else:
		#map the next sentences to this indx
      lenOfSens = len(nos)
      while lenOfSens > 1:
          print('indx + lenOfSens - 1',indx + lenOfSens - 1 ,'indx + 1',indx + 1)
          mapping[indx + lenOfSens - 1] = indx + 1
          lenOfSens = lenOfSens - 1
      indx = indx + len(nos) - 1
print('final mapping ',mapping)


for each in jsonData.keys():
    value = jsonData[each]
    len_of_list = len(value)
    lst = []
    print('each : ',each)
    for each_entry in value:
        sentNum = each_entry['sentNum']
        lst.append(mapping[sentNum])       
        
    for each_val in lst:
        print('lst : ',lst)
        if each_val not in sent_dict.keys():
            sent_dict[each_val] = deepcopy(lst)
            #print('sent_dict : ',sent_dict)
        else:
            prev_lst = sent_dict[each_val]
            print('prev_lst : ',prev_lst)
            prev_lst.extend(lst)
            sent_dict[each_val] = prev_lst
            #print('sent_dict',sent_dict)
    print(value)
    '''count = count - 1
    if count == 0:
        break'''
    
for each_key in sent_dict.keys():
    values = sent_dict[each_key]
    set_of_values = set(values)
    if each_key in set_of_values:
        set_of_values.remove(each_key)
    sent_dict[each_key] = list(set_of_values)
    
    
print("length of dictionary : ",len(sent_dict))
#print(sent_dict)

with open('C:/Personal_Docs/Course_Work/NLP/Simple_Standard_Wiki_ALignment/Data/Formatted_data/A.C.F._Fiorentina_simple_coref_json','w+') as outfile:
    json.dump(sent_dict,outfile)
    