# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 17:32:28 2016

@author: iralp
"""

from pycorenlp import StanfordCoreNLP
#import pprint
import json

nlp = StanfordCoreNLP("http://localhost:9000")

with open(r'C:/Personal_Docs/Course_Work/NLP/Simple_Standard_Wiki_ALignment/Data/Formatted_data/A.C.F._Fiorentina_simple') as fd:
    lines = fd.readlines()
    
#lines = "Pranathi is good.She is the best"
output = nlp.annotate("".join(lines), properties={
  'annotators': 'mention,coref',
  'outputFormat': 'json',
  'timeout':'10000000'
  })
  
#pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(output['corefs'])
with open('C:/Personal_Docs/Course_Work/NLP/Simple_Standard_Wiki_ALignment/Data/Formatted_data/A.C.F._Fiorentina_simple_corefs_3', 'w') as outfile:
    json.dumps(output,outfile)