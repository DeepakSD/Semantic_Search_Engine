'''
Created on Oct 30, 2017

@author: deepaks
'''
import collections
import io
import os
from urllib2 import *

from nltk import tokenize
from nltk.tokenize import word_tokenize
import pysolr

import pandas as pd


class DataProcessing:
    def fileToDataFrame(self, path):  
        data = []
        for f in sorted(os.listdir(path)):
            with io.open(path + f, 'r', encoding='utf-8', errors='ignore') as dataFile:
                data.append(dataFile.read())
        for i in range(len(data)):
            sentences = tokenize.sent_tokenize(data.pop(i).strip())
            temp = sentences.pop(0).split('\n\n')
            if len(temp) == 2:
                sentences.insert(0, temp[1])
            data.insert(i, sentences)
        indexSentenceMap = collections.OrderedDict()
        for i in range(0, len(data)):
            for j in range(0, len(data[i])):
                index = 'A' + str(i + 1) + 'S' + str(j + 1)
                indexSentenceMap[index] = set(word_tokenize(data[i][j]))
        
        for k, v in indexSentenceMap.items():
            print(k, v)
                
         
            
#        dFrame = pd.DataFrame(data)
#         dFrame.to_csv('MainData.csv', index=False)
        
    '''
    Test Function
    '''    
    def dfToJson(self):
        dFrame = pd.read_csv('/Users/deepaks/Documents/workspace/Semantic_Search_Engine/pkg/MainData.csv')
        dFrame.to_json('Mainjson.json', orient='index')
    
    def indexWithSolr(self):
        solr = pysolr.Solr('http://localhost:8983/solr/default')
        solr.add([
    {
        "id": "doc_1",
        "title": "A test document",
    },
    {
        "id": "doc_2",
        "title": "A Banana:Tasty or Dangerous?",
    },
])
        connection = urlopen('http://localhost:8983/solr/default/select?q=title:A&wt=python')
        response = eval(connection.read())
        print(response['response']['numFound'], "documents found.")
        
        for document in response['response']['docs']:
            print("Name =", document['id'])

    
if __name__ == '__main__':
    dp = DataProcessing()
    path = '/Users/deepaks/Documents/workspace/Semantic_Search_Engine/Data/'
    dp.fileToDataFrame(path)
    # dp.indexWithSolr()
#     dp.dfToJson()
