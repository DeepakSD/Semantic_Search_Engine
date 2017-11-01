'''
Created on Oct 30, 2017

@author: deepaks
'''
import collections
import io
import json
import os
import urllib2

from nltk import tokenize
from nltk.tokenize import word_tokenize
import pysolr

import pandas as pd


class DataProcessing:
    def fileToJson(self, path):  
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
        
#         for k, v in indexSentenceMap.items():
#             print(k, v)
        dFrame = pd.DataFrame(indexSentenceMap.items(), columns=['id', 'words'])
        dFrame.to_json('Mainjson.json', orient='records')
#        print(dFrame)
        
    '''
    Test function
    '''    
    def indexWithSolr(self):
        solr = pysolr.Solr('http://localhost:8983/solr/default')
#         with open("/Users/deepaks/Documents/workspace/Semantic_Search_Engine/pkg/Mainjson.json", 'r') as data_file:
#             my_data = data_file.read()
#         print(my_data)
#         solr.add([
#     {
#         "id": "doc_1",
#         "title": ["A test document"],
#     },
#     {
#         "id": "doc_2",
#         "title": ["A Banana:Tasty or Dangerous?"],
#     },
# ])    
        solr.add(json.load("/Users/deepaks/Documents/workspace/Semantic_Search_Engine/pkg/Mainjson.json"))
        connection = urllib2.urlopen('http://localhost:8983/solr/default/select?q=title:of&wt=python')
        response = eval(connection.read())
        print(response['response']['numFound'], "documents found.")
         
        for document in response['response']['docs']:
            print("Name =", document['id'])

    
if __name__ == '__main__':
    dp = DataProcessing()
    path = '/Users/deepaks/Documents/workspace/Semantic_Search_Engine/Data/'
    dp.fileToJson(path)
    dp.indexWithSolr()
