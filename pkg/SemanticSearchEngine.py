'''
Created on Oct 30, 2017

@author: deepaks
'''
import collections
import io
import json
import os

from nltk import tokenize
from nltk.tokenize import word_tokenize
import pysolr

import pandas as pd


class SemanticSearchEngine:
    def corpusToJson(self, path):  
        data = self.readArticles(path)
        data = self.removeArticleTitle(data)
        indexSentenceMap = self.createIndexMap(data)
        dFrame = pd.DataFrame(list(indexSentenceMap.items()), columns=['id', 'words'])
        jsonFileName = 'words.json'
        dFrame.to_json(jsonFileName, orient='records')
        return jsonFileName
        
    def readArticles(self, path):
        data = []
        for f in sorted(os.listdir(path), key=lambda x: int(x.split('.')[0])):
            with io.open(path + f, 'r', encoding='utf-8', errors='ignore') as dataFile:
                data.append(dataFile.read())
        return data
    
    def removeArticleTitle(self, data):
        for i in range(len(data)):
            sentences = tokenize.sent_tokenize(data.pop(i).strip())
            temp = sentences.pop(0).split('\n\n')
            if len(temp) == 2:
                sentences.insert(0, temp[1])
            data.insert(i, sentences)
        return data
            
    def createIndexMap(self, data):
        indexSentenceMap = collections.OrderedDict()
        for i in range(0, len(data)):
            for j in range(0, len(data[i])):
                index = 'A' + str(i + 1) + 'S' + str(j + 1)
                indexSentenceMap[index] = list(set(word_tokenize(data[i][j])))
        return indexSentenceMap
        
        
 
    # Refer https://github.com/Parsely/python-solr/blob/master/pythonsolr/pysolr.py 
    def indexWithSolr(self, jsonFileName):
        solr = pysolr.Solr('http://localhost:8983/solr/default')
        solr.delete(q='*:*')
        
        with open("/Users/deepaks/Documents/workspace/Semantic_Search_Engine/pkg/" + jsonFileName, 'rb') as jsonFile:
            entry = json.load(jsonFile)
        solr.add(entry)
        return solr
    
    def processQuery(self, query):
        return list(set(word_tokenize(query)))
    
    def searchInSolr(self, solr, query):
        tmp = ""
        for item in query:
            tmp = tmp + item + " "
        results = solr.search(q="words:" + tmp)
        print("Top 10 documents that closely match the query")
        for result in results:
            print(result['id'])
        '''
        Another format for search
        '''
#         connection = urllib2.urlopen('http://localhost:8983/solr/default/select?q=words:Time&wt=python')
#         response = eval(connection.read())
#         print(response['response']['numFound'], "documents found.")
#         for document in response['response']['docs']:
#             print("Name =", document['id'])

    
if __name__ == '__main__':
    sse = SemanticSearchEngine()
    path = '/Users/deepaks/Documents/workspace/Semantic_Search_Engine/Data/'
    jsonFileName = sse.corpusToJson(path)
    solr = sse.indexWithSolr(jsonFileName)
    query = input("Enter the input query: ")
    processedQuery = sse.processQuery(query)
    sse.searchInSolr(solr, processedQuery)     
