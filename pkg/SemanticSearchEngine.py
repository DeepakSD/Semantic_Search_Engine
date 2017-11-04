'''
Created on Oct 30, 2017

@author: deepaks
'''
import collections
import io
import os
import urllib2

from nltk import tokenize
from nltk.tokenize import word_tokenize
import pysolr

import pandas as pd


class SemanticSearchEngine:
    def corpusToJson(self, path):  
        data = self.readArticles(path)
        data = self.removeArticleTitle(data)
        indexSentenceMap = self.createIndexMap(data)
        dFrame = pd.DataFrame(indexSentenceMap.items(), columns=['id', 'words'])
        jsonFileName = 'words.json'
        dFrame.to_json(jsonFileName, orient='records')
        return jsonFileName
        
    def readArticles(self, path):
        data = []
        for f in sorted(os.listdir(path)):
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
                indexSentenceMap[index] = set(word_tokenize(data[i][j]))
        return indexSentenceMap
        
        
        
    # Refer https://github.com/Parsely/python-solr/blob/master/pythonsolr/pysolr.py 
    def indexWithSolr(self, jsonFileName):
        solr = pysolr.Solr('http://localhost:8983/solr/default')
        solr.delete(q='*:*')
        
        with open("/Users/deepaks/Documents/workspace/Semantic_Search_Engine/pkg/" + jsonFileName, 'rb') as jsonFile:
            entry = jsonFile.read()
        req = urllib2.Request('http://localhost:8983/solr/default/update/json?commit=true', entry)
        req.add_header('Content-Type', 'application/json')
        urllib2.urlopen(req)
        # print(response.read())
        return solr
    
    def searchInSolr(self, solr, query):
        results = solr.search('words:' + query)
        print("Top 10 documents with the term \"%s\"" % query)
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
    query = raw_input("Enter the input query: ")
    sse.searchInSolr(solr, query)
    
