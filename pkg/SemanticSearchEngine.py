'''
Created on Oct 30, 2017

@author: deepaks
'''
import collections
import io
import json
import os

from nltk import tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag 
import pysolr

import pandas as pd


class SemanticSearchEngine:

    def preprocessCorpus(self, path):  
        data = self.readArticles(path)
        data = self.removeArticleTitle(data)
        
        indexWordsMap = self.createIndexMap(data)
        wordsDFrame = pd.DataFrame(list(indexWordsMap.items()), columns=['id', 'words'])
        
        indexLemmaMap = self.lemmatizeWords(indexWordsMap)
        lemmaDFrame = pd.DataFrame(list(indexLemmaMap.items()), columns=['id', 'lemmas'])
        
        indexStemMap = self.stemWords(indexWordsMap)
        stemDFrame = pd.DataFrame(list(indexStemMap.items()), columns=['id', 'stems'])
        
        indexPOSMap = self.tagPOSWords(indexWordsMap)
        POSDFrame = pd.DataFrame(list(indexPOSMap.items()), columns=['id', 'POS'])
        
        
        
        jsonFileName = 'words.json'
        wordsDFrame.to_json(jsonFileName, orient='records')
        jsonFileName = 'lemmas.json'
        lemmaDFrame.to_json(jsonFileName, orient='records')
        jsonFileName = 'stems.json'
        stemDFrame.to_json(jsonFileName, orient='records')
        jsonFileName = 'pos.json'
        POSDFrame.to_json(jsonFileName, orient='records')
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
        indexWordsMap = collections.OrderedDict()
        for i in range(0, len(data)):
            for j in range(0, len(data[i])):
                index = 'A' + str(i + 1) + 'S' + str(j + 1)
                indexWordsMap[index] = list(set(word_tokenize(data[i][j])))
        return indexWordsMap
    
    def lemmatizeWords(self, indexWordsMap):
        indexLemmaMap = collections.OrderedDict()
        wnl = WordNetLemmatizer()
        for k, v in indexWordsMap.items():
            indexLemmaMap[k] = [wnl.lemmatize(word) for word in v]
        return indexLemmaMap
    
    def stemWords(self, indexWordsMap):
        indexStemMap = collections.OrderedDict()
        stemmer = PorterStemmer()
        for k, v in indexWordsMap.items():
            indexStemMap[k] = [stemmer.stem(word) for word in v]
        return indexStemMap
    
    def tagPOSWords(self, indexWordsMap):
        indexPOSMap = collections.OrderedDict()
        for k, v in indexWordsMap.items():
            indexPOSMap[k] = [pos_tag(word) for word in v]
        return indexPOSMap
    
    def extractHypernyms(self, indexWordsMap):
        indexHypernymMap = collections.OrderedDict()
 
    # Refer https://github.com/Parsely/python-solr/blob/master/pythonsolr/pysolr.py 
    def indexWordsWithSolr(self, jsonFileName):
        solr = pysolr.Solr('http://localhost:8983/solr/task2')
        solr.delete(q='*:*')
        
        with open("/Users/deepaks/Documents/workspace/Semantic_Search_Engine/pkg/" + jsonFileName, 'rb') as jsonFile:
            entry = json.load(jsonFile)
        solr.add(entry)
        return solr
    
    def processQuery(self, query):
        return list(set(word_tokenize(query)))
    
    def searchInSolr(self, solr, query):
        query = "words:" + " & words:".join(query)
        results = solr.search(query)
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
    jsonFileName = sse.preprocessCorpus(path)
    solr = sse.indexWordsWithSolr(jsonFileName)
    query = input("Enter the input query: ")
    processedQuery = sse.processQuery(query)
    sse.searchInSolr(solr, processedQuery) 
        
