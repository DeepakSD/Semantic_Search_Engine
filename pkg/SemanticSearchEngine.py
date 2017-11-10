'''
Created on Oct 30, 2017

@author: deepaks
'''
import collections
import io
import json
import os

from nltk import pos_tag 
from nltk import tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import pysolr

import pandas as pd


class SemanticSearchEngine:

    def preprocessCorpus(self, path):  
        data = self.readArticles(path)
        data = self.removeArticleTitle(data)
        
        indexWordsMap = self.createIndexMap(data)
        wordsDFrame = pd.DataFrame(list(indexWordsMap.items()), columns=['id', 'words'])
        
        jsonFileName = 'words.json'
        wordsDFrame.to_json(jsonFileName, orient='records')
        return indexWordsMap, jsonFileName
        
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
    
    def extractFeatures(self, indexWordsMap):
        indexLemmaMap = self.lemmatizeWords(indexWordsMap)
        lemmaDFrame = pd.DataFrame(list(indexLemmaMap.items()), columns=['id', 'lemmas'])
        indexStemMap = self.stemWords(indexWordsMap)
        stemDFrame = pd.DataFrame(list(indexStemMap.items()), columns=['id', 'stems'])
        indexPOSMap = self.tagPOSWords(indexWordsMap)
        POSDFrame = pd.DataFrame(list(indexPOSMap.items()), columns=['id', 'POS'])  
        indexHypernymMap = self.extractHypernyms(indexWordsMap)
        HypernymDFrame = pd.DataFrame(list(indexHypernymMap.items()), columns=['id', 'hypernyms'])
        indexHyponymMap = self.extractHyponyms(indexWordsMap)
        HyponymDFrame = pd.DataFrame(list(indexHyponymMap.items()), columns=['id', 'hyponyms'])
        indexMeronymMap = self.extractMeronyms(indexWordsMap)
        MeronymDFrame = pd.DataFrame(list(indexMeronymMap.items()), columns=['id', 'meronyms'])
        indexHolonymMap = self.extractHolonyms(indexWordsMap)
        HolonymDFrame = pd.DataFrame(list(indexHolonymMap.items()), columns=['id', 'holonyms'])
        
        jsonFileList = ['words.json', 'lemmas.json', 'stems.json', 'pos.json', 'hypernym.json', 'hyponym.json', 'meronym.json', 'holonym.json']
        lemmaDFrame.to_json(jsonFileList[1], orient='records')
        stemDFrame.to_json(jsonFileList[2], orient='records')
        POSDFrame.to_json(jsonFileList[3], orient='records')
        HypernymDFrame.to_json(jsonFileList[4], orient='records')
        HyponymDFrame.to_json(jsonFileList[5], orient='records')
        MeronymDFrame.to_json(jsonFileList[6], orient='records')
        HolonymDFrame.to_json(jsonFileList[7], orient='records')
        return jsonFileList
    
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
        for k, v in indexWordsMap.items():
            hypernymList = []
            '''Can use common Hypernyms for Task 4'''
            for word in v:
                synset = wn.synsets(word)
                if len(synset) > 0:
                    if len(synset[0].hypernyms()) > 0:
                        hypernymList.append(synset[0].hypernyms()[0].name().split('.')[0])
                else:
                    hypernymList.append(word)
            indexHypernymMap[k] = hypernymList
        return indexHypernymMap
    
    def extractHyponyms(self, indexWordsMap):
        indexHyponymMap = collections.OrderedDict()
        for k, v in indexWordsMap.items():
            hyponymList = []
            '''Can use common Hyponyms for Task 4'''
            for word in v:
                synset = wn.synsets(word)
                if len(synset) > 0:
                    if len(synset[0].hyponyms()) > 0:
                        hyponymList.append(synset[0].hyponyms()[0].name().split('.')[0])
                else:
                    hyponymList.append(word)
            indexHyponymMap[k] = hyponymList
        return indexHyponymMap
    
    def extractMeronyms(self, indexWordsMap):
        indexMeronymMap = collections.OrderedDict()
        for k, v in indexWordsMap.items():
            meronymList = []
            for word in v:
                synset = wn.synsets(word)
                if len(synset) > 0:
                    if len(synset[0].part_meronyms()) > 0:
                        meronymList.append(synset[0].part_meronyms()[0].name().split('.')[0])
                else:
                    meronymList.append(word)
            indexMeronymMap[k] = meronymList
        return indexMeronymMap
    
    def extractHolonyms(self, indexWordsMap):
        indexHolonymMap = collections.OrderedDict()
        for k, v in indexWordsMap.items():
            holonymList = []
            for word in v:
                synset = wn.synsets(word)
                if len(synset) > 0:
                    if len(synset[0].part_holonyms()) > 0:
                        holonymList.append(synset[0].part_holonyms()[0].name().split('.')[0])
                else:
                    holonymList.append(word)
            indexHolonymMap[k] = holonymList
        return indexHolonymMap
 
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
    
    def indexFeaturesWithSolr(self, jsonFileList):
        solr = pysolr.Solr('http://localhost:8983/solr/task3')
        solr.delete(q='*:*')
        
        for jsonFileName in jsonFileList:
            with open("/Users/deepaks/Documents/workspace/Semantic_Search_Engine/pkg/" + jsonFileName, 'rb') as jsonFile:
                entry = json.load(jsonFile)
            solr.add(entry)
        return solr
    
    
if __name__ == '__main__':
    sse = SemanticSearchEngine()
    path = '/Users/deepaks/Documents/workspace/Semantic_Search_Engine/Data/'
    # Task 2
    indexWordsMap, jsonFileName = sse.preprocessCorpus(path)
    solr = sse.indexWordsWithSolr(jsonFileName)
    query = input("Enter the input query: ")
    processedQuery = sse.processQuery(query)
    sse.searchInSolr(solr, processedQuery) 
    # Task 3
    jsonFileList = sse.extractFeatures(indexWordsMap)
    solr = sse.indexFeatureswithSolr(jsonFileList)
        
