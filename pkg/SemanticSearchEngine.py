'''
Created on Oct 30, 2017

@author: deepaks
'''
from _functools import reduce
import collections
import io
import json
import os

from nltk import pos_tag 
from nltk import tokenize
from nltk.corpus import wordnet as wn
from nltk.parse.corenlp import CoreNLPDependencyParser
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
        
        jsonFileName = 'Task2.json'
        wordsDFrame.to_json(jsonFileName, orient='records')
        return data, indexWordsMap, wordsDFrame, jsonFileName
        
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
    
    def extractFeatures(self, data, indexWordsMap):
        indexLemmaMap = self.lemmatizeWords(indexWordsMap)
        lemmaDFrame = pd.DataFrame(list(indexLemmaMap.items()), columns=['id', 'lemmas'])
        indexStemMap = self.stemWords(indexWordsMap)
        stemDFrame = pd.DataFrame(list(indexStemMap.items()), columns=['id', 'stems'])
        indexPOSMap = self.tagPOSWords(indexWordsMap)
        POSDFrame = pd.DataFrame(list(indexPOSMap.items()), columns=['id', 'POS'])
        indexHeadMap = self.findHeadWord(data)
        HeadDFrame = pd.DataFrame(list(indexHeadMap.items()), columns=['id', 'head'])  
        indexHypernymMap = self.extractHypernyms(indexWordsMap)
        HypernymDFrame = pd.DataFrame(list(indexHypernymMap.items()), columns=['id', 'hypernyms'])
        indexHyponymMap = self.extractHyponyms(indexWordsMap)
        HyponymDFrame = pd.DataFrame(list(indexHyponymMap.items()), columns=['id', 'hyponyms'])
        indexMeronymMap = self.extractMeronyms(indexWordsMap)
        MeronymDFrame = pd.DataFrame(list(indexMeronymMap.items()), columns=['id', 'meronyms'])
        indexHolonymMap = self.extractHolonyms(indexWordsMap)
        HolonymDFrame = pd.DataFrame(list(indexHolonymMap.items()), columns=['id', 'holonyms'])
        dfList = [lemmaDFrame, stemDFrame, POSDFrame, HeadDFrame, HypernymDFrame, HyponymDFrame, MeronymDFrame, HolonymDFrame]
        finalDFrame = reduce(lambda left, right: pd.merge(left, right, on='id'), dfList)
        
        jsonFileName = 'Task3.json'
        finalDFrame.to_json(jsonFileName, orient='records')
        return jsonFileName
    
    def lemmatizeWords(self, indexWordsMap):
        print("Lemmatizing...")
        indexLemmaMap = collections.OrderedDict()
        wnl = WordNetLemmatizer()
        for k, v in indexWordsMap.items():
            indexLemmaMap[k] = [wnl.lemmatize(word) for word in v]
        return indexLemmaMap
    
    def stemWords(self, indexWordsMap):
        print("Stemming...")
        indexStemMap = collections.OrderedDict()
        stemmer = PorterStemmer()
        for k, v in indexWordsMap.items():
            indexStemMap[k] = [stemmer.stem(word) for word in v]
        return indexStemMap
    
    def tagPOSWords(self, indexWordsMap):
        print("POS Tagging...")
        indexPOSMap = collections.OrderedDict()
        for k, v in indexWordsMap.items():
            indexPOSMap[k] = pos_tag(v)
        return indexPOSMap
    
    def findHeadWord(self, data):
        print("Head Word Extraction...")
        indexHeadMap = collections.OrderedDict()
        dependency_parser = CoreNLPDependencyParser('http://localhost:9000')
        for i in range(0, len(data)):
            for j in range(0, len(data[i])):
                index = 'A' + str(i + 1) + 'S' + str(j + 1)
                parsedSentence = list(dependency_parser.raw_parse(data[i][j]))[0]
                rootValue = list(list(parsedSentence.nodes.values())[0]['deps']['ROOT'])[0]
                for n in parsedSentence.nodes.values():
                    if n['address'] == rootValue:
                        indexHeadMap[index] = n['word']
                        break
        return indexHeadMap 
    
    def extractHypernyms(self, indexWordsMap):
        print("Hypernyms Extraction..")
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
        print("Hyponyms Extraction..")
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
        print("Meronyms Extraction..")
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
        print("Holonyms Extraction..")
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
        print("Indexing for Task2...")
        solr = pysolr.Solr('http://localhost:8983/solr/task2')
        solr.delete(q='*:*')
        
        with open("/Users/deepaks/Documents/workspace/Semantic_Search_Engine/pkg/" + jsonFileName, 'rb') as jsonFile:
            entry = json.load(jsonFile)
        solr.add(entry)
        return solr
    
    def processQueryToExtractWords(self, query):
        return list(set(word_tokenize(query)))
    
    def searchInSolr(self, solr, query):
        query = "words:" + " & words:".join(query)
        results = solr.search(query)
        print("Top 10 documents that closely match the query")
        for result in results:
            print(result['id'])
    
    def indexFeaturesWithSolr(self, jsonFileName):
        print("Indexing for Task3...")
        solr = pysolr.Solr('http://localhost:8983/solr/task3')
        solr.delete(q='*:*')
        
        with open("/Users/deepaks/Documents/workspace/Semantic_Search_Engine/pkg/" + jsonFileName, 'rb') as jsonFile:
            entry = json.load(jsonFile)
        solr.add(entry)
        return solr
    
    def processQueryToDoLemmatization(self, words):
        lemmas = []
        wnl = WordNetLemmatizer()
        for word in words:
            lemmas.append(wnl.lemmatize(word))
        return lemmas
    
    def processQueryToDoStemming(self, words):
        stems = []
        stemmer = PorterStemmer()
        for word in words:
            stems.append(stemmer.stem(word))
        return stems
    
    def processQueryToDoPOSTagging(self, words):
        posTags = []
        for word in words:
            posTags.append(pos_tag(word))
        return posTags
    
    def processQueryToExtractHeadWord(self, query):
        dependency_parser = CoreNLPDependencyParser('http://localhost:9000')
        headWord = None
        parsedSentence = list(dependency_parser.raw_parse(query))[0]
        rootValue = list(list(parsedSentence.nodes.values())[0]['deps']['ROOT'])[0]
        for n in parsedSentence.nodes.values():
            if n['address'] == rootValue:
                headWord = n['word']
                break
        return headWord 
    
    def processQueryToExtractHypernyms(self, words):
        hypernyms = []
        for word in words:
            '''Can use common Hypernyms for Task 4'''
            synset = wn.synsets(word)
            if len(synset) > 0:
                if len(synset[0].hypernyms()) > 0:
                    hypernyms.append(synset[0].hypernyms()[0].name().split('.')[0])
                else:
                    hypernyms.append(word)
        return hypernyms
    
    def processQueryToExtractHyponyms(self, words):
        hyponyms = []
        for word in words:
            '''Can use common Hyponyms for Task 4'''
            synset = wn.synsets(word)
            if len(synset) > 0:
                if len(synset[0].hyponyms()) > 0:
                    hyponyms.append(synset[0].hyponyms()[0].name().split('.')[0])
                else:
                    hyponyms.append(word)
        return hyponyms
    
    def processQueryToExtractMeronyms(self, words):
        meronyms = []
        for word in words:
            '''Can use common Meronyms for Task 4'''
            synset = wn.synsets(word)
            if len(synset) > 0:
                if len(synset[0].part_meronyms()) > 0:
                    meronyms.append(synset[0].part_meronyms()[0].name().split('.')[0])
                else:
                    meronyms.append(word)
        return meronyms
    
    def processQueryToExtractHolonyms(self, words):
        holonyms = []
        for word in words:
            '''Can use common Holonyms for Task 4'''
            synset = wn.synsets(word)
            if len(synset) > 0:
                if len(synset[0].part_holonyms()) > 0:
                    holonyms.append(synset[0].part_holonyms()[0].name().split('.')[0])
                else:
                    holonyms.append(word)
        return holonyms
    
    def processQueryToExtractAllFeatures(self, query):
        words = self.processQueryToExtractWords(query)
        lemmas = self.processQueryToDoLemmatization(words)
        stems = self.processQueryToDoStemming(words)
        posTags = self.processQueryToDoPOSTagging(words)
        headWord = self.processQueryToExtractHeadWord(query)
        hypernyms = self.processQueryToExtractHypernyms(words)
        hyponyms = self.processQueryToExtractHyponyms(words)
        meronyms = self.processQueryToExtractMeronyms(words)
        holonyms = self.processQueryToExtractHolonyms(words)
        return [words, lemmas, stems, posTags, headWord, hypernyms, hyponyms, meronyms, holonyms]
    
    def searchInSolrWithMultipleFeatures(self, solr, featuresList):
        query = "words:" + " & words:".join(featuresList[0])
        results = solr.search(query)
        print("Top 10 documents that closely match the query")
        for result in results:
            print(result['id'])

    
if __name__ == '__main__':
    sse = SemanticSearchEngine()
    path = '/Users/deepaks/Documents/workspace/Semantic_Search_Engine/Data/'
    # Task 2
    data, indexWordsMap, wordsDFrame, jsonFileName = sse.preprocessCorpus(path)
    solr = sse.indexWordsWithSolr('Task2.json')
    query = input("Enter the input query: ")
    processedQuery = sse.processQueryToExtractWords(query)
    sse.searchInSolr(solr, processedQuery) 
    # Task 3
#     jsonFileName = sse.extractFeatures(data, indexWordsMap)
    solr = sse.indexFeaturesWithSolr('Task3.json')
    query = input("Enter the input query: ")
    featuresList = sse.processQueryToExtractAllFeatures(query)
    sse.searchInSolrWithMultipleFeatures(solr, featuresList)
        
