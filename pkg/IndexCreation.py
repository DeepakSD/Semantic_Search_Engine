'''
Created on Nov 22, 2017

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
import csv


class IndexCreation():

    def preprocessCorpus(self, path):
        print("Pre-processing and Tokenizing...")
        data = self.readArticles(path)
        data = self.removeArticleTitle(data)

        indexWordsMap = self.createIndexMap(data)
        with io.open('MainData.csv', 'w', encoding='utf-8', errors='ignore') as f:
            w = csv.writer(f)
            w.writerows(indexWordsMap.items())
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
        wordsDFrame = pd.DataFrame(list(indexWordsMap.items()), columns=['id', 'words'])
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
        dfList = [wordsDFrame, lemmaDFrame, stemDFrame, POSDFrame, HeadDFrame, HypernymDFrame, HyponymDFrame, MeronymDFrame, HolonymDFrame]
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
            posTags = []
            for taggedWord in pos_tag(v):
                posTags.append(taggedWord[1])
            indexPOSMap[k] = posTags
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
        print("Hypernyms Extraction...")
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
        print("Hyponyms Extraction...")
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
        print("Meronyms Extraction...")
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
        print("Holonyms Extraction...")
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
    def indexFeaturesWithSolr(self, jsonFileName):
        print("Indexing...")
        solr = pysolr.Solr('http://localhost:8983/solr/task3')
        # solr.delete(q='*:*')

        with open("/Users/deepaks/Documents/workspace/Semantic_Search_Engine/pkg/" + jsonFileName, 'rb') as jsonFile:
            entry = json.load(jsonFile)
        solr.add(entry)


if __name__ == '__main__':
    ic = IndexCreation()
    path = '/Users/deepaks/Documents/workspace/Semantic_Search_Engine/Data/'
    data, indexWordsMap, wordsDFrame, jsonFileName = ic.preprocessCorpus(path)
    ic.indexFeaturesWithSolr(jsonFileName)

    jsonFileName = ic.extractFeatures(data, indexWordsMap)
    ic.indexFeaturesWithSolr(jsonFileName)
