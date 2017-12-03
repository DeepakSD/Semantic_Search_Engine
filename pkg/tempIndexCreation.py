# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:13:52 2017

@author: mohanakrishnavh
"""

import csv
import pandas as pd
import io
import os
import collections
import pysolr
import json

from nltk import tokenize
from nltk.tokenize import word_tokenize
from _functools import reduce
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.corpus import wordnet as wn


def preprocessCorpus(path):
    print("Pre-processing and Tokenizing...")
    data = readArticles(path)
    data = removeArticleTitle(data)

    indexWordsMap = createIndexMap(data)
    with io.open('data.csv', 'w', encoding='utf-8', errors='ignore') as f:
        w = csv.writer(f)
        w.writerows(indexWordsMap.items())
    wordsDFrame = pd.DataFrame(list(indexWordsMap.items()), columns=['id', 'words'])

    jsonFileName = 'Task2.json'
    wordsDFrame.to_json(jsonFileName, orient='records')
    return data, indexWordsMap, wordsDFrame, jsonFileName

def readArticles(path):
    data = []
    for f in sorted(os.listdir(path), key=lambda x: int(x.split('.')[0])):
        with io.open(path + f, 'r', encoding='utf-8', errors='ignore') as dataFile:
            data.append(dataFile.read())
    return data

def removeArticleTitle(data):
    for i in range(len(data)):
        sentences = tokenize.sent_tokenize(data.pop(i).strip())
        temp = sentences.pop(0).split('\n\n')
        if len(temp) == 2:
            sentences.insert(0, temp[1])
        data.insert(i, sentences)
    return data

def createIndexMap(data):
    indexWordsMap = collections.OrderedDict()
    for i in range(0, len(data)):
        for j in range(0, len(data[i])):
            index = 'A' + str(i + 1) + 'S' + str(j + 1)
            indexWordsMap[index] = list(set(word_tokenize(data[i][j])))
    return indexWordsMap

# Refer https://github.com/Parsely/python-solr/blob/master/pythonsolr/pysolr.py    
def indexFeaturesWithSolr(jsonFileName, inputChoice):
    print("Indexing...")
    solr = pysolr.Solr('http://localhost:8983/solr/task' + str(int(inputChoice) + 1))
    solr.delete(q='*:*')
    with open("/home/mohanakrishnavh/Desktop/NLP/NLPV4/Semantic_Search_Engine/pkg/" + jsonFileName, 'r') as jsonFile:
        entry = json.load(jsonFile)
    solr.add(entry)
    print("Indexing Completed")
    
def lemmatizeWords(indexWordsMap):
    print("Lemmatizing...")
    indexLemmaMap = collections.OrderedDict()
    wnl = WordNetLemmatizer()
    for k, v in indexWordsMap.items():
        indexLemmaMap[k] = [wnl.lemmatize(word) for word in v]
    return indexLemmaMap
    
def stemWords(indexWordsMap):
    print("Stemming...")
    indexStemMap = collections.OrderedDict()
    stemmer = PorterStemmer()
    for k, v in indexWordsMap.items():
        indexStemMap[k] = [stemmer.stem(word) for word in v]
    return indexStemMap
    
def tagPOSWords(indexWordsMap):
    print("POS Tagging...")
    indexPOSMap = collections.OrderedDict()
    for k, v in indexWordsMap.items():
        posTags = []
        for taggedWord in pos_tag(v):
            posTags.append(taggedWord[1])
        indexPOSMap[k] = posTags
    return indexPOSMap
    
def findHeadWord(data):
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
    
def extractHypernyms(indexWordsMap):
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
        print(k,indexHypernymMap[k])
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
 
def extractFeatures(data, indexWordsMap):
    #Words
    wordsDFrame = pd.DataFrame(list(indexWordsMap.items()), columns=['id', 'words'])    
    #Lemmatizing
    indexLemmaMap = lemmatizeWords(indexWordsMap)
    lemmaDFrame = pd.DataFrame(list(indexLemmaMap.items()), columns=['id', 'lemmas'])    
    #Stemming
    indexStemMap = stemWords(indexWordsMap)
    stemDFrame = pd.DataFrame(list(indexStemMap.items()), columns=['id', 'stems'])    
    #POSTagging
    indexPOSMap = tagPOSWords(indexWordsMap)
    POSDFrame = pd.DataFrame(list(indexPOSMap.items()), columns=['id', 'POS'])    
    #HeadWord
    indexHeadMap = findHeadWord(data)
    HeadDFrame = pd.DataFrame(list(indexHeadMap.items()), columns=['id', 'head'])    
    #Hypernyms
    indexHypernymMap = extractHypernyms(indexWordsMap)
    HypernymDFrame = pd.DataFrame(list(indexHypernymMap.items()), columns=['id', 'hypernyms'])    
    #Hyponyms
    indexHyponymMap = extractHyponyms(indexWordsMap)
    HyponymDFrame = pd.DataFrame(list(indexHyponymMap.items()), columns=['id', 'hyponyms'])    
    #Meronyms
    indexMeronymMap = extractMeronyms(indexWordsMap)
    MeronymDFrame = pd.DataFrame(list(indexMeronymMap.items()), columns=['id', 'meronyms'])    
    #Holonyms
    indexHolonymMap = extractHolonyms(indexWordsMap)
    HolonymDFrame = pd.DataFrame(list(indexHolonymMap.items()), columns=['id', 'holonyms'])    
    #Mering all feature dataframes into single datafram
    dfList = [wordsDFrame, lemmaDFrame, stemDFrame, POSDFrame, HeadDFrame, HypernymDFrame, HyponymDFrame, MeronymDFrame, HolonymDFrame]
    finalDFrame = reduce(lambda left, right: pd.merge(left, right, on='id'), dfList)
    jsonFileName = 'Task3.json'
    finalDFrame.to_json(jsonFileName, orient='records')
    return jsonFileName

path = '/home/mohanakrishnavh/Desktop/NLP/NLPV4/Semantic_Search_Engine/Data/'
inputChoice = input("Enter the option to continue with\n 1. Task2 \n 2. Task3\n 3. Task4\n ") 
data, indexWordsMap, wordsDFrame, jsonFileName = preprocessCorpus(path)
if inputChoice == "1":
    indexFeaturesWithSolr(jsonFileName, inputChoice)
elif inputChoice == "2":
    jsonFileName = extractFeatures(data, indexWordsMap)
    indexFeaturesWithSolr(jsonFileName, inputChoice)    