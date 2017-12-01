'''
Created on Oct 30, 2017

@author: deepaks
'''
import collections
import os
import sys

from nltk import pos_tag 
from nltk.corpus import wordnet as wn
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import pysolr

from pkg.IndexCreation import IndexCreation


class SemanticSearchEngine:
    
    def getArticleAndWordCount(self, path):
        print("Number of articles:", str(len(os.listdir(path))))
        indexSentenceMap = collections.OrderedDict()
        wordCount = 0
        ic = IndexCreation()
        data = ic.readArticles(path)
        data = ic.removeArticleTitle(data)
        for i in range(0, len(data)):
            for j in range(0, len(data[i])):
                tokenizedWords = word_tokenize(data[i][j])
                index = 'A' + str(i + 1) + 'S' + str(j + 1)
                indexSentenceMap[index] = data[i][j]
                wordCount += len(tokenizedWords)
        print("Number of words in the corpus:", str(wordCount))
        return indexSentenceMap

    def processQueryToExtractWords(self, query):
        return list(set(word_tokenize(query)))
    
    def searchInSolr(self, query, indexSentenceMap):
        solr = pysolr.Solr('http://localhost:8983/solr/task2')
        query = "words:" + " || words:".join(query)
        results = solr.search(query)
        print("Top 10 documents that closely match the query")
        for result in results:
            print(result['id'].ljust(10), indexSentenceMap[result['id']])
    
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
        for taggedWord in pos_tag(words):
            posTags.append(taggedWord[1])
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
    
    def searchInSolrWithMultipleFeatures(self, featuresList, indexSentenceMap):
        solr = pysolr.Solr('http://localhost:8983/solr/task3')
        params = {'defType': 'dismax',
                  'qf':'words^1.0 lemmas^1.0 stems^1.0 POS^1.0 head^1.0 hypernyms^1.0 hyponyms^1.0 meronyms^1.0 holonyms^1.0'
                  }
     
        query1 = "words:" + " || words:".join(featuresList[0])
        query2 = "lemmas:" + " || lemmas:".join(featuresList[1])
        query3 = "stems:" + " || stems:".join(featuresList[2])
        query4 = "POS:" + " || POS:".join(featuresList[3])
        query5 = "head:" + featuresList[4]
        query6 = "hypernyms:" + " || hypernyms:".join(featuresList[5])
        query7 = "hyponyms:" + " || hyponyms:".join(featuresList[6])
        query8 = "meronyms:" + " || meronyms:".join(featuresList[7])
        query9 = "holonyms:" + " || holonyms:".join(featuresList[8])
        query = [query1, query2, query3, query4, query5, query6, query7, query8, query9]
        joinedQuery = ' || '.join(item for item in query)
        results = solr.search(joinedQuery)
        results1 = solr.search(joinedQuery, **params)
        print()
        print("Top 10 documents that closely match the query")
        for result in results:
            print(result['id'].ljust(10), indexSentenceMap[result['id']])
        print()
        print("Top 10 documents that closely match the query1")
        for result in results1:
            print(result['id'].ljust(10), indexSentenceMap[result['id']])
            
#     def goldenSetRetrieval(self, query):
#         with open('/Users/deepaks/Documents/workspace/Semantic_Search_Engine/pkg/MainData.csv', 'r', encoding='utf-8', errors='ignore') as infile:
#             reader = csv.reader(infile)
#             indexWordsMap = {rows[0]:rows[1] for rows in reader}
#         processedQuery = self.processQueryToExtractWords(query)
#         countMap = dict()
#         for k, v in indexWordsMap.items():
#             count = 0
#             for word in processedQuery:
#                 if word in v:
#                     count += 1
#             countMap[k] = count
#         countMap = sorted(countMap.items(), key=operator.itemgetter(1), reverse=True)
#         count = 0
#         for item in countMap:
#             k, v = item
#             print(k)
#             count += 1
#             if(count == 10):
#                 break

    
if __name__ == '__main__':
    sse = SemanticSearchEngine()
    path = '/Users/deepaks/Documents/workspace/Semantic_Search_Engine/Data/'
    inputChoice = input("Enter the option to continue with\n 1. Task2 \n 2. Task3\n") 
    indexSentenceMap = sse.getArticleAndWordCount(path)
    query = input("Enter the input query: ")
    # sse.goldenSetRetrieval(query)
    # Task 2
    if inputChoice == "1":
        processedQuery = sse.processQueryToExtractWords(query)
        sse.searchInSolr(processedQuery, indexSentenceMap) 
    # Task 3
    elif inputChoice == "2":
        featuresList = sse.processQueryToExtractAllFeatures(query)
        sse.searchInSolrWithMultipleFeatures(featuresList, indexSentenceMap)
        
