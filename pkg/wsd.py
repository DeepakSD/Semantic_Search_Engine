# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 19:37:46 2017

@author: mohanakrishnavh
"""

from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk import pos_tag

sent = ['April']


def getWordnetTagLesk(tag):
    if tag.startswith('J'):
        return 'j'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return None
        
tags = pos_tag(sent)
wordtag = tags[0]
tag = wordtag[1]
print(type(tag))
print(lesk(sent,sent[0],getWordnetTagLesk(tag[0])))


#length = len(sent)
#for each in range(0,length):
#    tag = pos_tag(sent)
#    each_tag = tag[each][1]
#    leskwithPOS = lesk(sent,sent[each],pos=getWordnetTag(each_tag))
#    print(leskwithPOS)
    
#for each in sent
#    for ss in wn.synsets(each):
##        print(each,ss,ss.definition())
#    
#    def getWordnetTag(tag):
#        if tag.startswith('J'):
#            return 'j'
#        elif tag.startswith('V'):
#            return 'v'
#        elif tag.startswith('N'):
#            return 'n'
#        elif tag.startswith('R'):
#            return 'r'
#        else:
#            return None    
#    
#    def processQueryToDoWSD(self, words):
#        wsd = []        
#        tags = pos_tag(words)
#        for each in range(0,len(words)):
#            tag = tags[each][1]
#            
#        return wsd

