'''
Created on Oct 30, 2017

@author: deepaks
'''
import os
import pandas as pd

class DataProcessing:
    def fileToDataFrame(self,path):  
        data = []
        for f in os.listdir(path):
            with open(path+f,'r') as dataFile:
                data.append(dataFile.read())
        dFrame = pd.DataFrame(data)
        dFrame.to_csv('MainData.csv')
    
if __name__ == '__main__':
    dp = DataProcessing()
    path = '/Users/deepaks/Documents/workspace/Semantic_Search_Engine/Data/'
    dp.fileToDataFrame(path)