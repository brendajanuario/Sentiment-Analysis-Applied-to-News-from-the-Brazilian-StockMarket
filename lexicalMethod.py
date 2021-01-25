# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 17:05:38 2020

@author: Brenda Alexsandra Januário
"""

from sklearn.datasets import load_files
import financial_dictionary
import confusionMatrix
import sentiwordnetptbr
import sentiwordnetStemmingptbr

#dicionários SEM a apliação de stemming
sentiwordnet = sentiwordnetptbr.sentiwordnetptbr
financial_dic = financial_dictionary.financial_dic
full_financial_dic = financial_dictionary.full_financial_dic

#dicionários COM a apliação de stemming
sentiwordnetStemming = sentiwordnetStemmingptbr.sentiwordnetStemming
financial_dic_stemmed = financial_dictionary.financial_dic_stem
full_financial_dic_stem = financial_dictionary.full_financial_dic_stem

accuracy = 0

############################################################################
#########               CLASSIFICADOR SEM STEMMING                 #########
############################################################################

def loadFiles():
    new_data = load_files(r"database\notícias")
    X, y = new_data.data, new_data.target
    
    return X, y


def accuracyClassifier():
    encoding = 'utf-8'
    data_set, target = loadFiles()
    hit = 0
    for index in range(0, len(data_set)):
        hit = hit + newClassifier(str(data_set[index],encoding), target[index])
        
    return hit, confusionMatrix.confusion_matrix, len(data_set)
        
def newClassifier(new, target):
    score = 0
    words = new.split()
    for word in words:
        #comente o uso do dicionário que não deseja utilizar.
#        if word in full_financial_dic: #dicionário financeiro todas classes
#            score = score + full_financial_dic[word]
        
        if word in financial_dic: #dicionário financeiro positive/negative
            score = score + financial_dic[word]        
   
        else:    
            if word in sentiwordnet:
                score = score + sentiwordnet[word]
    return confusionMatrix.confusionMatrixGenerate(score, target)

print("\n\n\nExecução da base de dados sem stemming:\n")
accuracy, matrix, dataSet = accuracyClassifier()
print(accuracy," Acertos de ",dataSet, "\nAcurácia:  ", accuracy*100/dataSet, "%")
confusionMatrix.plotMatrix(matrix)

############################################################################
#########               CLASSIFICADOR COM STEMMING                 #########
############################################################################

def loadFilesStem():
    new_data = load_files(r"database\notícias stemming")
    X, y = new_data.data, new_data.target
    
    return X, y

def accuracyClassifierStem():
    encoding = 'utf-8'
    data_set, target = loadFilesStem()
    hit = 0
    for index in range(0, len(data_set)):
        hit = hit + newClassifierStem(str(data_set[index],encoding), target[index])
        
    return hit, confusionMatrix.confusion_matrix, len(data_set)

def newClassifierStem(new, target):
    score = 0
    words = new.split()
    
    for word in words:
        #comente o uso do dicionário que não deseja utilizar.
        if word in full_financial_dic_stem: #dicionário financeiro stem todas classes
            score = score + full_financial_dic_stem[word]
        
        if word in financial_dic_stemmed: #dicionário financeiro stem positive/negative
            score = score + financial_dic_stemmed[word]
              
        else:
            if word in sentiwordnetStemming:
                score = score + sentiwordnetStemming[word]
            
    return confusionMatrix.confusionMatrixGenerate(score, target)

print("\n\n\nExecução da base de dados com stemming:\n")
accuracy, matrix, dataSet = accuracyClassifierStem()
print(accuracy," Acertos de ",dataSet, "\nAcurácia:  ", accuracy*100/dataSet, "%")
confusionMatrix.plotMatrix(matrix)


