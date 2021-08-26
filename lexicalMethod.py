# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 17:05:38 2020

@author: Brenda Alexsandra Januário
"""

from sklearn.datasets import load_files
import financial_dictionary
import sentiwordnetptbr
import sentiwordnetStemmingptbr
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

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
    y_true = []
    y_pred = []
    
    for index in range(0, len(data_set)):
        y_true.append(target[index])
        y_pred.append(newClassifier(str(data_set[index],encoding), target[index]))
        
    
    f1score = f1_score(y_true, y_pred,average=None)
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy, f1score
        

def newClassifier(new, target):
    score = 0
    words = new.split()
    for word in words:
        #comente o uso do dicionário que não deseja utilizar.
        if word in full_financial_dic: #dicionário financeiro todas classes
            score = score + full_financial_dic[word]
        
        if word in financial_dic: #dicionário financeiro positive/negative
            score = score + financial_dic[word]        
   
        else:    
            if word in sentiwordnet:
                score = score + sentiwordnet[word]
                
    if score > 0:
        y_pred = 1
        
    if score < 0:
        y_pred = 0
        
    if score == 0:
        y_pred = 2
        
    return y_pred


print("\n\n\nExecução da base de dados sem stemming:\n")
accuracy, f1score = accuracyClassifier()
print('acurácia: ',accuracy ,' f1-score: ',f1score)

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
    y_true = []
    y_pred = []
    
    for index in range(0, len(data_set)):
        y_true.append(target[index])
        y_pred.append(newClassifierStem(str(data_set[index],encoding), target[index]))
        
    
    f1score = f1_score(y_true, y_pred,average=None)
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy, f1score

def newClassifierStem(new, target):
    score = 0
    words = new.split()
    y_pred = 0
    
    for word in words:
        #comente o uso do dicionário que não deseja utilizar.
        if word in full_financial_dic_stem: #dicionário financeiro stem todas classes
            score = score + full_financial_dic_stem[word]
        
        if word in financial_dic_stemmed: #dicionário financeiro stem positive/negative
            score = score + financial_dic_stemmed[word]
              
        else:
            if word in sentiwordnetStemming:
                score = score + sentiwordnetStemming[word]

    if score > 0:
        y_pred = 1
        
    if score < 0:
        y_pred = 0
        
    if score == 0:
        y_pred = 2
        
    return y_pred

print("\n\n\nExecução da base de dados com stemming:\n")
accuracy, f1score = accuracyClassifierStem()
print('acurácia: ',accuracy ,' f1-score: ',f1score)


#0 negativo
#1 positivo