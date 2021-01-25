# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:41:29 2020

@author: b1948
"""

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
confusion_matrix = np.zeros((3,3), dtype=np.int)

def plotMatrix(array):
    df_cm = pd.DataFrame(array, index = ["POSITIVA","NEGATIVA","NEUTRA"],
                      columns = ["POSITIVA","NEGATIVA","NEUTRA"])
    plt.figure(figsize = (8,6))
    sn.heatmap(df_cm, annot=True,fmt="d")
    
    
def confusionMatrixGenerate(score, target):
    global confusion_matrix
    if (score > 0 and target == 2): #
        confusion_matrix[0][0] = confusion_matrix[0][0] + 1
        return 1
    
    if (score > 0 and target == 0):
        confusion_matrix[0][1] = confusion_matrix[0][1] + 1
        return 0
    
    if (score > 0 and target == 1):
        confusion_matrix[0][2] = confusion_matrix[0][2] + 1
        return 0
    
    if (score < 0 and target == 2):    
        confusion_matrix[1][0] = confusion_matrix[1][0] + 1    
        return 0
    
    if (score < 0 and target == 0): #
        confusion_matrix[1][1] = confusion_matrix[1][1] + 1    
        return 1
    
    if (score < 0 and target == 1):
        confusion_matrix[1][2] = confusion_matrix[1][2] + 1
        return 0
    
    if (score == 0 and target == 2):    
        confusion_matrix[2][0] = confusion_matrix[2][0] + 1
        return 0
    
    if (score == 0 and target == 0):
        confusion_matrix[2][1] = confusion_matrix[2][1] + 1
        return 0
    
    if (score == 0 and target == 1): #
        confusion_matrix[2][2] = confusion_matrix[2][2] + 1
        return 1