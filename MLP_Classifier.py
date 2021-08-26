# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 20:02:31 2020

@author: b1948
"""
import re
from sklearn.datasets import load_files
stopwords = ['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'ele', 'das', 'seu', 'sua', 'ou', 'quando', 'muito', 'nos', 'ja', 'eu', 'tambem', 'so', 'pelo', 'pela', 'ate', 'isso', 'ela', 'entre', 'depois', 'sem', 'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'voce', 'essa', 'num', 'nem', 'suas', 'meu', 'minha', 'numa', 'pelos', 'elas', 'qual', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'dele', 'tu', 'te', 'voces', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo', 'estou', 'estamos', 'estao', 'estive', 'esteve', 'estivemos', 'estiveram', 'estava', 'estavamos', 'estavam', 'estivera', 'estiveramos', 'esteja', 'estejamos', 'estejam', 'estivesse', 'estivessemos', 'estivessem', 'estiver', 'estivermos', 'estiverem', 'hei', 'ha', 'havemos', 'hao', 'houve', 'houvemos', 'houveram', 'houvera', 'houveramos', 'haja', 'hajamos', 'hajam', 'houvesse', 'houvessemos', 'houvessem', 'houver', 'houvermos', 'houverem', 'houverei', 'houveremos', 'houverao', 'houveria', 'houveriamos', 'houveriam', 'sou', 'somos', 'sao', 'era', 'eramos', 'eram', 'fui', 'foi', 'fomos', 'foram', 'fora', 'foramos', 'seja', 'sejamos', 'sejam', 'fosse', 'fossemos', 'fossem', 'for', 'formos', 'forem', 'serei', 'sera', 'seremos', 'serao', 'seria', 'seriamos', 'seriam', 'tenho', 'tem', 'temos', 'tinha', 'tinhamos', 'tinham', 'tive', 'teve', 'tivemos', 'tiveram', 'tivera', 'tiveramos', 'tenha', 'tenhamos', 'tenham', 'tivesse', 'tivessemos', 'tivessem', 'tiver', 'tivermos', 'tiverem', 'terei', 'tera', 'teremos', 'terao', 'teria', 'teriamos', 'teriam']
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from statistics import stdev, fmean



def nayve_bayes_new(X, y, file):

    clf = MultinomialNB()
    
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(X, y)
    
    score = [] 
    f1Score = []
    
    for train_index, test_index in skf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        score.append(clf.fit(X[train_index], y[train_index]).score(X[test_index], y[test_index])) #acuracia
        y_pred_mlp = clf.predict(X[test_index])
        f1Score.append(f1_score(y[test_index], y_pred_mlp, average='binary'))

    print('acurácia: ', fmean(score), " desvio padrão: ", stdev(score))
    print('f1-score ', fmean(f1Score),"desvio padrão: ", stdev(f1Score))
    
    file.write("{}\t{}\n".format(fmean(score), stdev(score)))
    file.write("{}\t{}\n\n".format(fmean(f1Score), stdev(f1Score)))
    
    
def mlp_classifier_new(config_mlp, X, y, file):
    
    clf = MLPClassifier(hidden_layer_sizes=(config_mlp), max_iter=1000) 
    
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(X, y)
    
    score = [] 
    f1Score = []
    
    for train_index, test_index in skf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        score.append(clf.fit(X[train_index], y[train_index]).score(X[test_index], y[test_index])) #acuracia
        y_pred_mlp = clf.predict(X[test_index])
        f1Score.append(f1_score(y[test_index], y_pred_mlp, average='binary'))    
    
    print('acurácia: ', fmean(score), " desvio padrão: ", (stdev(score)))
    print('f1-score ', fmean(f1Score),"desvio padrão: ", (stdev(f1Score)))
    
    file.write("\ncamadas: {}\tneuronios: {}\n".format(len(config_mlp),config_mlp[0]))
    file.write("{}\t{}\n".format(fmean(score), stdev(score)))
    file.write("{}\t{}\n\n".format(fmean(f1Score), stdev(f1Score)))


def generate_config_mlp(X, y, file):
    file.write("***\n\n\n")

    camadas = [10, 25, 50, 100]
    neuronios = [1,3,5,10]
    
    for n_camadas in camadas:
        for n_neuronio in neuronios:
            config_mlp = (1,)*n_neuronio
    
            config_mlp= tuple(i * n_camadas for i in config_mlp)
            mlp_classifier_new(config_mlp, X, y, file)


def import_data():
    
    news_data_stm = load_files(r"database\notícias stemming")               
    news_data = load_files(r"database\notícias")
        
    X_stm, y_stm = news_data_stm.data, news_data_stm.target
    X, y = news_data.data, news_data.target

    return X_stm, y_stm, X, y


def regularize_text(X):
    
    documents = []

    for sen in range(0, len(X)):
        document = re.sub(r'\W', ' ', str(X[sen]))    
        # remove all single characters
        document = re.sub(r"\b[a-zA-Z]\b", ' ', document)
        
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        
        # Removing prefixed 'b'
        document = document.encode().decode('utf-8')
        
        documents.append(document)
        
    return documents


def make_array_news(X): 
    
    array_news = []
    
    vectorizer = CountVectorizer(max_features=1500, min_df=3,
                                 max_df=0.7, stop_words=stopwords)
    #BoW
    X_bow = vectorizer.fit_transform(X).toarray()
    array_news.append(X_bow)
    
    #TF
    X_tf = TfidfTransformer(use_idf=False).fit_transform(X_bow).toarray()
    array_news.append(X_tf)
    
    #TF-IDF
    X_tf_idf = TfidfTransformer(use_idf=True).fit_transform(X_bow).toarray()
    array_news.append(X_tf_idf)
    
    return array_news


def main():
    
    dic_X = {}
    dic_y = {}
    
    X_stm, y_stm, X, y = import_data()
    X_stm, X = regularize_text(X_stm), regularize_text(X)
    
    dic_X['array_new_stm'], dic_X['array_new'] = make_array_news(X_stm), make_array_news(X)
    dic_y['array_new_stm'], dic_y['array_new'] = y_stm, y
    
    for key in dic_X:
        print(key)
        for array in dic_X[key]:
            file_nb = open("results_naive_bayes.txt", "a") 
            file_mlp = open("results_mlp.txt", "a") 

            #nayve_bayes_new(array, dic_y[key], file_nb)
            generate_config_mlp(array, dic_y[key], file_mlp)
            
    file_nb.close()
    file_mlp.close()
    print('fim')
    
main()

    
    
'''frequencia de termos naive bayes
acrescentar desvio padrão'''


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    