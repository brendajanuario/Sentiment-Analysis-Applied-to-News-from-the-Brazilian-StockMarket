# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:50:16 2019

@author: b1948
"""
import re
import nltk    
nltk.download('stopwords')

stopwordsTeste = nltk.corpus.stopwords.words('portuguese')
stopwordsTeste.remove('não')
stopwords = ['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'ele', 'das', 'seu', 'sua', 'ou', 'quando', 'muito', 'nos', 'ja', 'eu', 'tambem', 'so', 'pelo', 'pela', 'ate', 'isso', 'ela', 'entre', 'depois', 'sem', 'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'voce', 'essa', 'num', 'nem', 'suas', 'meu', 'minha', 'numa', 'pelos', 'elas', 'qual', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'dele', 'tu', 'te', 'voces', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo', 'estou', 'estamos', 'estao', 'estive', 'esteve', 'estivemos', 'estiveram', 'estava', 'estavamos', 'estavam', 'estivera', 'estiveramos', 'esteja', 'estejamos', 'estejam', 'estivesse', 'estivessemos', 'estivessem', 'estiver', 'estivermos', 'estiverem', 'hei', 'ha', 'havemos', 'hao', 'houve', 'houvemos', 'houveram', 'houvera', 'houveramos', 'haja', 'hajamos', 'hajam', 'houvesse', 'houvessemos', 'houvessem', 'houver', 'houvermos', 'houverem', 'houverei', 'houveremos', 'houverao', 'houveria', 'houveriamos', 'houveriam', 'sou', 'somos', 'sao', 'era', 'eramos', 'eram', 'fui', 'foi', 'fomos', 'foram', 'fora', 'foramos', 'seja', 'sejamos', 'sejam', 'fosse', 'fossemos', 'fossem', 'for', 'formos', 'forem', 'serei', 'sera', 'seremos', 'serao', 'seria', 'seriamos', 'seriam', 'tenho', 'tem', 'temos', 'tinha', 'tinhamos', 'tinham', 'tive', 'teve', 'tivemos', 'tiveram', 'tivera', 'tiveramos', 'tenha', 'tenhamos', 'tenham', 'tivesse', 'tivessemos', 'tivessem', 'tiver', 'tivermos', 'tiverem', 'terei', 'tera', 'teremos', 'terao', 'teria', 'teriamos', 'teriam']


def preprocessamentoStemming(text):
    wordsList=[]
    stringResultado = ''
    text = re.sub( "Classificação da notícia", r"", text )#removendo numeros
    text = text.lower () #converter o texto para minusculo
    text = re.sub( r"\d", r"", text )#removendo numeros
    text = [i for i in text.split() if not i in stopwordsTeste]#removendo stopwords
    
    stemmer = nltk.stem.RSLPStemmer() #stemmer em portugues
    for word in text:        
        wordsList.append(stemmer.stem(word))    
        
    for word in wordsList:
        stringResultado = stringResultado + word + ' '
    dict = str.maketrans("ªº¨¬çàáâãçèéêíîòóôõùúû!“#$%&‘()*+–./:;<=>?@[]^_`{|}~-÷‚,„‘’“”–—¹º¼½¾","    caaaaceeeiioooouuu                                              ")
    text = stringResultado.translate(dict)#removendo caracteres especiais
    return text

def preprocessamento(text):
    stringResultado = ''
    wordsList=[]
    text = re.sub( "Classificação da notícia", r"", text )#removendo numeros
    text = text.lower () #converter o texto para minusculo
    text = re.sub( r"\d", r"", text )#removendo numeros
    wordsList = [i for i in text.split() if not i in stopwordsTeste]#removendo stopwords

    for word in wordsList:
        stringResultado = stringResultado + word + ' '
    dict = str.maketrans("ªº¨¬çàáâãçèéêíîòóôõùúû!“#$%&‘()*+–./:;<=>?@[]^_`{|}~-÷‚,„‘’“”–—¹º¼½¾","    caaaaceeeiioooouuu                                              ")
    text = stringResultado.translate(dict)#removendo caracteres especiais
    return text

stringResultado = ''
for n in range(1,2555):
    conteudo = ''
    f = open("new ("+str(n)+").txt", "rt", encoding="utf-8")

    g = preprocessamento(f.read()).split(' ')
    for el in g:
        if el != '':
            conteudo = conteudo + el + ' '
    f.close()

    file = open("new ("+str(n)+").txt", "w", encoding="utf-8")
    file.write(conteudo)
    file.close()

