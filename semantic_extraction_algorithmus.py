#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:23:58 2020

@author: wangyang
"""
#Importieren zunächst die benötigten Bibliotheken
import requests
import bs4
import numpy as np
import nltk 
import re
import networkx as nx
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from sklearn.metrics.pairwise import cosine_similarity


#Beginnen mit dem Herunterladen eines aktuellen Wikipedia-Texts
#url= 'https://en.wikipedia.org/wiki/Computer_vision'
#url='https://en.wikipedia.org/wiki/Natural_language_processing'
url='https://en.wikipedia.org/wiki/Automated_driving_system'
text=[]
response=requests.get(url)
content=bs4.BeautifulSoup(response.text,'html.parser')
paragraphs=content.select("p")
#Eine Liste von Text
for para in paragraphs:
    text.append(para.text)
    

#Schreiben den Text auf die Festplatte unter dem Verzeichnis save_dir
save_dir="/Users/wangyang/Desktop/text.txt"   
with open(save_dir,'w',encoding='utf-8') as f:
    for line in text:
        #strip() entfernt \n, sub() entfernt Zitat
        newline=re.sub(u"\\[.*?]","",line.strip('\n'))
        f.write(str(newline))


#Datei öffnen
fp=open(save_dir)
data=fp.read()


#Trennen die Elemente durch [.!?], um die Sätze zu erhalten
def splitTextIntoSentences(text):
    sentenceEnders=re.compile('[.!?]')
    sentences_list=sentenceEnders.split(text)
    return sentences_list
sentences_list=splitTextIntoSentences(data)


#Eine andere Methode, die nicht gut funktioniert
#from nltk import tokenize
#sentences_list=tokenize.sent_tokenize(data)


#Lade stopwords
stop_words= stopwords.words('english')
#Wortsegmentierung
word_tokenizer= RegexpTokenizer(r'\w+')
def seg_depart(sentence):
    word_tokens=word_tokenizer.tokenize(sentence)
    #Wenn der Satz herausgefiltert wird, gibt es [] zurück, wobei die Anzahl der Sätze unverändert bleibt
    filtered_sentence=[w for w in word_tokens if not (w in stop_words or w.isdigit())]
    return filtered_sentence
sentence_word_list=[]
for sentence in sentences_list:
    line_seg=seg_depart(sentence)
    sentence_word_list.append(line_seg)
   
    
tokenizer=Tokenizer()
#fit_on_texts kann jedes Wort im Text nummerieren
tokenizer.fit_on_texts(sentence_word_list)
#Erzeugen die Abbildung zwischen Wörtern und Indizes
vocab=tokenizer.word_index


#Das Vokabular der heruntergeladenen angelernten GloVe-Einbettungen enthält 2,2 Millionen Wörter
#Setzen den pfad zu den heruntergeladenen Wortvektoren
path_to_glove="/Users/wangyang/Desktop/glove.840B.300d.txt"
#Vwewenden nur die GloVe-Vektoren für Wörter, die in unserem eigenen Vokabular vorkommen
def get_glove(path_to_glove,vocab):
    embedding_weights={}
    count_all_words=0
    with open("glove.840B.300d.txt",'r', errors = 'ignore', encoding='utf8') as f:
        for line in f:
            vals=line.split()
            word = ''.join(vals[:-300])
            if word in vocab:
                count_all_words+=1
                coefs = np.asarray(vals[-300:], dtype='float32')
                #Normalisieren die Wort-Einbettungsvektoren
                coefs/=np.linalg.norm(coefs)
                embedding_weights[word]=coefs
            #Sobald die benötigten Wörter extrahiert werden, brechen den Vorgang ab
            if count_all_words==len(vocab)-1:
                break
    #Wenn das Wort nicht in der Liste enthalten ist,setzen den Vektor auf einen 300 dimensionalen Nullvektor
        if count_all_words<len(vocab)-1:
            for word,i in vocab.items():
                if word in embedding_weights:
                    continue
                else:
                    embedding_weights[word]=np.zeros((300,))
                    print(word)
    #Die Ausgabe ist ein Dictionary, das jedes Wort auf seinen Vektor abbildet
    return embedding_weights
word2embedding_dict=get_glove(path_to_glove, vocab)


sentence_vectors=[]
for line in sentence_word_list:
    if len(line)!=0:
        #Berechnen den Durchschnitt der Wortvektoren, um eine Vektordarstellung des Satzes zu erhalten
        v=sum(word2embedding_dict.get(word)for word in line)/(len(line))
    else:
        #Wenn der Satz [] ist, setzen den Satzvektor auf einen 300 dimensionalen Nullvektor
        v=np.zeros((300,))
    sentence_vectors.append(v)
    
    
#Berechnen die Cosinus-Ähnlichkeit zwischen Sätzen, um eine Ähnlichkeitsmatrix zu bilden
sim_mat=np.zeros([len(sentences_list),len(sentences_list)])
for i in range(len(sentences_list)):
    for j in range(len(sentences_list)):
        if i!=j:
            sim_mat[i][j]=cosine_similarity(sentence_vectors[i].reshape(1,300),
                                            sentence_vectors[j].reshape(1,300))
#Konstruieren eine Graphstruktur mit Sätzen als Knoten und Satzähnlichkeit als Übergangswahrscheinlichkeit
nx_graph= nx.from_numpy_array(sim_mat)
#Erhalten textrank-Werte aller Sätze
scores=nx.pagerank(nx_graph)
#Sortieren unverarbeitete Sätze basierend auf textrank-Werten
ranked_sentences= sorted(((scores[i],s)for i,s in enumerate(sentences_list)),reverse=True)
#Nehmen die 3 Sätze mit der höchsten Punktzahl als Thema
for i in range(3):
    print("Thema"+str(i+1)+"\n\n",ranked_sentences[i][1],'\n')



