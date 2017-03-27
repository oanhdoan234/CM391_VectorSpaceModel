# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 12:58:49 2017

@author: Oanh Doan
"""

import nltk
import re
from nltk.data import load
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

"""******************************************************************************************************************

========================================CREATE A TERM-BY-DOCUMENT MATRIX ============================================

******************************************************************************************************************"""

#Turn document into a list of news     
def splitNews(file):
    full_text = file.read()
    split_text = full_text.split(">>>>")
    return(split_text[1:len(split_text)])

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

#Remove punctuations
def removePunct(news):
    punct = ",./?:;\|][}{~`@#^()-_+=\"\'"
    punct_list = [ch for ch in punct]
    for ch in punct_list:
        if ch in news:
            news = news.replace(ch,'')
    return(news)
        
#Remove stopwords & lemmatize
def removeStopWord(news):
    wlem = WordNetLemmatizer()
    stoplist = stopwords.words('english')
    tagdict = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS','NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP','WP$', 'WRB']
    special_tag = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'RB', 'RBR', 'RBS', 'VB', 'VBG','VBD','VBN','VBP','VBZ']
    #Remove stop words
    clean_news          = [w for w in news.split() if w not in stoplist]
    #Tag words
    tagged_clean_news   = pos_tag(clean_news)
    #List of nontrivial words with tags
    wordtag_clean_news  = [[word,pos] for word,pos in tagged_clean_news]
    #Stem words
    stemmed_clean_news = []
    for word_tag in wordtag_clean_news:
        if word_tag[1] in special_tag:
            stem_word = wlem.lemmatize(word_tag[0],get_wordnet_pos(word_tag[1]))
            stemmed_clean_news.append(stem_word)
        else:
            stemmed_clean_news.append(word_tag[0])
    return(stemmed_clean_news)



    
if __name__ == '__main__':
    inpf = open("test.txt", encoding="utf8")
    news_coll =  splitNews(inpf)            #List of news (each news = 1 string =  1 list element)
    for news in news_coll:                     
        news_no_punct =  removePunct(news)
        stemmed_news  =  " ".join(removeStopWord(news_no_punct))
        print(stemmed_news)
        #print(n)
        #each_news = " ".join(removeStopWord(n))
        #print(each_news)
    
    
    


