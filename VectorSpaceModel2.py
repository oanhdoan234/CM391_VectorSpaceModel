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


# Turn document into a list of news
def splitNews(file):
    full_text = file.read()
    split_text = full_text.split(">>>>")
    return (split_text[1:len(split_text)])


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


# Remove punctuations
def removePunct(news):
    punct = ",./?:;\|][}{~`@#^()-_+=\"\'"
    punct_list = [ch for ch in punct]
    for ch in punct_list:
        if ch in news:
            news = news.replace(ch, '')
    return (news)


# Remove stopwords & lemmatize
def removeStopWord(news):
    wlem = WordNetLemmatizer()
    stoplist = stopwords.words('english')
    tagdict = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
               'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
               'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
    special_tag = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'RB', 'RBR', 'RBS', 'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']
    # Remove stop words
    clean_news = [w for w in news.split() if w not in stoplist]
    # Tag words
    tagged_clean_news = pos_tag(clean_news)
    # List of nontrivial words with tags
    wordtag_clean_news = [[word, pos] for word, pos in tagged_clean_news]
    # Stem words
    stemmed_clean_news = []
    for word_tag in wordtag_clean_news:
        if word_tag[1] in special_tag:
            stem_word = wlem.lemmatize(word_tag[0], get_wordnet_pos(word_tag[1]))
            stemmed_clean_news.append(stem_word)
        else:
            stemmed_clean_news.append(word_tag[0])
            # print(word_tag)                                #Check pos taggers
    return (stemmed_clean_news)


# Create Frequency Dictionary
def create_freq_dict(news_list):
    freq_dict = dict()

    # add terms and frequency to dict
    for i in range(0, len(news_list)):
        news = news_list[i]
        for w in news.split():
            if w in freq_dict.keys():
                freq_dict[w][i] += 1
            else:
                freq = [0 for i in range(0, len(news_list))]
                freq_dict[w] = freq
                freq_dict[w][i] += 1
    return freq_dict


def filterDictionary(news_list, dictionary):
    del_word = set()

    # delete words that have freq < 2 in each of all news
    for key in dictionary.keys():
        ct = 0
        for i in range(0, len(news_list)):
            if dictionary[key][i] < 2:
                ct = ct + 1
        if ct == len(news_list):
            del_word.add(key)

    # delete words that have total freq < 5 in all news
    for key in dictionary.keys():
        count = 0
        for i in range(len(news_list)):
            count = count + dictionary[key][i]
        if count < 5:
            del_word.add(key)

    # delete words that have freq > 0 in only 1 news
    for key in dictionary.keys():
        for i in range(len(news_list)):
            if dictionary[key][i] > 0:
                count = 0
                for k in range(0, i):
                    count = count + dictionary[key][k]
                for k in range(i + 1, len(news_list)):
                    count = count + dictionary[key][k]
                if count == 0: del_word.add(key)

    for key in del_word:
        del dictionary[key]

    return dictionary


if __name__ == '__main__':
    #Open file
    #inpf = open("test.txt", encoding="utf8")
    inpf = open("newsCorpora_full_updated_small.txt")
    #inpf = open("test_2.txt")

    #List of news (each news = 1 string =  1 list element)
    news_coll = splitNews(inpf)
    news_list = []
    for news in news_coll:
        news_no_punct = removePunct(news)
        stemmed_news = " ".join(removeStopWord(news_no_punct))
        news_list.append(stemmed_news)
    # print(news_list)


    raw_matrix = create_freq_dict(news_list)
    filter_matrix = filterDictionary(news_list, raw_matrix)
    for k in filter_matrix.keys():
        print(k, filter_matrix[k])
