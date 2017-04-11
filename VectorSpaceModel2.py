import nltk
import re
from nltk.data import load
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.stem import SnowballStemmer


import pickle
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier

"""******************************************************************************************************************
========================================CREATE A TERM-BY-DOCUMENT MATRIX ============================================
******************************************************************************************************************"""


# Turn document into a list of news:
'''
def splitNews(file):
    full_text = file.read()
    split_text = full_text.split(">>>>")[1:]
    b = split_text[:250]
    t = split_text[250:500]
    e = split_text[500:750]
    m = split_text[750:]
    label = []
    for news in b:
        label.append('b')
    for news in t:
        label.append('t')
    for news in e:
        label.append('e')
    for news in m:
        label.append('m')
    return [split_text, label]
'''
def splitNews(file):
    full_text = file.read()
    split_text = full_text.split(">>>>")[1:]
    b = split_text[:250]
    t = split_text[250:500]
    e = split_text[500:750]
    m = split_text[750:]
    label = []
    split_text_new = []
    for i in range(0, 250):
        split_text_new.append(b[i])
        split_text_new.append(t[i])
        split_text_new.append(e[i])
        split_text_new.append(m[i])
    for i in range(0, len(split_text_new)):
        if i % 4 == 0:
            label.append('b')
        elif i % 4 == 1:
            label.append('t')
        elif i % 4 == 2:
            label.append('e')
        elif i % 4 == 3:
            label.append('m')
    print(len(split_text_new))
    print(len(label))
    return [split_text_new, label]

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
    '''
    stemmer = SnowballStemmer("english")
    for word_tag in wordtag_clean_news:
        stemmed_clean_news.append(stemmer.stem(word_tag[0]))
    '''
    # for word in stemmed_clean_news:
        # print(word)
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


# Take a list of documents, create tfidf and lsa matrices
def vectorize(X_train_raw):
    # Tfidf vectorizer:
    #   - Strips out “stop words”
    #   - Filters out terms that occur in more than half of the docs (max_df=0.5)
    #   - Filters out terms that occur in only one document (min_df=2).
    #   - Selects the 10,000 most frequently occuring words in the corpus.
    #   - Normalizes the vector (L2 norm of 1.0) to normalize the effect of
    #     document length on the tf-idf values.
    vectorizer = TfidfVectorizer(max_df=0.5, max_features= None,
                             min_df=2, stop_words='english',
                             use_idf=True)

    # Build the tfidf vectorizer from the training data ("fit"), and apply it
    # ("transform").
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)

    print("  Actual number of tfidf features: %d" % X_train_tfidf.get_shape()[1])


    # feature_names = vectorizer.get_feature_names()
    # dense = X_train_tfidf.todense()
    # print(len(dense))
    # episode = dense[0].tolist()[0]
    # phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
    # print(len(phrase_scores))
    # print(sorted(phrase_scores, key=lambda t: t[1] * -1)[:5])
    # sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
    # for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:20]:
    #     print('{0: <20} {1}'.format(phrase, score))


    print("\nPerforming dimensionality reduction using LSA")
    t0 = time.time()

    # Project the tfidf vectors onto the first 150 principal components.
    # Though this is significantly fewer features than the original tfidf vector,
    # they are stronger features, and the accuracy is higher.
    svd = TruncatedSVD(100)
    lsa = make_pipeline(svd, Normalizer(copy=False))

    # Run SVD on the training data, then project the training data.
    X_train_lsa = lsa.fit_transform(X_train_tfidf)

    print("  done in %.3fsec" % (time.time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

    return (X_train_tfidf,X_train_lsa)



if __name__ == '__main__':

    ###############################################################################
    #  Load the raw text dataset.
    ###############################################################################

    print("Loading dataset...")


    # Open file
    inpf = open("shortlist.txt", encoding="utf8")

    # Split input file into a list of articles and a list of labels
    result = splitNews(inpf)
    raw_data = result[0]
    all_labels = result[1]

    # Preprocess news(remove stop words + stem)
    # and put news into news_list
    all_news = [] # list of string (news)
    for news in raw_data:
        news_no_punct = removePunct(news)
        stemmed_news = " ".join(removeStopWord(news_no_punct))
        all_news.append(stemmed_news)


    # NOTE: up till this point
    # We should have all articles in all_news
    # with their corresponding labels in all_labels


    # Split data into train and test data
    X_train_raw = [news for news in all_news[0:900]]    # train news, X % of all_news
    y_train = [l for l in all_labels[0:900]]            # train labels, corresponding X % of all_labels
    X_test_raw = [news for news in all_news[900:]]      # test news, (100-X) % of all_news
    y_test = [l for l in all_labels[900:]]              # test labels, corresponding (100-X) % of all_labels


    #print("  %d training examples (%d positive)" % (len(y_train), sum(y_train)))
    #print("  %d test examples (%d positive)" % (len(y_test), sum(y_test)))

    
    ###############################################################################
    #  Use LSA to vectorize the articles.
    ###############################################################################

    # # Apply transformations to the train data
    # (X_train_tfidf, X_train_lsa) = vectorize(X_train_raw)
    # print(X_train_tfidf.get_shape())
    # # Apply the transformations to the test data too
    # (X_test_tfidf, X_test_lsa) = vectorize(X_test_raw)
    # print(X_test_tfidf.get_shape())

    # Tfidf vectorizer:
	#   - Strips out “stop words”
	#   - Filters out terms that occur in more than half of the docs (max_df=0.5)
	#   - Filters out terms that occur in only one document (min_df=2).
	#   - Selects the 10,000 most frequently occuring words in the corpus.
	#   - Normalizes the vector (L2 norm of 1.0) to normalize the effect of 
	#     document length on the tf-idf values. 
    vectorizer = TfidfVectorizer(max_df=0.5, max_features= 10000,
								min_df=2, stop_words='english',
								use_idf=True)

	# Build the tfidf vectorizer from the training data ("fit"), and apply it 
	# ("transform").
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    print(X_train_tfidf.get_shape())

    print("  Actual number of tfidf features: %d" % X_train_tfidf.get_shape()[1])

    print("\nPerforming dimensionality reduction using LSA")
    t0 = time.time()

	# Project the tfidf vectors onto the first 150 principal components.
	# Though this is significantly fewer features than the original tfidf vector,
	# they are stronger features, and the accuracy is higher.
    svd = TruncatedSVD(100)
    lsa = make_pipeline(svd, Normalizer(copy=False))

	# Run SVD on the training data, then project the training data.
    X_train_lsa = lsa.fit_transform(X_train_tfidf)


    print("  done in %.3fsec" % (time.time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))


	# Now apply the transformations to the test data as well.
    X_test_tfidf = vectorizer.transform(X_test_raw)
    print(X_test_tfidf.get_shape())
    X_test_lsa = lsa.transform(X_test_tfidf)

    ###############################################################################
    #  Run classification of the test articles
    ###############################################################################
    # NOTE: The following section is similar to the example code from this link
    # https://github.com/chrisjmccormick/LSA_Classification/blob/master/runClassification_LSA.py


    print("\nClassifying tfidf vectors...")

    # Time this step.
    t0 = time.time()

    # Build a k-NN classifier. Use k = 5 (majority wins), the cosine distance, 
    # and brute-force calculation of distances.
    knn_tfidf = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine')
    knn_tfidf.fit(X_train_tfidf, y_train)                           # =========================> TRAINING

    # Classify the test vectors.
    # output = list of labels
    p = knn_tfidf.predict(X_test_tfidf)                             # =========================> PREDICTING


    # Measure accuracy
    numRight = 0
    for i in range(0,len(p)):
        if p[i] == y_test[i]: # compare predicted label with given label of testing data
            numRight += 1

    print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), float(numRight) / float(len(y_test)) * 100.0))

    # Calculate the elapsed time (in seconds)
    elapsed = (time.time() - t0)
    print("  done in %.3fsec" % elapsed)


    print("\nClassifying LSA vectors...")

    # Time this step.
    t0 = time.time()

    # Build a k-NN classifier. Use k = 5 (majority wins), the cosine distance, 
    # and brute-force calculation of distances.
    knn_lsa = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine')
    knn_lsa.fit(X_train_lsa, y_train)

    # Classify the test vectors.
    p = knn_lsa.predict(X_test_lsa)

    # Measure accuracy
    numRight = 0
    for i in range(0,len(p)):
        if p[i] == y_test[i]:
            numRight += 1

    print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), float(numRight) / float(len(y_test)) * 100.0))

    # Calculate the elapsed time (in seconds)
    elapsed = (time.time() - t0)    
    print("    done in %.3fsec" % elapsed)

