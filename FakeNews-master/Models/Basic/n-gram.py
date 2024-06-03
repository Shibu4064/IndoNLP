import os
import pandas as pd
import re
import numpy as np
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from scipy import sparse, hstack, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import string
import pickle
import load_feature
import argparse

import sys
sys.path.insert(1, '../Helper/')
import helper, config


def load_data(path):
    data_df = pd.read_csv(path)
    return data_df



def mp(args):

    model(load_feature.mp(X),Y,"MP")


def pos(args):

    model(load_feature.pos(),Y,"POS")



#Unigram
def unigram(args):
    X_train1 = load_feature.tfidf_wordF(X_train, X_train, 1, 1)
    X_test1 = load_feature.tfidf_wordF(X_train, X_test, 1, 1)
    model(X_train1, y_train, X_test1, y_test, "Unigram")


#Bigram
def bigram(args):
    X_train1 = load_feature.tfidf_wordF(X_train, X_train, 2, 2)
    X_test1 = load_feature.tfidf_wordF(X_train, X_test, 2, 2)
    model(X_train1, y_train, X_test1, y_test, "Bigram")

#Trigram
def trigram(args):
    X_train1 = load_feature.tfidf_wordF(X_train, X_train, 3, 3)
    X_test1 = load_feature.tfidf_wordF(X_train, X_test, 3, 3)
    model(X_train1, y_train, X_test1, y_test, "Trigram")


#U+B+T
def u_b_t(args):
    X_train1 = load_feature.tfidf_wordF(X_train, X_train, 1, 3)
    X_test1 = load_feature.tfidf_wordF(X_train, X_test, 1, 3)
    model(X_train1, y_train, X_test1, y_test, "U+B+T")


#C3
def char_3(args):
    X_train1 = load_feature.tfidf_charF(X_train, X_train, 3, 3)
    X_test1 = load_feature.tfidf_charF(X_train, X_test, 3, 3)
    model(X_train1, y_train, X_test1, y_test, "C3-gram")

#c4
def char_4(args):
    X_train1 = load_feature.tfidf_charF(X_train, X_train, 4, 4)
    X_test1 = load_feature.tfidf_charF(X_train, X_test, 4, 4)
    model(X_train1, y_train, X_test1, y_test, "C4-gram")

#c5
def char_5(args):
    X_train1 = load_feature.tfidf_charF(X_train, X_train, 5, 5)
    X_test1 = load_feature.tfidf_charF(X_train, X_test, 5, 5)
    model(X_train1, y_train, X_test1, y_test, "C5-gram")

#c3+c4+c5
def char_3_4_5(args):
    X_train1 = load_feature.tfidf_charF(X_train, X_train, 3, 5)
    X_test1 = load_feature.tfidf_charF(X_train, X_test, 3, 5)
    model(X_train1, y_train, X_test1, y_test, "C3+C4+C5")

#Linguistic
def lexical(args):

    X_char = load_feature.tfidf_charF(X, 3, 5)
    X_word = load_feature.tfidf_wordF(X, 1, 3)
    model(sparse.hstack((X_word, X_char)),Y,"Lexical")


#Word Embedding Fasttext
def word_300(args):
    model(load_feature.word_emb(300,X),Y,"Emb_F")


#Word Embedding News
def word_100(args):

    model(load_feature.word_emb(100,X),Y,"Emb_N")

#L+POS
def L_POS(args):

    model(sparse.hstack((load_feature.tfidf_charF(X, 3, 5), load_feature.tfidf_wordF(X, 1, 3), load_feature.pos())),Y,"L+POS")


#L+POS+Emb(F)
def L_POS_Emb_F(args):

    model(sparse.hstack((load_feature.tfidf_charF(X, 3, 5), load_feature.tfidf_wordF(X, 1, 3), load_feature.pos(), load_feature.word_emb(300,X))),Y,"L+POS+Emb(F)")

#L+POS+Emb(N)
def L_POS_Emb_N(args):
    
    model(sparse.hstack((load_feature.tfidf_charF(X, 3, 5), load_feature.tfidf_wordF(X, 1, 3), load_feature.pos(), load_feature.word_emb(100,X))),Y,"L+POS+Emb(N)")


#L+POS+E(F)+MP
def L_POS_Emb_F_MP(args):
    
    model(sparse.hstack((load_feature.tfidf_charF(X, 3, 5), load_feature.tfidf_wordF(X, 1, 3), load_feature.pos(), load_feature.word_emb(300,X), load_feature.mp(X))),Y,"L+POS+E(F)+MP")


#L+POS+E(N)+MP
def L_POS_Emb_N_MP(args):

    model(sparse.hstack((load_feature.tfidf_charF(X, 3, 5), load_feature.tfidf_wordF(X, 1, 3), load_feature.pos(), load_feature.word_emb(100,X), load_feature.mp(X))),Y,"L+POS+E(N)+MP")


#Allfeatures
def allfeatures(args):

    model(sparse.hstack((load_feature.tfidf_charF(X, 3, 5), load_feature.tfidf_wordF(X, 1, 3), load_feature.pos(), load_feature.word_emb(300,X), load_feature.word_emb(100,X), load_feature.mp(X))),Y,"Allfeatures")



def model(X_train, y_train, X_test, y_test, exp):
    if classifier == "SVM":
        clf = svm.SVC(kernel='linear', C=10, cache_size=7000)
    elif classifier == "LR":
        clf = LogisticRegression()
    
    elif classifier == "RF":
        class_weight = dict({1:1,0:25})
        clf = RandomForestClassifier(bootstrap=True,
                class_weight=class_weight,
                    criterion='gini',
                    max_depth=None, max_features='auto', max_leaf_nodes=None,
                    min_impurity_decrease=0.0, min_impurity_split=None,
                    min_samples_leaf=4, min_samples_split=10,
                    min_weight_fraction_leaf=0.0, n_estimators=300,
                    oob_score=False,
                    random_state=0,
                    verbose=0, warm_start=False)

    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    t, f, o, om = helper.getResult(y_test, y_pred)
    print("                                Overall               #               Fake                ")
    print("                   precision    recall      f1-score  #  precision    recall      f1-score")
    res = helper.printResult(exp,o,f)
    print(res)

    print("                                Macro               #               Fake                ")
    print("                   precision    recall      f1-score  #  precision    recall      f1-score")
    res = helper.printResult(exp,om,f)
    print(res)

    if save:
        path = args.model+"_results.txt"
        helper.saveResults(path, res)
    #Save Model
    outfile = open(config.API+args.model+'_'+exp+'.pkl', 'wb')
    pickle.dump(clf, outfile)
    outfile.close()



parser = argparse.ArgumentParser(description='Argparse!')
subparsers = parser.add_subparsers()

parser_p = subparsers.add_parser('Unigram')
parser_p.set_defaults(func=unigram)

parser_q = subparsers.add_parser('Bigram')
parser_q.set_defaults(func=bigram)

parser_p = subparsers.add_parser('Trigram')
parser_p.set_defaults(func=trigram)

parser_q = subparsers.add_parser('U+B+T')
parser_q.set_defaults(func=u_b_t)


parser_p = subparsers.add_parser('C3-gram')
parser_p.set_defaults(func=char_3)

parser_q = subparsers.add_parser('C4-gram')
parser_q.set_defaults(func=char_4)

parser_p = subparsers.add_parser('C5-gram')
parser_p.set_defaults(func=char_5)

parser_q = subparsers.add_parser('C3+C4+C5')
parser_q.set_defaults(func=char_3_4_5)


parser_p = subparsers.add_parser('Lexical')
parser_p.set_defaults(func=lexical)

parser_q = subparsers.add_parser('POS')
parser_q.set_defaults(func=pos)

parser_q = subparsers.add_parser('L_POS')
parser_q.set_defaults(func=L_POS)

parser_p = subparsers.add_parser('Emb_F')
parser_p.set_defaults(func=word_300)

parser_q = subparsers.add_parser('Emb_N')
parser_q.set_defaults(func=word_100)


parser_p = subparsers.add_parser('L+POS+E_F')
parser_p.set_defaults(func=L_POS_Emb_F)

parser_q = subparsers.add_parser('L+POS+E_N')
parser_q.set_defaults(func=L_POS_Emb_N)

parser_p = subparsers.add_parser('MP')
parser_p.set_defaults(func=mp)

parser_q = subparsers.add_parser('L+POS+E_F+MP')
parser_q.set_defaults(func=L_POS_Emb_F_MP)

parser_q = subparsers.add_parser('L+POS+E_N+MP')
parser_q.set_defaults(func=L_POS_Emb_N_MP)


parser_p = subparsers.add_parser('all_features')
parser_p.set_defaults(func=allfeatures)

parser.add_argument("model")
parser.add_argument("-s","--save", action="store_true")
args = parser.parse_args()
classifier = args.model
save = args.save
train_path = '../../../dataset/train_datasetClean.csv'
val_path = '../../../dataset/val_datasetClean.csv'
train = load_data(train_path)
val = load_data(val_path)
X_train = train.iloc[:, 0:2]
y_train = train.iloc[:, 2].values.ravel()
X_test = val.iloc[:, 0:2]
y_test = val.iloc[:, 2].values.ravel()
args.func(args)









