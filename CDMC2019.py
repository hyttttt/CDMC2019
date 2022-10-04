from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from gensim.models.word2vec import Word2Vec
from gensim import utils
import matplotlib as plt

class MyCorpus:
  '''An iterator that yields lists of str'''

  docs = []
  for i in range(1, 2):  #4168
    print(f'\rData collecting {int(i / 4168 * 100)}%', end="")

    i = "%04d" % i
    file_dir = r'C:\Users\88696\Desktop\NTUST\專題\CDMC2019\CDMC2019Task2Train\TRAIN\\' 
    file_dir += str(i)
    file_dir += '.seq'

    with open(file_dir) as f:
      doc = []

      for line in f:
        doc += utils.simple_preprocess(line)

      docs.append(doc)

  print(docs)
  print(f'\rData collecting 100%')

  def __iter__(self):
    for i in self.docs:
      yield i   

class MyLabels:
  label = pd.read_csv(r"C:\Users\88696\Desktop\NTUST\專題\CDMC2019\CDMC2019Task2Train.csv").drop(["no"], axis=1) 
  label = label.to_numpy()
  label = np.ravel(label)
  print(f'Label collecting 100%')


  def __iter__(self):
    for i in self.label:
      yield i    

def test(X, Y):
  print("Report preparing")

  report = pd.DataFrame({
  'model' : [],
  'accuracy' : []
  })

  ################### random forest ###################
  rf = ensemble.RandomForestClassifier(n_estimators = 100)
  score = cross_val_score(rf, X, Y, cv=10, scoring='accuracy')

  df = pd.DataFrame({
  'model' : ['RANDOM FOREST'],
  'accuracy' : [score.mean()]
  })
  report = pd.concat([report, df], ignore_index=True)
  #####################################################

  ################### KNN ###################
  knn = KNeighborsClassifier()
  score = cross_val_score(knn, X, Y, cv=10, scoring='accuracy')

  df = pd.DataFrame({
  'model' : ['KNN'],
  'accuracy' : [score.mean()]
  })
  report = pd.concat([report, df], ignore_index=True)
  ###########################################

  ################### SVM Linear ###################
  svm_linear = SVC(kernel='linear', gamma='auto')
  score = cross_val_score(svm_linear, X, Y, cv=10, scoring='accuracy')

  df = pd.DataFrame({
  'model' : ['SVM Linear'],
  'accuracy' : [score.mean()]
  })
  report = pd.concat([report, df], ignore_index=True)
  ##################################################

  ################### SVM Poly ###################
  svm_poly = SVC(kernel='poly', gamma='scale')
  score = cross_val_score(svm_poly, X, Y, cv=10, scoring='accuracy')

  df = pd.DataFrame({
  'model' : ['SVM Poly'],
  'accuracy' : [score.mean()]
  })
  report = pd.concat([report, df], ignore_index=True)
  ################################################

  ################### SVM RBF ###################
  svm_rbf = SVC(kernel='rbf', gamma='scale')
  score = cross_val_score(svm_rbf, X, Y, cv=10, scoring='accuracy')

  df = pd.DataFrame({
  'model' : ['SVM RBF'],
  'accuracy' : [score.mean()]
  })
  report = pd.concat([report, df], ignore_index=True)
  ###############################################

  return report

def TFIDF():
  docs = MyCorpus()
  labels = MyLabels()
  tv = TfidfVectorizer()
  print("TF-IDF preparing")

  X = []
  for doc in docs:
    sentence = ''
    for word in doc:
      sentence += word + ' '

    X.append(sentence)

  tv.fit_transform(X)
  X = tv.fit_transform(X)
  X = X.toarray()
  X = np.array(X)

  max_idf = max(tv.idf_)
  idf_weight = defaultdict(lambda: max_idf, [(word, tv.idf_[i]) for word, i in tv.vocabulary_.items()])

  Y = labels.label

  report = test(X, Y)
  print('【TF-IDF】')
  print(report)
  print("")

  return idf_weight

def one():
  return 1.0

def W2V(sg=0, vector_size=100, window=5, cbow_mean=1, tfidf_weight=defaultdict(one)):
  docs = MyCorpus()
  labels = MyLabels()

  if sg==0:
    type = "CBOW"
  else:
    type = "SG"

  if tfidf_weight['a random word'] == 1.0:
    docVec_type = "mean"
  else:
    docVec_type = "tfidf_weighted"

  print(f"Word2Vec {type} preparing")
  cbow_model = Word2Vec(sentences=docs, sg=sg, vector_size=vector_size, window=window, cbow_mean=cbow_mean)
  docVec = []
  for doc in docs:
    vecs = np.array([cbow_model.wv[word] * tfidf_weight[word] for word in doc if word in cbow_model.wv])
    docVec.append(np.mean(vecs, axis=0))
    
  report = test(docVec, labels.label)
  print(f'【W2V】【{type}】【{docVec_type}】')
  print(report)
  print("")      

tfidf_weight = TFIDF()

W2V(sg=0, vector_size=50, window=19)
W2V(sg=0, vector_size=50, window=19, tfidf_weight=tfidf_weight)

W2V(sg=1, vector_size=50, window=27)
W2V(sg=1, vector_size=50, window=27, tfidf_weight=tfidf_weight)
