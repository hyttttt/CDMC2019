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
  for i in range(1, 4168):  #4168
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

  X = tv.fit_transform(X)
  X = X.toarray()
  X = np.array(X)

  Y = labels.label
  Y.ravel()

  report = test(X, Y)
  print('【TF-IDF】')
  print(report)
  print("")

def W2V_CBOW():
  docs = MyCorpus()
  labels = MyLabels()

  
  print("Word2Vec CBOW preparing")
  cbow_model = Word2Vec(sentences=docs, sg=0, vector_size=50, window=19, cbow_mean=1)
  docVec = []
  for doc in docs:
    vecs = np.array([cbow_model.wv[word] for word in doc if word in cbow_model.wv])
    docVec.append(np.mean(vecs, axis=0))
    
  report = test(docVec, labels.label)
  print('【W2V】【CBOW】')
  print(report)
  print("")
  

def W2V_SG():
  docs = MyCorpus()
  labels = MyLabels()

  print("Word2Vec SG preparing")
  sg_model = Word2Vec(sentences=docs, sg=1, vector_size=50, window=27)
  docVec = []
  for doc in docs:
    vecs = np.array([sg_model.wv[word] for word in doc if word in sg_model.wv])
    docVec.append(np.mean(vecs, axis=0))
  
  report = test(docVec, labels.label)
  print(f'【W2V】【SG】')
  print(report)
  print("")
      
    
#W2V_CBOW()
W2V_SG()
#TFIDF()