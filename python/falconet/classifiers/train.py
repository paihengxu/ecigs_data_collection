"""
Trains + evaluates classifiers to predict categorical label.  Takes care of model tuning (mostly).

Adrian Benton
1/9/2017
"""

import argparse, datetime, os, pickle, random, re, sys, time

from copy import deepcopy

import ujson as json

import numpy as np
import scipy.sparse as sp

import urllib.request
from zipfile import ZipFile

from sklearn import metrics
from sklearn.linear_model import LogisticRegression, SGDClassifier

from falconet import settings
# from falconet.twokenize import tokenizeRawTweetText as tokenize
from falconet.tokenizer import tokenize_raw_text as tokenize
from falconet.utils import Alphabet, connect_to_bitbucket
from falconet.io import open_file
from falconet.classifiers import classifier

SEED = 12345

# Klugey way to print to stdout and a log file
class Tee:
  def __init__(self, outPath=None):
    self.stdoutFile = sys.stdout
    
    if outPath is not None:
      self.outPath = outPath
  
  def write(self, *args):
    self.stdoutFile.write(*args)
    if self.outPath:
      self.outFile = open_file(self.outPath, 'at')
      self.outFile.write(*args)
      self.outFile.close()
  
  def flush(self):
    self.stdoutFile.flush()

# These are set when home directory is properly set by command line argument
HOME_DIR  = None
RSC_DIR   = None
DATA_DIR  = None
MODEL_DIR = None
LOG_DIR   = None

def initHome(homeDir):
  global HOME_DIR, RSC_DIR, DATA_DIR, MODEL_DIR, LOG_DIR
  
  HOME_DIR  = homeDir
  RSC_DIR   = os.path.join(HOME_DIR, 'resources')
  DATA_DIR  = os.path.join(HOME_DIR, 'data')
  MODEL_DIR = os.path.join(HOME_DIR, 'models')
  LOG_DIR   = os.path.join(HOME_DIR, 'logs')
  
  if not os.path.exists(HOME_DIR):
    os.mkdir(HOME_DIR)
  if not os.path.exists(RSC_DIR):
    os.mkdir(RSC_DIR)
  if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
  if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
  if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

def loadData(path, depvars, proptest=None):
  """
  Read data.  Each line contains a single JSON record with the tweet text, labels it's
  been assigned, and train/dev/test fold.  Missing labels are denoted with null values.
  If examples are not assigned to folds, then we train the model by cross-fold validation,
  otherwise we train it by tuning on dev set.
  
  Parameters
  ----------
  path : str
      path to data file
  depvars : [ str ]
      dependent variables to extract, extracts all dependent 
  proptest :  float
      if set, constructs test set by setting this proportion of examples as test
  
  Returns
  ----------
  trainDocs   :  [ str ]
      documents to train on, just text
  testDocs    : [ str ]
      documents to test on, text
  trainLabels : [ [ str ] ]
      labels for train set, one for each dependent variable
  testLabels  : [ [ str ] ]
      labels for test set
  folds       : numpy int vector
      fold each example is placed in
  tuningFolds : [ int ]
      which folds to evaluate on
  depvars     : [ str ] 
      all dependent variables if depvars was empty
  alphabets   : { str:{ str:int } } 
      dictionary of labels for each label type
  """

  random.seed(SEED)
  np.random.seed(SEED)
  
  labelAlphabets = {v:Alphabet() for v in depvars}
  trainDocs   = []
  testDocs    = []
  trainLabels = []
  testLabels  = []
  
  # keep track of label frequency
  labelCounts = [{} for v in depvars]
  
  # See if we need to split into 5 folds, or if they are given explicitly.
  # When numeric folds are given, assumes highest index fold is the test fold,

  
  allDepVars  = set()    # all dependent features in data
  hasDevFold  = False # tuning fold explicitly set
  testFoldIdx = -1
  f = open_file(os.path.join(DATA_DIR, path), 'rt')
  
  for ln in f:
    try:
      tweet = json.loads(ln)
    except ValueError:
      continue
    # keep track of all dependent variables
    allDepVars |= set([v for v in tweet['label'].keys()])
    
    if 'fold' in tweet and tweet['fold'] == 'dev':
      hasDevFold = True
    elif 'fold' in tweet and (type(tweet['fold'])==int or re.match('\d+', tweet['fold'])):
      testFoldIdx = max(testFoldIdx, int(tweet['fold']))
  
  f.close()
  
  if not depvars:
    depvars = sorted(list(allDepVars))
    labelCounts = [{} for v in depvars]
    labelAlphabets = {v: Alphabet() for v in depvars}

  if testFoldIdx > -1: # folds are already numbered, treat highest as test fold
    NUM_FOLDS = 1 + testFoldIdx
    tuningFolds = list(range(NUM_FOLDS - 1))
  elif not hasDevFold: # assign to folds myself
    NUM_FOLDS   = 5
    tuningFolds = list(range(NUM_FOLDS - 1))
    # TODO these are not being assigned!
  else:
    NUM_FOLDS   = 3
    tuningFolds = [1]
  testFoldIdx = NUM_FOLDS - 1
  folds = []
  
  f = open_file(os.path.join(DATA_DIR, path), 'rt')
  for ln in f:
    try:
      tweet = json.loads(ln)
    except ValueError:
      continue
    
    # read from tweet fields
    
    # make our own test set by rolling a die, if fold is not given
    if (proptest is not None) and ('fold' not in tweet):
      tweet['fold'] = 'test' if random.random() < proptest else 'train'
    
    fold = tweet['fold']
    labels = [tweet['label'][v] if (v in tweet['label']) and
                                   (tweet['label'][v] is not None) and
                                   ((type(tweet['label'][v]) != str) or
                                    tweet['label'][v].strip()) else None for v in depvars]
    
    if 'text' in tweet:
      text = tweet['text']
    else:
      text = tweet['tweet']['text'] # pull out text from the embedded tweet
    
    for alpha, label, counts in zip([labelAlphabets[v] for v in depvars], labels, labelCounts):
      if label != None:
        alpha.put(label)
        if label not in counts:
          counts[label] = 0
        counts[label] += 1
    
    if fold == 'train':
      trainDocs.append( text )
      trainLabels.append( labels )
      
      if hasDevFold: # train is fold 0, dev is 1, test is 2
        folds.append(0)
      else:  # TODO what if hasDevFold == False?? is this correct moving this into an else?
        folds.append(np.random.randint(0, NUM_FOLDS - 1))
    elif fold == 'dev': # we have an explicitly set dev fold
      trainDocs.append( text )
      trainLabels.append( labels )
      folds.append(1)  # TODO is this correct now if hasDevFold?
    elif fold == 'test':
      testDocs.append( text )
      testLabels.append( labels )
    elif type(tweet['fold'])==int or re.match('\d+', tweet['fold']):
      if int(tweet['fold']) == testFoldIdx:
        testDocs.append( text )
        testLabels.append( labels )
      else:
        trainDocs.append( text )
        trainLabels.append( labels )
        folds.append(int(tweet['fold']))
    else: # Should never hit this
      raise Exception('Example missing fold!', text, labels)
  
  f.close()
  
  alphabets = {v:alpha._wToI for v, alpha in labelAlphabets.items()}
  
  # make the class with the most examples be the negative one.  May want to change this
  # eventually to let user set positive class.
  for counts, v in zip(labelCounts, depvars):
    majWord    = max([(c, w) for w, c in counts.items()])[1]
    oldNegWord = ([w for w in alphabets[v] if alphabets[v][w]==0])[0]
    alphabets[v][majWord], alphabets[v][oldNegWord] = alphabets[v][oldNegWord], alphabets[v][majWord]

  return trainDocs, testDocs, trainLabels, testLabels, folds, tuningFolds, depvars, alphabets


def downloadData(key, secret, dataPath=None):
  """
  Pulls training data from bitbucket repo.  Need to pass OAuth key and secret
  and paste the redirect URL to download file automatically.  If dataPath is
  not set, then downloads all training sets.
  """
  
  bitbucket, base_uri = connect_to_bitbucket(key, secret)
  
  if dataPath is None: # Download all files, but not those we have already have locally
    resp = bitbucket.get(base_uri)
    downloads = json.loads(resp.content.decode('ascii'))
    dataUris = [obj['links']['self']['href'] for obj in downloads['values'] if obj['name'].endswith('.json')
                                                                            or obj['name'].endswith('.json.gz')]
    
    for dataUri in dataUris:
      dataPath = os.path.basename(dataUri)
      resp     = bitbucket.get(dataUri)
      fileBits = resp.content
      
      outFile = open(os.path.join(DATA_DIR, dataPath), 'wb')
      outFile.write(fileBits)
      outFile.close()
  
  else: # Just download one
    if os.path.exists(os.path.join(DATA_DIR, dataPath)):
      print ('Already have %s, skipping download' % (dataPath))
    else:
      resp = bitbucket.get(base_uri + dataPath)
      fileBits = resp.content
      outFile = open(os.path.join(DATA_DIR, dataPath), 'wb')
      outFile.write(fileBits)
      outFile.close()

def downloadEmbeddings(embeddingUrl=settings.EMBEDDINGS_URL):
  """ Download and unzip GloVE embeddings from web. """
  
  print('Downloading Twitter embeddings from %s, this may take a while. . .' % (embeddingUrl))
  
  savedPath = os.path.join(RSC_DIR, "glove.twitter.27B.zip")
  
  resp = urllib.request.urlopen(embeddingUrl)
  outFile = open(savedPath, 'wb')
  outFile.write(resp.read())
  outFile.close()
  
  # Re-open the newly-created file with ZipFile()
  zf = ZipFile(savedPath)
  zf.extractall(path = RSC_DIR)
  zf.close()
  
  # clean up zip file
  os.remove(savedPath)

def initSGD(loss='hinge', l1=1.0, l2=1.0, n_iter=100, n_jobs=1, class_weight=None):
  """ Initializes a linear classifier. """
  return SGDClassifier(loss=loss, penalty='elasticnet', alpha=(l1+l2),
                       l1_ratio=l1/(l1+l2), fit_intercept=True,
                       n_iter=n_iter, shuffle=True, verbose=0,
                       n_jobs=n_jobs, random_state=None,
                       learning_rate='optimal',
                       class_weight=class_weight, warm_start=False, average=10)

def initLogReg(loss='log', l1=1.0, l2=1.0, n_iter=100, n_jobs=1, class_weight=None):
  """ Initializes a logistic regression classifier, ignores loss and L1 penalty. """
  return LogisticRegression(penalty='l2', tol=0.0001, C=l2,
                            solver='liblinear', max_iter=n_iter,
                            multi_class='ovr', n_jobs=n_jobs,
                            class_weight=class_weight)

def sweep(docs, y_gold, y_dictionary, folds, tuningFolds, depVar, beta=1.0, maxPropDropped=0.5,
          n_jobs=1, featureSettings=(None, None, None), regSettings=(None, None),
          balance=False):
  """
  Sweep over regularization constants to maximize dev f-score, tune by cross-fold validation.
  
  Parameters
  ----------
  docs : [ [ str ] ]
      tokenized tweet corpus
  y_gold : [ str ]
      label for each tweet
  y_dictionary : { str:int } 
      mapping from label to index
  folds : numpy int vector
      which fold each example belongs to
  tuningFolds : [ int ]
      which folds to tune on
  depVar : str
      dependent variable we extracted
  ngramRange : (int, int)
      min and max ngram order
  embeddingPath: str
      path to word embeddings
  beta : float
      weighting of precision/recall in f-score
  maxPropDropped : float
      maximum proportion of examples allowed dropped by threshold
  featureSettings : (int, int, int)
      tuple of (MIN_NGRAM_ORDER, MAX_NGRAM_ORDER, EMBEDDING_DIM), where None means we sweep over this
  regSettings : (float, float)
      (L1, L2), we don't actually use L1 regularization at the moment, but may in the future
  n_jobs : int
      number of cores for training
  balance : bool
      should example weights be balance inversely proportional to class prevalance 
  Returns
  ----------
  bestClassifier : TextTokenClassifier
      highest f-score classifier
  accs    : [ (float, (...)) ]
      accuracy along with model hyperparameters
  fscores : [ (float, (...)) ]
      f-score along with hyperparameters
  """
  accs    = []
  fscores = []  
  n_iter = 100 # TODO: possibly pull this out as a parameter
  
  bestClassifier = None
  bestFscore = -1.
  class_weight = None if not balance else 'balanced'
  
  # model = initSGD('log', 0.0, 1.0, n_iter, n_jobs, class_weight=class_weight)
  model = initLogReg('log', 0.0, 1.0, n_iter, n_jobs, class_weight=class_weight)
  
  # Default parameter sweeps
  l1Sweep_dflt      = [0.0]
  l2Sweep_dflt      = [1.e-6, 1.e-5, 1.e-3, 1.e-2, 1.e-1, 1.0, 5.0, 10.0]
  ngramSweep_dflt   = [(-1, -1), (1, 2), (1, 3)]
  embeddingDim_dflt = [-1, 50, 100, 200]

  l1Sweep        = l1Sweep_dflt if regSettings[0] is None else [regSettings[0]]
  l2Sweep        = l2Sweep_dflt if regSettings[1] is None else [regSettings[1]]
  ngramSweep     = ngramSweep_dflt if (featureSettings[0] is None) or (featureSettings[1] is None) \
                   else [(featureSettings[0], featureSettings[1])]
  embeddingSweep = embeddingDim_dflt if featureSettings[2] is None else [featureSettings[2]]  
  loss = 'log'  
  # sweep over different possible feature sets
  for ngramRange in ngramSweep:
    for embeddingDim in embeddingSweep:      
      if embeddingDim != -1:
        embeddingPath = os.path.join(RSC_DIR, 'glove.twitter.27B.%dd.txt' % (embeddingDim))
      else:
        embeddingPath = None
      
      # init a classifier and tokenize the data -- avoids needless work 
      classifierModel, acc, fscore = classifier.TextTokenClassifier.fit(docs, y_gold,
                                                      y_dictionary, folds, tuningFolds,
                                                      model, ngramRange,
                                                      embeddingPath=embeddingPath, beta=beta,
                                                      maxPropDropped=maxPropDropped,
                                                      name=depVar)
      
      #tokens = [tokenize(d.lower()) for d in docs]
      
      X = classifierModel.extract(docs, pretokenized=False)
      y = np.asarray([y_dictionary[value] if value in y_dictionary else -1 for value in y_gold])
      
      for l2 in l2Sweep:
        for l1 in l1Sweep:
          params = (loss, l1, l2, beta, ngramRange, embeddingDim)
          
          print('---- "%s" loss=%s, ngrams=%s, embeddingDim=%s, l1=%e, l2=%e, beta=%s ----' %
                (depVar, loss, str(ngramRange), embeddingDim, l1, l2, beta))
          
          model = initLogReg(loss, l1, l2, n_iter, n_jobs, class_weight=class_weight)
          acc, fscore = classifierModel.fitNewData(X, y, folds, tuningFolds, model,
                                                   beta, maxPropDropped)
          
          if ((fscore > bestFscore) or
              ((fscore >= bestFscore) and
               (classifierModel._threshold < bestClassifier._threshold))):
            bestClassifier = deepcopy(classifierModel)
            bestClassifier._name = depVar
            bestFscore = fscore
            bestParams = params
          
          accs.append((acc, params))
          fscores.append((fscore, params))
          
          sys.stdout.flush()
  
  return bestClassifier, accs, fscores, bestParams

def trainModel(DATA, folds, tuningFolds, y_dictionary, depVar, modelPrefix, beta,
               maxPropDropped, nJobs, dataPath='', featureSettings=(None, None, None), 
               regSettings=(None, None), propTest=0.2, balance=False):
  """
  Trains and evaluates a classifier by sweeping over feature set and regularization parameters.
  Does not sweep over any parameters that are passed on command line.
  """
  random.seed(SEED)
  np.random.seed(SEED)  
  # Train a classifier
  trainDocs, trainLabels = DATA[0]  
  classifierModel, accs, fscores, bestParams = sweep(trainDocs, trainLabels, y_dictionary,
                                                     folds, tuningFolds, depVar,
                                                     beta=beta,
                                                     maxPropDropped=maxPropDropped,
                                                     n_jobs=nJobs,
                                                     featureSettings=featureSettings,
                                                     regSettings=regSettings,
                                                     balance=balance)
  
  # Evaluate classifier on test
  testDocs, testLabels = DATA[1]  
  predLabels = classifierModel.predict(testDocs, pretokenized=False)  
  predLabelInts = []
  for label in predLabels:
    if label == 'None':
      predLabelInts.append(-1)
    elif label in y_dictionary:
      predLabelInts.append(y_dictionary[label])
    else:
      raise Exception('Cannot recognize predicted label: %s' % (label))
  predInts = np.asarray(predLabelInts)
  
  testLabelInts = []
  for label in testLabels:
    if label in y_dictionary:
      testLabelInts.append(y_dictionary[label])
    elif label is None:
      testLabelInts.append(-1)
    else:
      raise Exception('Unrecognized label:', label)
  testInts = np.asarray(testLabelInts)
  
  filteredPred = predInts[(predInts!=-1)&(testInts!=-1)]
  filteredTest = testInts[(predInts!=-1)&(testInts!=-1)]
  
  print('%d/%d test examples excluded' % (((predInts==-1)&(testInts!=-1)).sum(),
                                          (testInts!=-1).sum()))
  
  droppedTest = testInts[(predInts==-1)&(testInts!=-1)]
  
  # get an idea of distribution of example labels vs. those we drop  
  keptDistStr    = '  '.join(['%s:%d' % (name, (filteredTest==intLabel).sum())
                              for intLabel, name in sorted(classifierModel._labelDict.items())])
  droppedDistStr = '  '.join(['%s:%d' % (name, (droppedTest==intLabel).sum())
                              for intLabel, name in sorted(classifierModel._labelDict.items())])
  
  print('Test kept label distribution:', keptDistStr)
  print('Test dropped label distribution:', droppedDistStr)  
  
  avgFscore = 'macro' if len(classifierModel._labelDict) > 2 else 'binary'
  testAcc = metrics.accuracy_score(filteredTest, filteredPred)
  testFscore = metrics.f1_score(filteredTest, filteredPred, average=avgFscore)
  
  print('Test (loss=%s, l1=%.4f, l2=%.4f, beta=%.2f, ngram=%s, embedding=%s): accuracy %.4f, f1-macro %.4f' % (
    *bestParams,
    testAcc, testFscore))

  print('Label dictionary:%s' % {v: k for k, v in y_dictionary.items()})
  print('---- Test confusion matrix ----')
  testConfMax = metrics.confusion_matrix(filteredTest, filteredPred)

  print(str(testConfMax))
  
  # Save best model along with test performance, for later inspection
  classifierModel._testFscore  = testFscore
  classifierModel._testAcc     = testAcc
  classifierModel._testConfMax = testConfMax
  classifierModel._testKeptLabelDist    = keptDistStr # examples with high confidence
  classifierModel._testDroppedLabelDist = droppedDistStr # examples where we have low confidence, skipped tagging
  
  # contains all arguments to retrain this model
  classifierModel._trainArgs = {'dataPath':dataPath,
                                'maxpropdropped':maxPropDropped, 'l1':bestParams[1], 
                                'l2':bestParams[2],
                                'minngram':bestParams[-2][0], 'maxngram':bestParams[-2][1],
                                'embeddingdim':bestParams[-1], 'beta':beta, 'depvar':depVar,
                                'prefix':os.path.basename(modelPrefix), 'proptest':propTest,
                                'balance':balance}
  
  serPath = '%s_DEP-%s_DROPPED-%s_NGRAM-(%s,%s)_BETA-%s_EMB-%s.pickle' % (modelPrefix,
                                                                          depVar,
                                                                          maxPropDropped,
                                                                          bestParams[-2][0],
                                                                          bestParams[-2][1],
                                                                          beta,
                                                                          bestParams[-1])
  
  classifierModel.serialize(serPath)
  
  # make sure we can load model again
  classifierModel = classifier.TextTokenClassifier.deserialize(serPath)
