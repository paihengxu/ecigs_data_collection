"""
Classifier to annotate tweets by the tokens in the document.

Adrian Benton
1/9/2016
"""

import os, pickle, sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from functools import reduce
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
# from falconet.twokenize import tokenizeRawTweetText as tokenize
from falconet.tokenizer import tokenize_raw_text as tokenize
from falconet.io import open_file

# TODO: NLTK English stopwords.  Find a better stopword list for Twitter twokenizer
STOPWORDS = {'yours', 'until', 'll', 'm', 'again', 'further', 'an', 'against', 'were', 'the',
             'am', 'being', 'y', 'was', 'during', 'it', 'ma', 'my', 'before', 'and', 'hers',
             'which', 'not', 'ain', 'after', 'same', 'myself', 'into', 'himself', 'there',
             'hasn', 'how', 'themselves', 'is', 'its', 'needn', 'he', 'once', 'o', 'most',
             'will', 'while', 't', 'all', 'very', 'at', 'than', 'yourselves', 'then',
             'shouldn', 'does', 'over', 'each', 'don', 'your', 'his', 'herself', 'between',
             'should', 'yourself', 'we', 'won', 'here', 'have', 's', 'where', 'haven', 'above',
             'couldn', 'isn', 'they', 'few', 'she', 'below', 'these', 'or', 'theirs', 'on', 'd',
             'with', 'down', 'be', 'from', 'just', 've', 'doing', 'do', 'you', 'when', 'own',
             'such', 'their', 'as', 'to', 'so', 'only', 'had', 'has', 'aren', 'why', 'what',
             'her', 'any', 'out', 'been', 'a', 'more', 'mightn', 'other', 'weren', 'didn',
             'him', 'our', 'ourselves', 'because', 're', 'having', 'mustn', 'wasn', 'if',
             'this', 'both', 'whom', 'did', 'about', 'no', 'off', 'are', 'up', 'that', 'of',
             'by', 'those', 'too', 'hadn', 'shan', 'now', 'under', 'them', 'can', 'for',
             'wouldn', 'ours', 'who', 'some', 'in', 'but', 'nor', 'itself', 'doesn', 'through',
             'i', 'me'}

mean_reshape = lambda vecs: (sum(vecs)/len(vecs)).reshape((1, vecs[0].shape[0]))
sum_reshape  = lambda vecs: (sum(vecs)).reshape((1, vecs[0].shape[0]))

# """ Stupid feature stuff for TextTokenClassifier. """


class NoTokenizeVectorizer(CountVectorizer):
  """ Cannot pickle functions, and this assumes input is tokenized... """
  
  def __init__(self, *args, **kwargs):
    super(NoTokenizeVectorizer, self).__init__(*args, **kwargs)
    
    self.analyzer = lambda x: x
  
  """ Cannot pickle functions, even if they are identity.... """
  def __getstate__(self):
    return dict((k, v) for (k, v) in self.__dict__.items() if k != 'analyzer')
  
  def __setstate__(self, state):
    self.__dict__ = state
    self.analyzer = lambda x: x

class FeatureExtractor:
  def extract(self, docs):
    """
    Parameters
    ----------
    docs : [ [ str ] ]
        tokenized tweets
    
    Returns
    ----------
    features : (N, D) numpy float32 array
    """
    raise NotImplementedError
  
  def getDim(self):
    """
    Returns
    ----------
    dim : int
        dimensionality of feature vectors
    """
    raise NotImplementedError
  
  def getNames(self):
    """
    Returns
    ----------
    names : [ str ]
        name for each feature in vector
    """
    raise NotImplementedError
  
  def minibatchYield(self, docs, batchSize=1000, verbose=False):
    """
    Batches documents together and yields feature vectors.  May not be worth the extra
    machinery.
    
    Parameters
    ----------
    docs : iterable( [ str ] )
        each tweet tokenized, coming in as a stream
    batchSize : int
        size of batches to yield
    verbose : bool
        whether to print how many features we extracted 
    
    Yields
    ----------
    features : numpy float32 array
    """
    
    docBuffer = []
    consumed = 0
    
    for d in  docs:
      docBuffer.append(d)
      consumed += 1
      if len(docBuffer) >= batchSize: # Buffer's full
        X = self.extract(docBuffer)
        yield X
        
        docBuffer = []
        if verbose:
          print('Extracted:', consumed)

class NgramExtractor(FeatureExtractor):
  """
  Extracts TF-IDF-weighted feature vectors of n-gram features.  Assumes vocabulary and IDF
  weights have already been fit.
  """
  
  def __init__(self, vectorizers):
    """
    Parameters
    ----------
    vectorizers : [ NoTokenTfIdfVectorizer ]
        vectorizers for different n-gram orders, assume text is already tokenized and keeps
        track of its arguments
    """

    self._vectorizers = vectorizers
    self._dim = sum([len(v.vocabulary_) for v in self._vectorizers])
    
    # store names of ngram features prefixed with n-gram order and which vectorizer was applied
    self._names = []
    for vIdx, v in enumerate(self._vectorizers):
      ngram = '%d-%d-%d-' % (vIdx, *v.ngram_range)
      self._names += map(lambda x: ngram + x[0],
                         sorted(v.vocabulary_.items(), key=lambda wi: wi[1]))
  
  def extract(self, docs):
    return np.hstack([v.transform(docs).todense() for v in self._vectorizers])
  
  def getDim(self):
    return self._dim
  
  def getNames(self):
    return self._names

class EmbeddingExtractor(FeatureExtractor):
  """ Extracts sum/average of token embeddings in a tweet. """
  
  def _ldWordVectors(self, path, vocab):
    """
    Reads in word embeddings restricted to a subset of tokens.
    
    Parameters
    ----------
    path : str
        path where word embeddings are saved
    vocab : { Object:int } 
        which embeddings we should pull and their index
    
    Returns
    ----------
    embeddings : (N, D) numpy float array 
        embedding matrix
    """
    
    wvecs = {}
    
    embDim = 0
    
    f = open_file(path, 'rt', encoding='utf8')
    for i, line in enumerate(f):
      word = line[:line.find(' ')]
      
      if word in vocab:
        try:
          wemb = np.asarray([float(w) for w in line.strip().split()[1:]])
          embDim = wemb.shape[0]
          wvecs[vocab[word]] = wemb
        except Exception as ex:
          raise ex
      
      # if not i % 1000000:
      #   print ('Loading word vecs: %.1fM checked, %d found' % (i/10.**6, len(wvecs)))
    f.close()
    
    embeddings = np.zeros((len(vocab), embDim))
    for index in wvecs:
      embeddings[index,:] = wvecs[index]
    
    print('Loaded word vecs: %d unigrams found' % (len(wvecs)))
    
    return embeddings
  
  def __init__(self, pathToVectors, vocabulary, aggFnName='sum', embeddings=None):
    """
    Parameters
    ----------
    pathToVectors : str
        pretrained word vectors.  Example formatted files at
        http://nlp.stanford.edu/data/glove.twitter.27B.zip
    vocabulary : { Object:int }
        which tokens to pull embeddings for, and their index
    aggFnName : str
        denotes how to merge word embeddings \in {sum, mean}
    embeddings : numpy float array
        embedding matrix, if already computed 
    """
    
    self._path = pathToVectors
    self._vocabulary = vocabulary
    
    if embeddings is None:
      self._embeddings = self._ldWordVectors(pathToVectors, self._vocabulary)
    else:
      self._embeddings = embeddings
    
    self._dim        = self._embeddings.shape[1]
    
    if aggFnName == 'sum':
      self._aggFn = sum_reshape
    elif aggFnName == 'mean':
      self._aggFn = mean_reshape
    else:
      raise Exception('Do not recognize agg function: %s' % (aggFnName))
    
    self._aggName = aggFnName
    
    self._names = list(map(lambda x: 'word2vec-%d-%s_' % (self._dim, self._aggName) + x[0],
                           sorted(self._vocabulary.items(), key=lambda wi: wi[0])))
  
  def extract(self, docs):
    vecs = []
    
    for d in docs:
      evecs = [self._embeddings[self._vocabulary[token]]
               for token in d if token in self._vocabulary]

      if evecs:
        vecs.append(self._aggFn(evecs))
      else:
        vecs.append(np.zeros((1, self._dim)))
    
    X = np.vstack(vecs)
    return X
  
  def getDim(self):
    return self._dim
  
  def getNames(self):
    return self._names

  """ Cannot pickle functions. . . """
  def __getstate__(self):
    return dict((k, v) for (k, v) in self.__dict__.items() if k != '_aggFn')
  
  def __setstate__(self, state):
    self.__dict__ = state
    
    if self._aggName == 'sum':
      self._aggFn = sum_reshape
    elif self._aggName == 'mean':
      self._aggFn = mean_reshape
    else:
      raise Exception('Do not recognize agg function: %s' % (self._aggName))

class BiasExtractor(FeatureExtractor):
  """ Just extracts an all-ones bias term for each example. """
  def __init__(self):
    pass

  def extract(self, docs):
    X = np.asarray([[1.0] for d in docs])
    
    return X
  
  def getDim(self):
    return 1

  def getNames(self):
    return ['bias']

class SeriesExtractor(FeatureExtractor):
  """
  Runs feature extractors in series and concatenates the vectors.
  """
  
  def __init__(self, extractors):
    self._extractors = extractors
  
  def extract(self, docs):
    Xs = [e.extract(docs) for e in self._extractors]
    X = np.hstack( Xs )
    
    return X
  
  def getDim(self):
    return sum(map(lambda e: e.getDim(), self._extractors))
  
  def getNames(self):
    names = []
    for e in self._extractors:
      names += reduce(lambda ein: 'e%d-%s' % (ein[0], ein[1]), enumerate(e.getNames()), [])
    
    return names

class TextTokenClassifier:
  """ Classify tokenized tweets. """
  
  def __init__(self, extractor, labelDict, model, threshold=0.0, name='YeOldeClassifier', testFscore=None, testAcc=None, testConfMax=None):
    """
    Parameters
    ----------
    extractor : FeatureExtractor
        maps from collection of tokenized tweets to a feature matrix
    labelDict : { int:str }
        maps from output label index to label string
    model : sklearn model
        makes predictions from feature matrix
    threshold : float
        threshold for prediction
    name : str
        friendly name for me
    testFscore : float
        f-score on test set
    testAcc : float
        accuracy on test set
    testConfMax : np array
        confusion matrix on test set
    """
    
    self._extractor = extractor
    self._model     = model
    self._labelDict = labelDict
    self._name      = name
    self._modelType = model.__class__.__name__
    self._threshold = threshold
    
    # make it easier to report model performance
    self._testFscore  = testFscore
    self._testAcc     = testAcc
    self._testConfMax = testConfMax
  
  def annotate(self, tweet, annotations, metadata):
    """ TODO: Interface called by Mark's backend code """
    
    annotations[self._name] = self.predict(metadata['tokens'], pretokenized=True)
    
    return annotations
  
  def deserialize(path):
    f = open_file(path, 'rb')
    classifier = pickle.load(f)
    f.close()
    
    return classifier
  
  def serialize(self, path):
    outFile = open_file(path, 'wb')
    pickle.dump(self, outFile)
    outFile.close()
  
  def extract(self, docs, pretokenized=True):
    if pretokenized:
      X = self._extractor.extract( docs )
    else:
      X = self._extractor.extract( [tokenize(t) for t in docs] )

    return X
  
  def fitNewData(self, X, y, folds, tuningFolds, model, beta=1.0, maxPropDropped=0.30):
    '''
    Keep feature extractors fixed, but retrain the model on new data.
    
    Parameters
    ----------
    X : numpy float array
        features
    y : numpy int vector
        gold labels
    folds : numpy int vector
        fold each example belongs to
    tuningFolds : [ int ]
        which folds to tune on
    model : sklearn.linear_model.SGDClassifier
        model to train, with regularization parameters set.  Should be able to supply fancier
        models so long as they have "fit" and "decision_function" methods
    beta : float
        how much to weight precision vs. recall when choosing threshold
    maxPropDropped : float
        maximum proportion of examples that should be dropped
    
    Returns
    ----------
    bestAccuracy : float
    bestFscore   : float
    '''
    origN = X.shape[0]
    
    # Drop examples with missing labels
    # TODO there is a bug here that drops all examples if fold is not set, i might have fixed it
    folds = [fold for y, fold in zip(y, folds) if y!=-1]
    X = X[y!=-1,:]
    y = y[y!=-1]
    N = X.shape[0]
    
    # pick threshold based cross-fold validation by weighted f-score 
    
    yhat_scores = []
    
    print('Fitting on folds... ', end='')
    allDevIdxes = []
    for tuneFold in tuningFolds:
      trainIdxes = [i for i, fold in enumerate(folds) if fold != tuneFold]
      devIdxes   = [i for i, fold in enumerate(folds) if fold == tuneFold]
      allDevIdxes.extend(devIdxes)
      
      model.fit(X[trainIdxes,:], y[trainIdxes])
      scores = model.decision_function(X[devIdxes,:])
      if len(scores.shape)==1: # binary predictions
        scores = np.vstack([-scores, scores]).T
      
      yhat_scores.append(scores)
      print('%d... ' % (tuneFold), end='')
    print('')
    
    yhat_scores  = np.vstack(yhat_scores)
    
    if maxPropDropped > 0.0:
      cand_threshs = np.sort(np.max(yhat_scores, axis=1))
    else:
      cand_threshs = [float('-inf')]
    
    # calculate accuracy, precision, recall, f-score for different thresholds:
    accs, fs, n_dropped = [], [], []
    
    isBinary  = yhat_scores.shape[1] == 2
    fscoreAvg = 'binary' if isBinary else 'macro'
    
    preds = np.argmax(yhat_scores, axis=1)
    for threshold in cand_threshs:
      dropped = np.max(yhat_scores, axis=1) < threshold
      filteredPreds = preds[~dropped]
      filteredGold  = (y[allDevIdxes])[~dropped]
      
      numLabels = len(set(np.unique(filteredGold)) | set(np.unique(filteredPreds)))
      
      fscoreAvgTmp = fscoreAvg
      
      # Have to do this when we have filtered down to only two labels and have dropped 1...
      posLabel = np.max(filteredGold)
      if numLabels == 2:
        fscoreAvgTmp = 'binary'
      elif numLabels == 1:
        continue # filtered out all positive examples, nothing to learn
      
      try:
        fs.append(fbeta_score(filteredGold, filteredPreds,
                              beta=beta, average=fscoreAvgTmp,
                              pos_label=posLabel)) # weighted f-score
      except Exception as ex:
        import pdb; pdb.set_trace()
        raise ex
      
      n_dropped.append(dropped.sum())
    
    # keep thresholds that drop sufficiently few examples
    fewDropped = [i for i, n in enumerate(n_dropped) if float(n)/N <= maxPropDropped]
    
    filteredFs = [fs[i] for i in fewDropped]
    bestFscoreIdx = np.argmax(filteredFs)
    
    bestFscore    = fs[bestFscoreIdx]
    bestThreshold = cand_threshs[bestFscoreIdx]
    bestNDropped = n_dropped[bestFscoreIdx]
    
    dropped = np.max(yhat_scores, axis=1) < bestThreshold
    filteredPreds = preds[~dropped]
    filteredGold  = (y[allDevIdxes])[~dropped]
    
    bestAcc  = accuracy_score(filteredGold, filteredPreds)
    bestPrec = precision_score(filteredGold, filteredPreds, average=fscoreAvg)
    bestRec  = recall_score(filteredGold, filteredPreds, average=fscoreAvg)
    
    print("X-val N_dropped: %d/%d, Accuracy: %0.2f, Precision: %0.2f, Recall: %0.2f, %s weighted F1: %0.2f, Threshold: %e" %
        (bestNDropped, N, bestAcc, bestPrec, bestRec, fscoreAvg, bestFscore, bestThreshold))
    
    model.fit( X, y ) # train on all folds
    self._model = model
    self._threshold = bestThreshold
    
    return bestAcc, bestFscore
  
  def fit(docs, y_gold, y_dictionary, folds, tuningFolds, model, ngramRange=(1, 3), embeddingPath=None, beta=1.0, maxPropDropped=0.5, name='YeOldeClassifier'):
    '''
    Convenience method.  Fits model to data and reports CV macro F1-score.
    
    Parameters
    ----------
    docs : [ str ]
        tweet texts
    y_gold : [ str ]
        true labels
    y_dictionary : { str:int }
        mapping from label to index
    folds : numpy int vector
        fold each example belongs to
    tuningFolds : [ int ]
        which folds to evaluate on
    model : sklearn.linear_model.SGDClassifier
        model to train, with regularization parameters set.  Should be able to supply fancier
        models so long as they have "fit" and "decision_function" methods
    ngramRange : ( int, int )
        min and max order for n-grams
    embeddingPath : str
        path to word embeddings.  If None, do not extract word embedding features
    beta : float
        how much to weight precision vs. recall when choosing threshold
    maxPropDropped : float
        maximum proportion of examples that should be dropped
    name : str
        friendly name for this classifier
    
    Returns
    ----------
    classifier   : TextTokenClassifier
    bestAccuracy : float
    bestFscore   : float
    '''
    
    # Come up with vocabulary, feature extractors
    tokens = [tokenize(d) for d in docs]
    
    es = []
    if (ngramRange[0] < 0 and ngramRange[1] < 0) and (not embeddingPath):
      es.append(BiasExtractor()) # not extracting any features, just a majority baseline
    else:
      if ngramRange[0] > 0 and ngramRange[1] > 0:
        vectorizers = []
        
        vectorizer = NoTokenizeVectorizer(encoding=u'utf-8', decode_error=u'strict',
                                     strip_accents=None, analyzer=lambda x: x,
                                     stop_words=STOPWORDS, ngram_range=ngramRange,
                                     max_df=0.9, min_df=2, max_features=10000)
        
        vectorizer.fit(tokens)
        vectorizers.append(vectorizer)
        print('Init %d-%d-gram extractor' % ngramRange)
        
        es.append(NgramExtractor(vectorizers))
      
      if embeddingPath:
        maxEmbFeatures = 10000
        
        # Find most common tokens to extract
        vectorizer = NoTokenizeVectorizer(encoding=u'utf-8', decode_error=u'strict',
                                     strip_accents=None, analyzer=lambda x: x,
                                     stop_words=STOPWORDS, ngram_range=(1, 1),
                                     max_df=0.9, min_df=2, max_features=maxEmbFeatures)
        
        vectorizer.fit(tokens)
        vocabulary = vectorizer.vocabulary_
        
        es.append(EmbeddingExtractor(embeddingPath, vocabulary, 'sum'))
        es.append(EmbeddingExtractor(embeddingPath, vocabulary, 'mean'))
    
    extractor = SeriesExtractor(es)
    
    X = extractor.extract(tokens)
    origN = X.shape[0]
    y = np.asarray([y_dictionary[value] if value in y_dictionary else -1 for value in y_gold])
    
    classifier = TextTokenClassifier(extractor, {i:w for w,i in y_dictionary.items()},
                                     model, threshold=0.0, name=name)
    
    bestAcc, bestFscore = classifier.fitNewData(X, y, folds, tuningFolds, model,
                                                beta=beta, maxPropDropped=maxPropDropped)
    
    return classifier, bestAcc, bestFscore
  
  def predict(self, docs, pretokenized=True):
    ''' 
    Parameters
    ----------
    docs : [ str ] or [ [ str ] ]
        tweet texts or tokens if already tokenized
    pretokenized : bool
        defaults to True, expects input to be token list
    
    Returns
    ----------
    preds : [ str ]
        predicted labels
    '''
    
    X = self.extract( docs, pretokenized )
    
    scores = self._model.decision_function(X)
    
    # make sure we only make predictions for those with sufficiently high confidence
    if len(scores.shape)==1:
      scores = np.vstack([-scores, scores]).T
    
    predInts = np.argmax(scores, axis=1)
    predInts[np.max(scores, axis=1) < self._threshold] = -1 # avoid making a prediction
    
    preds = [self._labelDict[intLabel] if intLabel >= 0 else 'None'
             for intLabel in predInts]
    
    return preds
  
  def confidence(self, docs, pretokenized=True):
    '''
    Parameters
    ----------
    docs : [ str ] or [ [ str ] ]
        tweet texts or tokens if already tokenized
    pretokenized : bool
        defaults to True, expects input to be token list
    
    Returns
    ----------
    labelScore : [ { str:float } ]
        confidence scores for each label for a document
    '''
    
    if pretokenized:
      X = self._extractor.extract( docs )
    else:
      X = self._extractor.extract( [tokenize(t) for t in docs] )
    
    scores = self._model.decision_function(X)
    
    # if only two labels to predict, then flip sign on confidence -- score is for positive
    # class
    if len(scores.shape)==1:
      scores = np.vstack([-scores, scores]).T
    
    labelScores = []
    for row in scores:
      labelScores.append( {self._labelDict[i]:v for i, v in enumerate(row)} )
    
    return labelScores
