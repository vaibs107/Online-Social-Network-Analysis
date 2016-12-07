"""
Online Social Network Analysis - Project 3

In this project, I have built a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data comes from
the website IMDB.com.

The below code first preprocesses the data in different ways (creating different
features), then compares the cross-validation accuracy of each approach. Then,
compute accuracy on a test set and do some analysis of the errors.

The output of this analysis is provided in Log.txt.
"""

from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request


def download_data():
    """ Download he IMDB movies' data and unzip it.
    """
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.

    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenizes a string.
    The string is converted to lowercase.
    If keep_internal_punct is False, then only the alphanumerics (letters, numbers and underscore) are returned.
    If keep_internal_punct is True, then also punctuation that is inside of a word is retained. 
    E.g., for the token "isn't" if keep_internal_punct=True then token will be "isn't"
    otherwise, it is split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.
    """
    if(keep_internal_punct):
        array1 = np.array(re.sub('[^A-Za-z0-9\']+', ' ', doc.strip('\?').lower()).split())
        return array1
    else:
        array2 = np.array(re.sub('[^A-Za-z0-9]+', ' ', doc.strip('\?').lower()).split())
        return array2
   
    pass


def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.
    """
    for k in Counter(tokens).keys():
        feats["token="+k] = Counter(tokens).get(k)
    pass


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. 

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.
    """
    i=0
    j=0
    window = []
    for l in range(0, len(tokens)-2):
        window = tokens[l:l+k]
        for i in range(len(window)-1):
            for j in range(i+1, len(window)):
                if 'token_pair='+window[i]+'__'+window[j] in feats:
                    feats['token_pair='+window[i]+'__'+window[j]] += 1
                else:
                    feats['token_pair='+window[i]+'__'+window[j]] = 1
    pass


neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])


def lexicon_features(tokens, feats):
    """
    Adds features indicating how many times a token appears that matches either
    the neg_words or pos_words (defined above). The matching ignores
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.
    """
    neg_words_count = 0
    pos_words_count = 0
    for t in tokens:
        if t.lower() in neg_words:
            neg_words_count += 1
        elif t.lower() in pos_words:
            pos_words_count += 1
    feats['pos_words'] = pos_words_count
    feats['neg_words'] = neg_words_count
    pass


def featurize(tokens, feature_fns):
    """
    Computes all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.
    """
    feats = defaultdict(lambda: 0)
    for f in feature_fns:
        f(tokens, feats)
    return sorted(feats.items(), key = lambda x:(x[0],x[1]), reverse=False)
    pass


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, this function creates a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix:  documentation --> https://goo.gl/f5TiF1
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically 
    """
    data = defaultdict(lambda: 0)
    matrix_data = []
    row = []
    column = []
    f =[]
    pos = 0
    
    
    for i in range(len(tokens_list)):
        tmp = featurize(tokens_list[i],feature_fns)
        f.append(tmp)
        for j in range(len(tmp)):
            if tmp[j][0] not in data:
                data[tmp[j][0]] = 1
            else:
                data[tmp[j][0]]  += 1
                
    sorted_data = sorted(data.keys())
                
    #Vocab
    if vocab == None:
        vocab = defaultdict(lambda: 0)
        for d in sorted_data:
            if data[d] >= min_freq:
                vocab[d] = pos
                pos+=1
    
    #Data, row, column for csr matix
    for i in range(len(f)):
        for j in f[i]:
            if j[0] in vocab:
                matrix_data.append(j[1])
                row.append(i)
                column.append(vocab[j[0]])
    return csr_matrix((matrix_data,(row,column)), shape=(len(tokens_list),len(vocab)),dtype='int64'),vocab
    pass


def accuracy_score(truth, predicted):
    """ Computes accuracy of predictions.
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Computes the average testing accuracy over k folds of cross-validation. 
    Sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    cv = KFold(len(labels), k)
    accuracies = []
    for train_ind, test_ind in cv:
        clf.fit(X[train_ind], labels[train_ind])
        predictions = clf.predict(X[test_ind])
        accuracies.append(accuracy_score(labels[test_ind], predictions))
    return np.mean(accuracies)
    pass


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerates all possible classifier settings and compute the
    cross validation accuracy for each setting. This is used
    to determine which setting has the best accuracy.

    For each setting, a LogisticRegression classifier is constructed
    and cross-validation accuracy is computed for that setting.

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.
    """
    result = []
    
    for i in range(len(punct_vals)):
        for l in min_freqs:
            tokens = [tokenize(d, punct_vals[i]) for d in docs]
            for j in range(len(feature_fns)):
                X,vocab = vectorize(tokens, [feature_fns[j]], l)
                accuracy = cross_validation_accuracy(LogisticRegression(), X, labels, 5)
                each_result = {}
                each_result['punct'] = punct_vals[i]
                each_result['features'] = feature_fns[j]
                each_result['min_freq'] = l
                each_result['accuracy'] = accuracy
                result.append(each_result)
                for k in range(j+1, len(feature_fns)):
                        X,vocab = vectorize(tokens, [feature_fns[j], feature_fns[k]], l)
                        accuracy = cross_validation_accuracy(LogisticRegression(), X, labels, 5)
                        each_result = {}
                        each_result['punct'] = punct_vals[i]
                        each_result['features'] = [feature_fns[j],feature_fns[k]]
                        each_result['min_freq'] = l
                        each_result['accuracy'] = accuracy
                        result.append(each_result)
            X,vocab = vectorize(tokens, feature_fns, l)
            accuracy = cross_validation_accuracy(LogisticRegression(), X, labels, 5)
            each_result = {}
            each_result['punct'] = punct_vals[i]
            each_result['features'] = feature_fns
            each_result['min_freq'] = l
            each_result['accuracy'] = accuracy
            result.append(each_result)
    final_res = sorted(result, key = lambda x:x['accuracy'], reverse=True)
    
    return final_res
    pass


def plot_sorted_accuracies(results):
    """
    Plots all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Saves to "accuracies.png".
    """
    freq = [2,5,10]
    temp_list = []
    for n_folds in freq:
        for i in range(len(results)):
            if(results[i].get('min_freq') == n_folds):
                temp_list.append(results[i].get('accuracy'))
    temp_list.sort()
    
    plt.figure()
    plt.plot(temp_list)
    plt.xlabel('setting')
    plt.ylabel('accuracy')
    plt.savefig('accuracies.png')

    pass


def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    the mean accuracy of all combinations with a particular
    setting is calculated in this function. 
    For example, compute the mean accuracy of all runs with
    min_freq=2.
    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """  
    accuracy_result ={}    
    for every_result in results:
        for k in every_result:
            if k == 'features':
                function_names=[]
                for j in every_result[k]:
                    function = str(j).split()[1] 
                    function_names.append(function)
                functions= k+"= "+"  ".join(function_names)
                if functions in accuracy_result:
                    accuracy_result[functions].append(every_result['accuracy'])
                else:
                    accuracy_result[functions] = [every_result['accuracy']]
            elif k=='punct':
                functions= k+"= "+ str(every_result[k])
                if functions in accuracy_result:
                    accuracy_result[functions].append(every_result['accuracy'])
                else:
                    accuracy_result[functions] = [every_result['accuracy']]
            elif k=='min_freq':
                functions = k+"= "+ str(every_result[k])
                if functions in accuracy_result:
                    accuracy_result[functions].append(every_result['accuracy'])
                else:
                    accuracy_result[functions] = [every_result['accuracy']]
   
    mean_accuracy_list =[]
    for every_acc_result in accuracy_result:
        mean_accuracy_list.append((np.mean(accuracy_result[every_acc_result]),every_acc_result))
    return sorted(mean_accuracy_list, key = lambda x:x[0], reverse = True)
    
    pass


def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    tokens_list = [tokenize(d,keep_internal_punct=best_result['punct']) for d in docs]
    X_fit_best, vocab_fit_best = vectorize(tokens_list, best_result['features'], min_freq=best_result['min_freq'])
    clf = LogisticRegression()
    return clf.fit(X_fit_best,labels),vocab_fit_best    
    
    pass


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.

    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """    
    top_coefs = []
    coef = clf.coef_[0]
    sorted_vocab_keys = sorted(vocab.keys())
    sorted_index_coef = np.argsort(coef)
    if label==0:
        top_n_coef_index = sorted_index_coef[:n]
    elif label==1:
        top_n_coef_index = sorted_index_coef[::-1][:n]
    
    for i in top_n_coef_index:
        top_coefs.append((sorted_vocab_keys[i],abs(coef[i])))
    
    return top_coefs
    
    pass


def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab is 
    passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    test_docs, test_labels = read_data(os.path.join('data', 'test'))
    tokens_list = [tokenize(d,keep_internal_punct=best_result['punct']) for d in test_docs]
    X_test, vocab = vectorize(tokens_list, best_result['features'], min_freq=best_result['min_freq'],vocab = vocab)
    return test_docs,test_labels,X_test

    pass


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the largest margin. 
    By using the .predict_proba function of LogisticRegression <https://goo.gl/4WXbYA>, 
    we can get the predicted probabilities of each class for each instance.
    Firstly all incorrectly classified documents are identified,
    then sorted in descending order of the predicted probability
    for the incorrect class.
    
    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.
    Returns:
      Nothing; see Log.txt for example printed output.
    """
    predicted_vals = clf.predict(X_test)
    predicted_probabilities = clf.predict_proba(X_test)
    misclassified=[]
    top_misclassified = []
    for i in range(len(predicted_probabilities)):
        if(predicted_vals[i]!=test_labels[i]):           
            misclassified.append((predicted_vals[i],test_labels[i],test_docs[i],predicted_probabilities[i][predicted_vals[i]])) 
    
    top_misclassified = sorted(misclassified,key=lambda x:x[3],reverse=True)[:n]
    for every_misclassified in top_misclassified:
        print("truth=",every_misclassified[1],"predicted=",every_misclassified[0],"proba=",every_misclassified[3])
        print(every_misclassified[2])

    pass


def main():
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()
