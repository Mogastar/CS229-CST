from __future__ import division
import locale
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import sklearn as sk
import re
import pickle
import glob
import scipy
import itertools
import operator
import copy
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import linear_model
from numpy import inf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import matplotlib as mpl
import sys
from sklearn.metrics import roc_curve, auc
from matplotlib import colors


#os.chdir('E:\Stanford\Courses\CS 229\Project\CS229-CST')

locale.setlocale(locale.LC_ALL, '')

# Define directories
data_dir = os.path.join(os.getcwd(), 'Datasets/')
py_dir = os.path.join(os.getcwd(), 'Datasets/pythonquestions/')
r_dir = os.path.join(os.getcwd(), 'Datasets/rquestions/')
# Define a stemmer
stemmer = nltk.SnowballStemmer('english')


'''
###############################################################################
Functions
###############################################################################
'''


def load_data(dir):
    '''Load the data.'''
    
    # File names
    answers_file = os.path.join(dir, 'Answers.csv')
    questions_file = os.path.join(dir, 'Questions.csv')
    tags_file = os.path.join(dir, 'Tags.csv')
    
    # Load data
    answers = pd.read_csv(answers_file)
    questions = pd.read_csv(questions_file)
    tags = pd.read_csv(tags_file)
    
    # Convert OwnerUserId to int64 and fill NAs with -1
    answers.OwnerUserId = answers.OwnerUserId.fillna(-1.0).astype('int64')
    questions.OwnerUserId = questions.OwnerUserId.fillna(-1.0).astype('int64')
    
    # Convert to dates
    answers.CreationDate = pd.to_datetime(answers.CreationDate)
    questions.CreationDate = pd.to_datetime(questions.CreationDate)
    
    # Create merged dataset
    df = answers.set_index('ParentId').join(questions.set_index('Id'), 
                           lsuffix = '_answers', rsuffix = '_questions')
    df.reset_index(inplace = True)
    df.rename(columns = {'index': 'Id_questions'}, inplace = True)
    
    print ("Loaded data")
    return (df, tags)


def process_data(df0, threshold = 3):
    '''Add features.'''
    
    # Only keep answers whose question score is >= 5
    df = df0[df0.Score_questions >= threshold]
    
    # DeltaT between question and answer dates
    df['DeltaT'] = df['CreationDate_answers'] - df['CreationDate_questions']
    # Length of the answer and question bodies
    df['Bodylength_answers'] = [len(body) for body in df['Body_answers']]
    df['Bodylength_questions'] = [len(body) for body in df['Body_questions']]
    df['Bodylength_std'] = df['Bodylength_answers'] / df['Bodylength_questions']
    
    # Number of href links
    df['LinksNumber'] = df['Body_answers'].apply(lambda s: s.count('href'))
    
    # Number of code parts
    df['CodeNumber'] = df['Body_answers'].apply(lambda s: s.count('<code>'))
    
    # Standardized scores
    df['Score_std'] = [float(df['Score_answers'].iloc[i]) / max(1.0, 
                      df['Score_questions'].iloc[i]) for i in range(len(df))]

    print ("Processed data")
    return df


def get_voc(df, vocfile):
    '''Get vocabulary from the bodies of answers and write to file.'''
    
    # Get vocabulary
    allwords = []
    for i in range(len(df)):
        word_stream = df['Body_answers'].iloc[i]
        allwords += stemming(word_stream).keys()
        if (i % 10000 == 0):
            print ("Done {0}/{1}".format(i+1, len(df)))
    voc = set(allwords)
            
    # Write to file
    with open(vocfile, 'w') as vocf:
        for word in voc:
            vocf.write("%s \n" % word)
    
    print ("Got vocabulary")
    return voc


def process_voc(vocfile, word_files = [], process = False):
    '''Read, remove some words from the vocabulary and write it.'''
    
    # Read and sort vocabulary
    with open(vocfile, 'r') as vocf:
        voc = [line.rstrip() for line in vocf]
    voc.sort(cmp = locale.strcoll)
    
    # Return if not processing
    if not process:
        print ("Read vocabulary")
        return voc
    
    # Add words from files, such as HTML tags
    for word_file in word_files:
        with open(word_file, 'r') as wf:
            words = [line.rstrip() for line in wf]
        voc += words
            
    # Write to file
    voc = list(set(voc))
    voc.sort(cmp = locale.strcoll)
    with open(vocfile, 'w') as vocf:
        for word in voc:
            vocf.write("%s \n" % word)

    print ("Processed vocabulary")
    return voc
        

def get_design(df, voc, start, end, work_dir, word_files = []):
    ''' 
    Get elements for sparse design matrix in COO format. 
      - df: whole dataframe
      - voc: dictionary('word': index)
      - start, end: line indexes we want to process [start, end) 
    '''
    
    val = []
    row = []
    col = []
    # Construct sparse matrix
    for i in range(start, end):
        word_stream = df['Body_answers'].iloc[i]
        word_nb = stemming(word_stream, word_files)
        for word in word_nb:
            val.append(word_nb[word])
            row.append(i)
            col.append(voc[word])
    # Save matrix
    with open(os.path.join(work_dir, 'val_{0}_{1}.txt'.format(start, end-1)),
              'w') as valf:
        pickle.dump(val, valf)
    with open(os.path.join(work_dir, 'row_{0}_{1}.txt'.format(start, end-1)),
              'w') as rowf:
        pickle.dump(row, rowf)
    with open(os.path.join(work_dir, 'col_{0}_{1}.txt'.format(start, end-1)),
              'w') as colf:
        pickle.dump(col, colf)
    # Print message
    print ("Got design matrix for indices [{0}, {1}).".format(start, end))
        
        
def aggregate_design(work_dir):
    ''' 
    Aggregate design matrix given in COO format. 
      - work_dir: directory where we can expect to find
        - val_*_*.txt files
        - row_*_*.txt files
        - col_*_*.txt files
      - shape: tuple for the aggregated matrix shape
    '''
    
    # Values
    val= []
    for valname in glob.glob(os.path.join(work_dir, 'val_*.txt')):
        with open(valname, 'r') as valf:
            val += pickle.load(valf)
    # Row
    row = []
    for rowname in glob.glob(os.path.join(work_dir, 'row_*.txt')):
        with open(rowname, 'r') as rowf:
            row += pickle.load(rowf)
    # Col
    col = []
    for colname in glob.glob(os.path.join(work_dir, 'col_*.txt')):
        with open(colname, 'r') as colf:
            col += pickle.load(colf)
    
    # Sort by row for easier slicing
    sorted_row_idx = np.argsort(row)
    val = np.array(val)[sorted_row_idx]
    row = np.array(row)[sorted_row_idx]
    col = np.array(col)[sorted_row_idx]
    
    print ("Aggregated sparse matrix.")
    return val, row, col
        

def stemming(word_stream, word_files = []):
    '''
    Take a string and return a dictionary with words and their number of 
    occurences.
      - word_stream: string
      - word_files: list of filenames strings for additional words
                    to check for
    '''
    
    # Get words
    wordList = re.findall(r"\w+|[^\w\s]", word_stream)
    dictList = [stemmer.stem(word) for word in wordList]

    # Remove some words
    newList = []
    for word in dictList:
        # Remove one character words
        if len(word) < 2:
            continue
        # Remove words too long
        if len(word) > 15:
            continue
        # Remove numbers
        if (word.isdigit()):
            continue
        # Remove words with at least 2 digits
        digit_count = sum(c.isdigit() for c in word)
        if digit_count > 2:
            continue
        # Add word to processed vocabulary otherwise
        newList.append(word)
    
    # Create count dictionary
    dict_count = dict(Counter(newList))

    # Add specific words if they appear
    for word_file in word_files:
        with open(word_file, 'r') as wf:
            words = [line.rstrip() for line in wf]
            for word in words:
                occur = word_stream.count(word)
                if occur != 0:
                    dict_count[word] = occur

    return dict_count


def Reg_nS_Deltat(score, time, nbins = 1000, tol = 1e-5):
    '''
    Running regression on DeltaT and Score_std
    '''
    
    time = time.dt.total_seconds()
    time_sorted = time.sort_values()
    time_sorted = time_sorted[time_sorted >= 0]
    ind_sorted  = time_sorted.index.values # indices as numpy array
    score_sorted = score.reindex(time_sorted.index)
    score_sorted = score_sorted.values     # sorted scores as numpy array
    
    _, time_bins = pd.qcut(time_sorted, nbins, retbins = True, 
                           duplicates = "drop")
    score_quantile = np.array_split(score_sorted, nbins)
    ind_quantile   = np.array_split(ind_sorted, nbins)
    
    time_picks = np.zeros(nbins);
    score_picks = np.zeros(nbins);

    for i in range(nbins):
        score_picks[i] = np.max(score_quantile[i])
        indices        = ind_quantile[i]
        ind_max        = indices[np.argmax(score_quantile[i])]
        time_picks[i]  = time_sorted.loc[ind_max]
    
    score_picks[score_picks < tol] = tol
    score_picks_log = np.log(score_picks).reshape(len(score_picks), 1)
    time_picks = time_picks.reshape(len(time_picks), 1)
    
    reg = linear_model.LinearRegression()
    reg.fit(time_picks, score_picks_log)
    
    return reg, score_picks_log, time_picks, time_sorted, score_sorted, time_bins


def GaussianDA(X, y, analysis_type):
    '''
    The Gaussian Discriminant Analysis model
    '''

    if (analysis_type == "Linear"):
        GDA = LinearDiscriminantAnalysis(solver = "svd", store_covariance=True)
    else:
        GDA = QuadraticDiscriminantAnalysis(store_covariance=True)

    GDA.fit(X, y)
    return GDA

def Logistic_Regres(X, y):
    '''
    Logistic Regression model
    '''
    LRM = LogisticRegression(penalty='l2', tol=0.01)
    LRM.fit(X, y)
    return LRM

def RFClassifier(X, y):
    '''
    Random Forest Classifier
    '''
    clf = RandomForestClassifier(n_estimators=20)
    n_iter_search = 20
    param_dist = {"max_depth": [3, None],
                  "min_samples_split": sp_randint(2, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search)
    random_search.fit(X, y)
    return random_search

def separate(val, row, col, y, test_size, seed = 0):
    '''
    Separate a sparse matrix in COO format and a vector y into 2 sets
        - set 0 of size (1 - test_size) %
        - set 1 of size test_size %.
    '''
    
    # Separate indices
    ind = np.arange(y.shape[0])
    ind0, ind1 = sk.model_selection.train_test_split(ind, 
                                                     test_size = test_size, 
                                                     random_state = seed)
    ind0.sort()
    ind1.sort()
    ind1_set = set(ind1)
    
    # Separate X
    sparse_ind0 = []
    sparse_ind1 = []
    not0 = 0
    not1 = 0
    row2 = copy.copy(row)
    last_row = -1
    for i in range(len(row)):
        if row[i] in ind1_set:
            sparse_ind1.append(i)
            row2[i] -= not1
            if last_row != row[i]:
                not0 += 1
        else:
            sparse_ind0.append(i)
            row2[i] -= not0
            if last_row != row[i]:
                not1 += 1
        last_row = row[i]
        
    f0 = operator.itemgetter(*sparse_ind0)
    val0 = np.array(f0(val))
    row0 = np.array(f0(row2))
    col0 = np.array(f0(col))
    f1 = operator.itemgetter(*sparse_ind1)
    val1 = np.array(f1(val))
    row1 = np.array(f1(row2))
    col1 = np.array(f1(col))
    
    # Separate y
    y0 = y[ind0, :]
    y1 = y[ind1, :]
    
    return val0, row0, col0, y0, val1, row1, col1, y1

def plot_ellipse(splot, mean, cov, color):
    '''
    Plotting ellipses for Normal distributions
    '''
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    # filled Gaussian at 2 standard deviation
    ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
                              180 + angle, facecolor=color,
                              edgecolor='yellow',
                              linewidth=2, zorder=2)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)
    splot.set_xticks(())
    splot.set_yticks(())
    

'''
###############################################################################
Data manipulation
###############################################################################
'''


threshold = 5
first_time = False

# Choose dataset
work_dir = r_dir
# Load data
df0, tags = load_data(work_dir)
# Process data
df = process_data(df0, threshold = 5)
del df0

# Get vocabulary (first time)
if first_time:
    voc = get_voc(df, os.path.join(work_dir, 'Vocabulary.txt'))
    
# Read dictionary (other times)
word_files = [os.path.join(data_dir, 'HTML_tags.txt')]
voc_list = process_voc(os.path.join(work_dir, 'Vocabulary.txt'), 
                       word_files, process = first_time)   
voc = dict(itertools.izip(voc_list, range(len(voc_list))))
     
# Get design matrix (first time)
if first_time:
    for i in range((len(df) + 9999) / 10000):
        start = 10000 * i
        end = min(10000 * (i+1), len(df))
        get_design(df, voc, start, end, work_dir, word_files)
        
# Aggregate design matrix in sparse format
val, row, col = aggregate_design(work_dir)

# Vectors to split
accepted = np.array(df['IsAcceptedAnswer'], dtype = int)
deltaT_s = np.array(df['DeltaT'].dt.total_seconds())
bodylength = np.array(df['Bodylength_std'])
y = np.stack((accepted, deltaT_s, bodylength), axis = 1)

# Separate datasets
val_temp, row_temp, col_temp, y_temp, val_test, row_test, col_test, y_test = \
    separate(val, row, col, y, test_size = 0.01)
val_train, row_train, col_train, y_train, val_cv, row_cv, col_cv, y_cv = \
    separate(val_temp, row_temp, col_temp, y_temp, test_size = 0.1)
del val_temp, row_temp, col_temp, y_temp

# Construct sparse matrices
X_train = scipy.sparse.coo_matrix((val_train, (row_train, col_train)),
                                  shape = (y_train.shape[0], len(voc)))
X_cv = scipy.sparse.coo_matrix((val_cv, (row_cv, col_cv)),
                                  shape = (y_cv.shape[0], len(voc)))
X_test = scipy.sparse.coo_matrix((val_test, (row_test, col_test)),
                                 shape = (y_test.shape[0], len(voc)))

# Delete unused shit
del val, row, col
del val_train, row_train, col_train
del val_cv, row_cv, col_cv
del val_test, row_test, col_test


'''
###############################################################################
Tests
###############################################################################
'''


# NB

bernoulli = False

if bernoulli:
    NB = BernoulliNB()
else:
    NB = MultinomialNB()
NB.fit(X_train, y_train[:, 0])

# Best words

if first_time:
    order = np.argsort(NB.coef_[0, :])
    with open('I_have_the_best_words.txt', 'w') as f:
        for i in range(200):
            f.write(voc_list[a[-(i + 1)]])
            f.write('\n')

# Predictions 

pred_NB_train = NB.predict(X_train)
proba_NB_train = np.max(NB.predict_proba(X_train), axis = 1)
pred_NB_cv = NB.predict(X_cv)
proba_NB_cv = np.max(NB.predict_proba(X_cv), axis = 1)
pred_NB_test = NB.predict(X_test)
proba_NB_test = np.max(NB.predict_proba(X_test), axis = 1)

# Regression on time difference and standardized score

reg, score_log, time_picks, time_sorted, score_sorted, bins = \
    Reg_nS_Deltat(df['Score_std'], df['DeltaT'], 5000)
pred = np.exp(reg.predict(time_picks))
plt.plot(time_sorted, score_sorted, '.', markersize = 3)
plt.plot(time_picks, pred, 'r-')
plt.plot(time_picks, np.zeros(pred.shape), 'k-')
plt.axis([-0.1e8, 3e8, -1, 6])
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlabel('Time difference (in seconds)')
plt.ylabel('Standardized score')
plt.title('Regression on time difference and standardized score')
#plt.savefig("regression_Time_Score.png", dpi=1200)

# Time bins

bins[-1] = inf
time_bin_train = pd.cut(y_train[:, 1], bins, labels = False,
                        include_lowest = True) / 5000
time_bin_train[np.isnan(time_bin_train)] = 0
time_bin_cv = pd.cut(y_cv[:, 1], bins, labels = False,
                        include_lowest = True) / 5000
time_bin_cv[np.isnan(time_bin_cv)] = 0
time_bin_test = pd.cut(y_test[:, 1], bins, labels = False,
                        include_lowest = True) / 5000
time_bin_test[np.isnan(time_bin_test)] = 0

# Logistic regression

envlp_train = np.exp(reg.predict(y_train[:, 1].reshape(y_train.shape[0], 1)))
envlp_cv = np.exp(reg.predict(y_cv[:, 1].reshape(y_cv.shape[0], 1)))
envlp_test = np.exp(reg.predict(y_test[:, 1].reshape(y_test.shape[0], 1)))

data_train = np.stack((pred_NB_train, proba_NB_train, 
                       envlp_train.flatten(), y_train[:, 2],
                       time_bin_train), axis = 1)
data_cv = np.stack((pred_NB_cv, proba_NB_cv, 
                    envlp_cv.flatten(), y_cv[:, 2],
                    time_bin_cv), axis = 1)
data_test = np.stack((pred_NB_test, proba_NB_test, 
                      envlp_test.flatten(), y_test[:, 2],
                      time_bin_test), axis = 1)

value_train = y_train[:, 0]
value_cv = y_cv[:, 0]
value_test = y_test[:, 0]

plt.plot(data_train[value_train == 0, 1], data_train[value_train == 0, 4], "r.")
plt.plot(data_train[value_train == 1, 1], data_train[value_train == 1, 4], "b.")
plt.show()

LRM = Logistic_Regres(data_train[:, [1, 2]], value_train)
value_pred = LRM.predict(data_cv[:, [1, 2]])
accuracy = np.mean(value_pred == value_cv)
print("Logistic Regression: {0:.2f}%".format(100*accuracy))

NN = MLPClassifier(alpha = .005)
NN.fit(data_train[:, [1, 2]], value_train)
value_pred = NN.predict(data_cv[:, [1, 2]])
accuracy = np.mean(value_pred == value_cv)
print("Neural Nets: {0:.2f}%".format(100*accuracy))

RFC = RFClassifier(data_train[:, [1, 2]], value_train)
value_pred = RFC.predict(data_cv[:, [1, 2]])
accuracy = np.mean(value_pred == value_cv)
print("Random Forest Classifier with Cross Validation: {0:.2f}%".format(100*accuracy))


## Plotting Gaussian discriminant analysis

GDA = GaussianDA(data_train[:, [1, 2]], value_train, "Linear")
value_pred = GDA.predict(data_cv[:, [1, 2]])
GDA_accuracy = np.mean(value_pred == value_cv)
print("Gaussian Discriminant Analysis: {0:.2f}%".format(100*GDA_accuracy))

plotGDA = plt.subplot()
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)
plt.xlabel('Naive Bayes Estimators')
plt.ylabel('Exponential of negative time difference')
plt.plot(data_cv[value_cv == 0, [1]], data_cv[value_cv == 0, [2]], 
         'r.', markersize = 3, label = 'Unaccepted answers')
plt.plot(data_cv[value_cv == 1, [1]], data_cv[value_cv == 1, [2]],
         'b.', markersize = 3, label = 'Accepted answers')
plt.plot(GDA.means_[0, 0], GDA.means_[0, 1], 'k.', markersize = 15)
plt.plot(GDA.means_[1, 0], GDA.means_[1, 1], 'k.', markersize = 15)
plot_ellipse(plotGDA, GDA.means_[0], GDA.covariance_, 'red')
plot_ellipse(plotGDA, GDA.means_[1], GDA.covariance_, 'blue')
plt.xlim(0.5, 1.1)
plt.ylim(0.0, 2.1)
plt.xticks(0.5 + 0.25*np.arange(3))
plt.yticks(0.25*np.arange(7))
nx, ny = 200, 100
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                     np.linspace(y_min, y_max, ny))
Z = GDA.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
               norm=colors.Normalize(0., 1.))
plt.contour(xx, yy, Z, [1-GDA_accuracy], linewidths=2., colors='k')
plt.legend(loc='upper left')
plt.title('Gaussian Discriminant Analysis before normalization')
plt.savefig("GDA.png", dpi=1200)
plt.show()

# Changed GDA

GDA_mod = GaussianDA(data_train[:, [1, 4]], value_train, "Linear")
value_pred = GDA_mod.predict(data_cv[:, [1, 4]])
GDA_mod_accuracy = np.mean(value_pred == value_cv)
print("Gaussian Discriminant Analysis: {0:.2f}%".format(100*GDA_mod_accuracy))

plotGDA_mod = plt.subplot()
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)
plt.xlabel('Naive Bayes Estimators')
plt.ylabel('Quantile of time difference')
plt.plot(data_cv[value_cv == 0, [1]], data_cv[value_cv == 0, [4]], 
         'r.', markersize = 3, label = 'Unaccepted answers')
plt.plot(data_cv[value_cv == 1, [1]], data_cv[value_cv == 1, [4]],
         'b.', markersize = 3, label = 'Accepted answers')
plt.plot(GDA_mod.means_[0, 0], GDA_mod.means_[0, 1], 'k.', markersize = 15)
plt.plot(GDA_mod.means_[1, 0], GDA_mod.means_[1, 1], 'k.', markersize = 15)
plot_ellipse(plotGDA_mod, GDA_mod.means_[0], GDA_mod.covariance_, 'red')
plot_ellipse(plotGDA_mod, GDA_mod.means_[1], GDA_mod.covariance_, 'blue')
plt.xlim(0.49, 1.065)
plt.ylim(0.0, 1.015)
plt.xticks(0.5 + 0.25*np.arange(3))
plt.yticks(0.25*np.arange(5))
nx, ny = 200, 100
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                     np.linspace(y_min, y_max, ny))
Z = GDA_mod.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
               norm=colors.Normalize(0., 1.))
plt.contour(xx, yy, Z, [1-GDA_mod_accuracy], linewidths=2., colors='k')
plt.legend(loc='upper left')
plt.title('Gaussian Discriminant Analysis')
plt.savefig("GDA_mod.png", dpi=1200)
plt.show()


'''
###############################################################################
ROC
###############################################################################
'''


# Classifier

clas_GDA = OneVsRestClassifier(GDA)
clas_LRM = OneVsRestClassifier(LRM)
clas_NN = OneVsRestClassifier(NN)
clas_RFC = OneVsRestClassifier(RFC)
clas_GDA_mod = OneVsRestClassifier(GDA_mod)

# Score

y_score_GDA = clas_GDA.fit(data_train[:, [1, 2]], 
                           value_train).decision_function(data_cv[:, [1, 2]])
y_score_LRM = clas_LRM.fit(data_train[:, [0, 2, 3]], 
                           value_train).decision_function(data_cv[:, [0, 2, 3]])
y_score_NN = clas_NN.fit(data_train[:, [0, 2, 3]], 
                         value_train).decision_function(data_cv[:, [0, 2, 3]])
y_score_RFC = clas_RFC.fit(data_train[:, [1, 2, 3]], 
                           value_train).decision_function(data_cv[:, [1, 2, 3]])
y_score_GDA_mod = clas_GDA_mod.fit(data_train[:, [1, 4]], 
                                   value_train).decision_function(data_cv[:, [1, 4]])

# Roc curve

fpr_GDA, tpr_GDA, _ = roc_curve(value_cv, y_score_GDA)
roc_auc_GDA = auc(fpr_GDA, tpr_GDA)
fpr_LRM, tpr_LRM, _ = roc_curve(value_cv, y_score_LRM)
roc_auc_LRM = auc(fpr_LRM, tpr_LRM)
fpr_NN, tpr_NN, _ = roc_curve(value_cv, y_score_NN)
roc_auc_NN = auc(fpr_NN, tpr_NN)
fpr_RFC, tpr_RFC, _ = roc_curve(value_cv, y_score_RFC)
roc_auc_RFC = auc(fpr_RFC, tpr_RFC)
fpr_GDA_mod, tpr_GDA_mod, _ = roc_curve(value_cv, y_score_GDA_mod)
roc_auc_GDA_mod = auc(fpr_GDA_mod, tpr_GDA_mod)

# Plot

plt.figure()
lw = 2
plt.plot(fpr_GDA, tpr_GDA, color = 'darkorange', lw = lw, 
         label = 'GDA (area = %0.2f)' % roc_auc_GDA)
plt.plot(fpr_LRM, tpr_LRM, color = 'blue', lw = lw, 
         label = 'LRM (area = %0.2f)' % roc_auc_LRM)
plt.plot(fpr_NN, tpr_NN, color = 'red', lw = lw, 
         label = 'NN (area = %0.2f)' % roc_auc_NN)
plt.plot(fpr_RFC, tpr_RFC, color = 'darkgreen', lw = lw, 
         label = 'RFC (area = %0.2f)' % roc_auc_RFC)
plt.plot(fpr_GDA_mod, tpr_GDA_mod, color = 'purple', lw = lw, 
         label = 'Quantile GDA (area = %0.2f)' % roc_auc_GDA_mod)
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig("ROC.png", dpi = 1200)
plt.show()