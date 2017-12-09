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


def Reg_nS_Deltat(score, time, nbins = 1000):
    '''
    Running regression on DeltaT and Score_std
    '''
    time = time.dt.total_seconds()
    time_sorted = time.sort_values()
    time_sorted = time_sorted[time_sorted >= 0]
    ind_sorted  = time_sorted.index.values # indices as numpy array
    score_sorted = score.reindex(time_sorted.index)
    score_sorted = score_sorted.values     # sorted scores as numpy array
    
    score_quantile = np.array_split(score_sorted, nbins)
    ind_quantile   = np.array_split(ind_sorted, nbins)
    
    time_picks = np.zeros(nbins);
    score_picks = np.zeros(nbins);

    for i in range(nbins):
        score_picks[i] = np.max(score_quantile[i])
        indices        = ind_quantile[i]
        ind_max        = indices[np.argmax(score_quantile[i])]
        time_picks[i]  = time_sorted.loc[ind_max]
    
    score_picks[score_picks < 1e-5] = 1e-5    
    score_picks_log = np.log(score_picks).reshape(len(score_picks), 1)
    time_picks = time_picks.reshape(len(time_picks), 1)
    
    reg = linear_model.LinearRegression()
    reg.fit(time_picks, score_picks_log)
    return reg, score_picks_log, time_picks


def separate(val, row, col, y, test_size, seed = 0):
    '''
    Separate a sparse matrix in COO format and a vector y into 2 sets
        - set 0 of size (1 - test_size) %
        - set 1 of size test_size %.
    '''
    
    # Separate indices
    ind = np.arange(len(y))
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
    y0 = y[ind0]
    y1 = y[ind1]
    
    return val0, row0, col0, y0, val1, row1, col1, y1


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
y = np.array(df['IsAcceptedAnswer'], dtype = int)

# Separate datasets
val_temp, row_temp, col_temp, y_temp, val_test, row_test, col_test, y_test = \
    separate(val, row, col, y, test_size = 0.01)
val_train, row_train, col_train, y_train, val_cv, row_cv, col_cv, y_cv = \
    separate(val_temp, row_temp, col_temp, y_temp, test_size = 0.1)
del val_temp, row_temp, col_temp, y_temp

# Construct sparse matrices
X_train = scipy.sparse.coo_matrix((val_train, (row_train, col_train)),
                                  shape = (len(y_train), len(voc)))
del val_train, row_train, col_train
X_cv = scipy.sparse.coo_matrix((val_cv, (row_cv, col_cv)),
                                  shape = (len(y_cv), len(voc)))
del val_cv, row_cv, col_cv
X_test = scipy.sparse.coo_matrix((val_test, (row_test, col_test)),
                                 shape = (len(y_test), len(voc)))
del val_test, row_test, col_test


'''
###############################################################################
Tests
###############################################################################
'''


# Tests

MNB = MultinomialNB()
MNB.fit(X_train, y_train)
y_MNB = MNB.predict(X_cv)
accuracy_MNB = np.mean(y_MNB == y_cv)
print(accuracy_MNB)

BNB = BernoulliNB()
BNB.fit(X_train, y_train)
y_BNB = BNB.predict(X_cv)
accuracy_BNB = np.mean(y_BNB == y_cv)
print(accuracy_BNB)


TRA = MNB.predict(X_train)
accuracy = np.mean(TRA == y_train)
print(accuracy)

score = df['Score_std']
time = df['DeltaT']

reg, score_log, time = Reg_nS_Deltat(df['Score_std'], df['DeltaT'], 5000)
time_days = np.to_datetime(time)
pred = np.exp(reg.predict(time))
plt.plot(df['DeltaT'], df['Score_std'], '.')
plt.plot(time, pred, 'r-')

