import locale
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import re
import pickle
import glob
import scipy
import itertools
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize

# os.chdir('E:\Stanford\Courses\CS 229\Project\CS229-CST')

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
        
        
def aggregate_design(work_dir, shape):
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
            
    # Construct sparse matrix
    sparse = scipy.sparse.coo_matrix((val, (row, col)), shape = shape)
    
    print ("Aggregated sparse matrix.")
    return sparse
        

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


'''
###############################################################################
Main
###############################################################################
'''


# Choose dataset
work_dir = r_dir
# Load data
df0, tags = load_data(work_dir)
# Process data
df = process_data(df0, threshold = 5)
del df0
# Get vocabulary (first time)
#voc = get_voc(df, os.path.join(work_dir, 'Vocabulary.txt'))
# Read dictionary (other times)
word_files = [os.path.join(data_dir, 'HTML_tags.txt')]
voc_list = process_voc(os.path.join(work_dir, 'Vocabulary.txt'), 
                       word_files, process = False)   
voc = dict(itertools.izip(voc_list, range(len(voc_list))))
     
# Get design matrix (first time)
#for i in range((len(df) + 9999) / 10000):
#    start = 10000 * i
#    end = min(10000 * (i+1), len(df))
#    get_design(df, voc, start, end, work_dir, word_files)
# Aggregate design matrix in sparse format
sparse_X = aggregate_design(work_dir, (len(df), len(voc)))
#X = sparse_X.todense()

# Separate sets
#df, df_test = sk.model_selection.train_test_split(df, test_size = 0.01)
#df_train, df_cv = sk.model_selection.train_test_split(df, test_size = 0.1)

