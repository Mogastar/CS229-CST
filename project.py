import locale
import nltk
import numpy as np
import os
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import re

locale.setlocale(locale.LC_ALL, "English")

# Define directories
data_dir = os.path.join(os.getcwd(), 'Datasets/')
py_dir = os.path.join(os.getcwd(), 'Datasets/pythonquestions/')
r_dir = os.path.join(os.getcwd(), 'Datasets/rquestions/')
# Define a stemmer
stemmer = nltk.SnowballStemmer('english')


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


def process_data(df):
    '''Add features.'''
    
    # DeltaT between question and answer dates
    df['DeltaT'] = df['CreationDate_answers'] - df['CreationDate_questions']
    
    # Length of the answer and question bodies
    df['Bodylength_answers'] = [len(body) for body in df['Body_answers']]
    df['Bodylength_questions'] = [len(body) for body in df['Body_questions']]
    df['Bodylenth_std'] = df['Bodylength_answers'] / df['Bodylength_questions']
    
    # Number of href links
    df['LinksNumber'] = df['Body_answers'].apply(lambda s: s.count('href'))
    
    # Standardized scores
    df['Score_std'] = [float(df['Score_answers'].iloc[i]) / max(1.0, 
                      df['Score_questions'].iloc[i]) for i in range(len(df))]

    print ("Processed data")


def get_voc(df, vocfile):
    '''Get vocabulary from the bodies of answers and write to file.'''
    
    # Get vocabulary
    allwords = []
    for i in range(len(df)):
        allwords += [stemmer.stem(word) for word in re.findall(r"\w+|[^\w\s]", 
                             df['Body_answers'].iloc[i])]
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
        voc_raw = [line.rstrip() for line in vocf]
    voc_raw.sort(cmp = locale.strcoll)
    
    # Return if not processing
    if  not process:
        print ("Read vocabulary")
        return voc_raw
    
    # Process and write
    voc_proc = []
    for word in voc_raw:
        # Remove numbers
        if (word.isdigit()):
            continue
        # Add other things to check for
        if False:
            pass
        voc_proc.append(word)
        
    # Add words from files, such as HTML tags
    for word_file in word_files:
        with open(word_file, 'r') as wf:
            words = [line.rstrip() for line in wf]
        voc_proc += words
            
    # Write to file
    voc_proc.sort(cmp = locale.strcoll)
    with open(vocfile, 'w') as vocf:
        for word in voc_proc:
            vocf.write("%s \n" % word)

    print ("Processed vocabulary")
    return voc_proc
        

def NLP_for_answer(answer, tags):
    ''' Fuck I do not know what I am doing '''

    pass
    

#def main():
#   '''Do all of our shit.'''

# Choose dataset
work_dir = r_dir
# Load data
df, tags = load_data(work_dir)
# Process data
process_data(df)
# Get vocabulary (first time)
#voc = get_voc(df, os.path.join(work_dir, 'Vocabulary.txt'))
# Read dictionary (other times)
voc = process_voc(os.path.join(work_dir, 'Vocabulary.txt'), 
                  [os.path.join(data_dir, 'HTML_tags.txt')],
                  process = True)
  
    
    
    
    
    
#if (__name__ == '__main__'):
#    main()
