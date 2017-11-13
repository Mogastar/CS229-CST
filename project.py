import nltk
import numpy as np
import os
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import re


# Define directories
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

def get_voc(df):
    '''Get vocabulary from the bodies of answers.'''
    
    voc = set()
    for i in range(len(df)):
        voc = voc.union(set([stemmer.stem(word) for word in re.findall(r"\w+|[^\w\s]", 
                             df['Body_answers'].iloc[i])]))
        if (i % 10000):
            print ("Done {0}/{1}".format(i+1, len(df)))
    
    print ("Got vocabulary")
    return voc
        

def NLP_for_answer(answer, tags):
    ''' Fuck I do not know what I am doing '''

    pass
    

#def main():
#   '''Do all of our shit.'''

# Load data
df, tags = load_data(r_dir)
# Process data
process_data(df)
# Get and write vocabulary
voc = get_voc(df)
with open(os.path.join(r_dir, 'Vocabulary.txt'), 'w') as vocf:
    for word in voc:
        vocf.write("%s \n" % word)
# Do other shit



    
    
    
    
    
    
    
#if (__name__ == '__main__'):
#    main()
