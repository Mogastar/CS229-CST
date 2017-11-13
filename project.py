import numpy as np
import os
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import re


# Define directories
pydir = os.path.join(os.getcwd(), 'Datasets/pythonquestions/')
rdir = os.path.join(os.getcwd(), 'Datasets/rquestions/')


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

<<<<<<< HEAD
   
def get_voc(df):
    '''Get our vocabulary.'''
    
    voc = set()
    for i in range(len(df)):
        voc.add(set([word.lower() for word in re.findall(r"\w+|[^\w\s]", 
                     df['Body_answers'].iloc[i], re.UNICODE)]))
    return voc
        
=======
<<<<<<< HEAD
def NLP_for_answer(answer, tags):
    ''' Fuck I do not know what I am doing '''

    pyanswers, 
    

# def main():
=======
>>>>>>> 1f0b6e401492f6152b5e2e84849d035f111319b0
    
#def main():
>>>>>>> 1ddd45f073e8465ac7161e099633ce6e26335135
#   '''Do all of our shit.'''

# Load data
df, tags = load_data(pydir)
# Process data
process_data(df)

    
    
    
    
    
    
    
#if (__name__ == '__main__'):
#    main()
