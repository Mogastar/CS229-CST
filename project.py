import numpy as np
import os
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt


# Define directories
pydir = os.path.join(os.getcwd(), 'Datasets/pythonquestions/')
rdir = os.path.join(os.getcwd(), 'Datasets/pythonquestions/')


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
    
    return (answers, questions, tags)


def main():
    '''Do all of our shit.'''

    # Load data
    ranswers, rquestions, rtags = load_data(rdir)
    
    
    
    
    
    
if (__name__ == '__main__'):
    main()