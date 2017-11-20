import locale
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import re


locale.setlocale(locale.LC_ALL, '')

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
    df['Bodylength_std'] = df['Bodylength_answers'] / df['Bodylength_questions']
    
    # Number of href links
    df['LinksNumber'] = df['Body_answers'].apply(lambda s: s.count('href'))
    
    # Number of code parts
    df['CodeNumber'] = df['Body_answers'].apply(lambda s: s.count('<code>'))
    
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
        # Remove one character words
        if len(word) <= 1:
            continue
        # Remove numbers
        if (word.isdigit()):
            continue
        # Remove words with at least 2 digits
        digit_count = sum(c.isdigit() for c in word)
        if digit_count > 2:
            continue
        # Add word to processed vocabulary otherwise
        voc_proc.append(word)
        
    # Add words from files, such as HTML tags
    for word_file in word_files:
        with open(word_file, 'r') as wf:
            words = [line.rstrip() for line in wf]
        voc_proc += words
            
    # Write to file
    voc_proc = list(set(voc_proc))
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
  
# Separate sets
df_train, df_test = sk.model_selection.train_test_split(df, test_size = 0.01, 
                                                        random_state = 0)

# Train linear regression of Score_std over DeltaT
X_train = np.array([t.total_seconds() for t in df_train.DeltaT])
y_train = df_train.Score_std
logy_train = np.log(y_train[y_train > 0])
lm = sk.linear_model.LinearRegression()
loglm = sk.linear_model.LinearRegression()
lm.fit(X_train.reshape(-1, 1), y_train)
loglm.fit(X_train[y_train > 0].reshape(-1, 1), logy_train)
# Test it
X_test = np.array([t.total_seconds() for t in df_test.DeltaT])
y_test = df_test.Score_std
logy_test = np.log(y_test[y_test > 0])
y_pred = lm.predict(X_test.reshape(-1, 1))
logy_pred = lm.predict(X_test[y_test > 0].reshape(-1, 1))    
# Plot
plt.scatter(X_test, y_test, color = 'blue', label = 'Test data')
plt.plot(X_test, y_pred, color = 'red', label = 'Linear regression')
plt.xlabel('Time elapsed between question and answer (s)')
plt.ylabel('Standardized score')
plt.legend()
#plt.savefig(os.path.join(work_dir, 'Score_VS_DeltaT.png'), dpi = 1200)
plt.show()
plt.scatter(X_test[y_test > 0], logy_test, color = 'blue', label = 'Test data')
plt.plot(X_test[y_test > 0], logy_pred, color = 'red', label = 'Linear regression')
plt.xlabel('Time elapsed between question and answer (s)')
plt.ylabel('Log(Standardized score)')
plt.legend()
#plt.savefig(os.path.join(work_dir, 'log_Score_VS_DeltaT.png'), dpi = 1200)
plt.show()

# Train linear regression of Score_std over Bodylength_std
X_train = df_train.Bodylength_std
y_train = df_train.Score_std
lm = sk.linear_model.LinearRegression()
lm.fit(X_train.values.reshape(-1, 1), y_train)
# Test it
X_test = df_test.Bodylength_std
y_test = df_test.Score_std
y_pred = lm.predict(X_test.values.reshape(-1, 1))
# Plot
plt.scatter(X_test, y_test, color = 'blue', label = 'Test data')
plt.plot(X_test, y_pred, color = 'red', label = 'Linear regression')
plt.xlabel('Standardized bodylength')
plt.ylabel('Standardized score')
plt.legend()
#plt.savefig(os.path.join(work_dir, 'Score_VS_Bodylength.png'), dpi = 1200)
plt.show()

# Train logistic regression of IsAcceptedAnswer over Score_std
X_train = df_train.Score_std
y_train = df_train.IsAcceptedAnswer
lgm = sk.linear_model.LogisticRegression()
lgm.fit(X_train.reshape(-1, 1), y_train)
# Test it
X_test = df_test.Score_std
y_test = df_test.IsAcceptedAnswer
y_pred = lgm.predict(X_test.reshape(-1, 1))
# Plot
plt.scatter(X_test[y_test == True], y_test[y_test == True], 
            color = 'blue', label = 'Accepted answers')
plt.scatter(X_test[y_test == False], y_test[y_test == False], 
            color = 'red', label = 'Accepted answers')
# Plot boundary line
coef = lgm.coef_[0]
xx = np.linspace(0, 8)
yy = -coef[0] / coef[1] * xx - (lgm.intercept_[0]) / coef[1]
plt.plot(xx, yy, color = 'black', label = 'Boundary line')
plt.xlabel('Number of links')
plt.ylabel('Number of code blocks')
plt.legend()
plt.savefig(os.path.join(work_dir, 'AcceptedAnswer_VS_(LinksNumber+CodeNumber).png'), dpi = 1200)
plt.show()


# Train logistic regression of IsAcceptedAnswer over LinksNumber
X_train = df_train[['LinksNumber', 'CodeNumber']]
y_train = df_train.IsAcceptedAnswer
lgm = sk.linear_model.LogisticRegression()
lgm.fit(X_train, y_train)
# Test it
X_test = df_test[['LinksNumber', 'CodeNumber']]
y_test = df_test.IsAcceptedAnswer
y_pred = lgm.predict(X_test)
# Plot
plt.scatter(X_test[y_test == True].LinksNumber, X_test[y_test == True].CodeNumber, 
            color = 'blue', label = 'Accepted answers')
plt.scatter(X_test[y_test == False].LinksNumber, X_test[y_test == False].CodeNumber, 
            color = 'red', label = 'Accepted answers')
# Plot boundary line
coef = lgm.coef_[0]
xx = np.linspace(0, 8)
yy = -coef[0] / coef[1] * xx - (lgm.intercept_[0]) / coef[1]
plt.plot(xx, yy, color = 'black', label = 'Boundary line')
plt.xlabel('Number of links')
plt.ylabel('Number of code blocks')
plt.legend()
plt.savefig(os.path.join(work_dir, 'AcceptedAnswer_VS_(LinksNumber+CodeNumber).png'), dpi = 1200)
plt.show()

   
#if (__name__ == '__main__'):
#    main()
