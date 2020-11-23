#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Initialize OK
from client.api.notebook import Notebook
ok = Notebook('proj2.ok')


# # Project 2: Spam/Ham Classification
# ## Feature Engineering, Logistic Regression, Cross Validation
# ## Due Date: Tuesday 8/6/19, 11:59PM
# 
# **Collaboration Policy**
# 
# Data science is a collaborative activity. While you may talk with others about
# the project, we ask that you **write your solutions individually**. If you do
# discuss the assignments with others please **include their names** at the top
# of your notebook.

# **Collaborators**: *list collaborators here*

# ## This Assignment
# In this project, you will use what you've learned in class to create a classifier that can distinguish spam (junk or commercial or bulk) emails from ham (non-spam) emails. In addition to providing some skeleton code to fill in, we will evaluate your work based on your model's accuracy and your written responses in this notebook.
# 
# After this project, you should feel comfortable with the following:
# 
# - Feature engineering with text data
# - Using sklearn libraries to process data and fit models
# - Validating the performance of your model and minimizing overfitting
# - Generating and analyzing precision-recall curves
# 
# ## Warning
# We've tried our best to filter the data for anything blatantly offensive as best as we can, but unfortunately there may still be some examples you may find in poor taste. If you encounter these examples and believe it is inappropriate for students, please let a TA know and we will try to remove it for future semesters. Thanks for your understanding!

# ## Score Breakdown
# Question | Points
# --- | ---
# 1a | 1
# 1b | 1
# 1c | 2
# 2 | 3
# 3a | 2
# 3b | 2
# 4 | 2
# 5 | 2
# 6a | 1
# 6b | 1
# 6c | 2
# 6d | 2
# 6e | 1
# 6f | 3
# 7 | 6
# 8 | 6
# 9 | 3
# 10 | 15
# Total | 55

# # Part I - Initial Analysis

# In[2]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style = "whitegrid", 
        color_codes = True,
        font_scale = 1.5)
import sklearn
import sklearn.datasets
import sklearn.linear_model


# ### Loading in the Data
# 
# In email classification, our goal is to classify emails as spam or not spam (referred to as "ham") using features generated from the text in the email. 
# 
# The dataset consists of email messages and their labels (0 for ham, 1 for spam). Your labeled training dataset contains 8348 labeled examples, and the test set contains 1000 unlabeled examples.
# 
# Run the following cells to load in the data into DataFrames.
# 
# The `train` DataFrame contains labeled data that you will use to train your model. It contains four columns:
# 
# 1. `id`: An identifier for the training example
# 1. `subject`: The subject of the email
# 1. `email`: The text of the email
# 1. `spam`: 1 if the email is spam, 0 if the email is ham (not spam)
# 
# The `test` DataFrame contains 1000 unlabeled emails. You will predict labels for these emails and submit your predictions to Kaggle for evaluation.

# In[3]:


from utils import fetch_and_cache_gdrive
fetch_and_cache_gdrive('1SCASpLZFKCp2zek-toR3xeKX3DZnBSyp', 'train.csv')
fetch_and_cache_gdrive('1ZDFo9OTF96B5GP2Nzn8P8-AL7CTQXmC0', 'test.csv')

original_training_data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Convert the emails to lower case as a first step to processing the text
original_training_data['email'] = original_training_data['email'].str.lower()
test['email'] = test['email'].str.lower()

original_training_data


# In[ ]:





# ### Question 1a
# First, let's check if our data contains any missing values. Fill in the cell below to print the number of NaN values in each column. If there are NaN values, replace them with appropriate filler values (i.e., NaN values in the `subject` or `email` columns should be replaced with empty strings). Print the number of NaN values in each column after this modification to verify that there are no NaN values left.
# 
# Note that while there are no NaN values in the `spam` column, we should be careful when replacing NaN labels. Doing so without consideration may introduce significant bias into our model when fitting.
# 
# *The provided test checks that there are no missing values in your dataset.*
# 
# <!--
# BEGIN QUESTION
# name: q1a
# points: 1
# -->

# In[4]:


original_training_data=original_training_data.fillna('')
print(original_training_data['subject'].isna().sum())
print(original_training_data['email'].isna().sum())
print(original_training_data['id'].isna().sum())
print(original_training_data['spam'].isna().sum())


# In[5]:


ok.grade("q1a");


# ### Question 1b
# 
# In the cell below, print the text of the first ham and the first spam email in the original training set.
# 
# *The provided tests just ensure that you have assigned `first_ham` and `first_spam` to rows in the data, but only the hidden tests check that you selected the correct observations.*
# 
# <!--
# BEGIN QUESTION
# name: q1b
# points: 1
# -->

# In[6]:



first_ham = original_training_data[original_training_data['spam']==0].iloc[0][1]
first_spam = original_training_data[original_training_data['spam']==1].iloc[0][1]
print(first_ham)
print(first_spam)


# In[7]:


ok.grade("q1b");


# ### Question 1c
# 
# Discuss one thing you notice that is different between the two emails that might relate to the identification of spam.
# 
# <!--
# BEGIN QUESTION
# name: q1c
# manual: True
# points: 2
# -->
# <!-- EXPORT TO PDF -->

# For spam, the email column contains <> symbol, but there isn't in ham.

# ## Training Validation Split
# The training data we downloaded is all the data we have available for both training models and **validating** the models that we train.  We therefore need to split the training data into separate training and validation datsets.  You will need this **validation data** to assess the performance of your classifier once you are finished training. Note that we set the seed (random_state) to 42. This will produce a pseudo-random sequence of random numbers that is the same for every student. Do not modify this in the following questions, as our tests depend on this random seed.

# In[8]:


from sklearn.model_selection import train_test_split

train, val = train_test_split(original_training_data, test_size=0.1, random_state=42)


# # Basic Feature Engineering
# 
# We would like to take the text of an email and predict whether the email is ham or spam. This is a *classification* problem, so we can use logistic regression to train a classifier. Recall that to train an logistic regression model we need a numeric feature matrix $X$ and a vector of corresponding binary labels $y$.  Unfortunately, our data are text, not numbers. To address this, we can create numeric features derived from the email text and use those features for logistic regression.
# 
# Each row of $X$ is an email. Each column of $X$ contains one feature for all the emails. We'll guide you through creating a simple feature, and you'll create more interesting ones when you are trying to increase your accuracy.

# In[9]:


#len(train['email'].str.contains('drug'))
train['email'].str.contains('body').value_counts()
#'drug', 'bank', 'prescription', 'memo', 'private'


# ### Question 2
# 
# Create a function called `words_in_texts` that takes in a list of `words` and a pandas Series of email `texts`. It should output a 2-dimensional NumPy array containing one row for each email text. The row should contain either a 0 or a 1 for each word in the list: 0 if the word doesn't appear in the text and 1 if the word does. For example:
# 
# ```
# >>> words_in_texts(['hello', 'bye', 'world'], 
#                    pd.Series(['hello', 'hello worldhello']))
# 
# array([[1, 0, 0],
#        [1, 0, 1]])
# ```
# 
# *The provided tests make sure that your function works correctly, so that you can use it for future questions.*
# 
# <!--
# BEGIN QUESTION
# name: q2
# points: 3
# -->

# In[10]:


def words_in_texts(words, texts):
    '''
    Args:
        words (list-like): words to find
        texts (Series): strings to search in
    
    Returns:
        NumPy array of 0s and 1s with shape (n, p) where n is the
        number of texts and p is the number of words.
    '''
    
    df = texts.str.contains(words[0]).to_frame().astype(int)
    words=words[1:]
    for i in words:
        #newcolumn=texts.str.contains(i).to_frame()
        df[i]=texts.str.contains(i).to_frame().astype(int)

        #df=df.assign(j=texts.str.contains(i).to_frame())
    
    indicator_array = df.to_numpy()

    return indicator_array
#words_in_texts(['hello', 'bye', 'world'], pd.Series(['hello', 'hello worldhello']))


# In[11]:


ok.grade("q2");


# In[12]:


train[train['email'].str.contains('not')]['spam'].value_counts()
#['spam'].value_counts()


# # Basic EDA
# 
# We need to identify some features that allow us to distinguish spam emails from ham emails. One idea is to compare the distribution of a single feature in spam emails to the distribution of the same feature in ham emails. If the feature is itself a binary indicator, such as whether a certain word occurs in the text, this amounts to comparing the proportion of spam emails with the word to the proportion of ham emails with the word.
# 

# The following plot (which was created using `sns.barplot`) compares the proportion of emails in each class containing a particular set of words. 
# 
# ![training conditional proportions](./images/training_conditional_proportions.png "Class Conditional Proportions")
# 
# Hint:
# - You can use DataFrame's `.melt` method to "unpivot" a DataFrame. See the following code cell for an example.

# In[13]:


from IPython.display import display, Markdown
df = pd.DataFrame({
    'word_1': [1, 0, 1, 0],
    'word_2': [0, 1, 0, 1],
    'type': ['spam', 'ham', 'ham', 'ham']
})
display(Markdown("> Our Original DataFrame has some words column and a type column. You can think of each row is a sentence, and the value of 1 or 0 indicates the number of occurances of the word in this sentence."))
display(df);
display(Markdown("> `melt` will turn columns into variale, notice how `word_1` and `word_2` become `variable`, their values are stoed in the value column"))
display(df.melt("type"))


# ### Question 3a
# 
# Create a bar chart like the one above comparing the proportion of spam and ham emails containing certain words. Choose a set of words that are different from the ones above, but also have different proportions for the two classes. Make sure to only consider emails from `train`.
# 
# <!--
# BEGIN QUESTION
# name: q3a
# manual: True
# format: image
# points: 2
# -->
# <!-- EXPORT TO PDF format:image -->

# In[14]:


train=train.reset_index(drop=True) # We must do this in order to preserve the ordering of emails to labels for words_in_texts
word_pd=pd.DataFrame(words_in_texts(['like','head','not','news','thanks','help'],train['email'] )).rename(columns={0: "like", 1: "head",2:'not',3:'news',

                                                                                                   4: "thanks", 5: "help"}).assign(type=train['spam']).replace({'type': {0: 'ham', 1: 'spam'}}).melt("type")
#word_pd['value']
sns.barplot(x='variable',y='value',hue='type',data=word_pd,ci=None)
plt.xlabel('words')
plt.ylabel('Proportions of Emails')
plt.ylim((0, 1))
plt.title('Frequency of Words in Spam/Ham Emails')
#original_training_data
#...


# When the feature is binary, it makes sense to compare its proportions across classes (as in the previous question). Otherwise, if the feature can take on numeric values, we can compare the distributions of these values for different classes. 
# 
# ![training conditional densities](./images/training_conditional_densities2.png "Class Conditional Densities")
# 

# ### Question 3b
# 
# Create a *class conditional density plot* like the one above (using `sns.distplot`), comparing the distribution of the length of spam emails to the distribution of the length of ham emails in the training set. Set the x-axis limit from 0 to 50000.
# 
# <!--
# BEGIN QUESTION
# name: q3b
# manual: True
# format: image
# points: 2
# -->
# <!-- EXPORT TO PDF format:image -->

# In[15]:


train_n1=train
train_n1['Length of email body']=train_n1['email'].str.len()
sns.distplot(train_n1[train_n1['spam']==0]['Length of email body'],hist=None,label='Ham')
sns.distplot(train_n1[train_n1['spam']==1]['Length of email body'],hist=None,label='Spam')
plt.xlim((0, 50000))
plt.xlabel('Distribution')


# In[16]:


test


# # Basic Classification
# 
# Notice that the output of `words_in_texts(words, train['email'])` is a numeric matrix containing features for each email. This means we can use it directly to train a classifier!

# ### Question 4
# 
# We've given you 5 words that might be useful as features to distinguish spam/ham emails. Use these words as well as the `train` DataFrame to create two NumPy arrays: `X_train` and `Y_train`.
# 
# `X_train` should be a matrix of 0s and 1s created by using your `words_in_texts` function on all the emails in the training set.
# 
# `Y_train` should be a vector of the correct labels for each email in the training set.
# 
# *The provided tests check that the dimensions of your feature matrix (X) are correct, and that your features and labels are binary (i.e. consists of 0 and 1, no other values). It does not check that your function is correct; that was verified in a previous question.*
# <!--
# BEGIN QUESTION
# name: q4
# points: 2
# -->

# In[17]:


some_words = ['drug', 'bank', 'prescription', 'memo', 'private']

X_train = words_in_texts(some_words, train['email'])
Y_train = train['spam']

X_train[:5], Y_train[:5]


# In[18]:


ok.grade("q4");


# In[19]:


Y_train 


# ### Question 5
# 
# Now we have matrices we can give to scikit-learn! Using the [`LogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier, train a logistic regression model using `X_train` and `Y_train`. Then, output the accuracy of the model (on the training data) in the cell below. You should get an accuracy around 0.75.
# 
# *The provided test checks that you initialized your logistic regression model correctly.*
# 
# <!--
# BEGIN QUESTION
# name: q5
# points: 2
# -->

# In[20]:


from sklearn.linear_model import LogisticRegression


model = sklearn.linear_model.LogisticRegression(fit_intercept=True)
model.fit(X_train, Y_train)

training_accuracy = ((model.predict(X_train)==Y_train).sum())/len(X_train)
print("Training Accuracy: ", training_accuracy)


# In[21]:


ok.grade("q5");


# ## Evaluating Classifiers

# That doesn't seem too shabby! But the classifier you made above isn't as good as this might lead us to believe. First, we are evaluating accuracy on the training set, which may lead to a misleading accuracy measure, especially if we used the training set to identify discriminative features. In future parts of this analysis, it will be safer to hold out some of our data for model validation and comparison.
# 
# Presumably, our classifier will be used for **filtering**, i.e. preventing messages labeled `spam` from reaching someone's inbox. There are two kinds of errors we can make:
# - False positive (FP): a ham email gets flagged as spam and filtered out of the inbox.
# - False negative (FN): a spam email gets mislabeled as ham and ends up in the inbox.
# 
# These definitions depend both on the true labels and the predicted labels. False positives and false negatives may be of differing importance, leading us to consider more ways of evaluating a classifier, in addition to overall accuracy:
# 
# **Precision** measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FP}}$ of emails flagged as spam that are actually spam.
# 
# **Recall** measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FN}}$ of spam emails that were correctly flagged as spam. 
# 
# **False-alarm rate** measures the proportion $\frac{\text{FP}}{\text{FP} + \text{TN}}$ of ham emails that were incorrectly flagged as spam. 
# 
# The following image might help:
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/700px-Precisionrecall.svg.png" width="500px">
# 
# Note that a true positive (TP) is a spam email that is classified as spam, and a true negative (TN) is a ham email that is classified as ham.

# ### Question 6a
# 
# Suppose we have a classifier `zero_predictor` that always predicts 0 (never predicts positive). How many false positives and false negatives would this classifier have if it were evaluated on the training set and its results were compared to `Y_train`? Fill in the variables below (answers can be hard-coded):
# 
# *Tests in Question 6 only check that you have assigned appropriate types of values to each response variable, but do not check that your answers are correct.*
# 
# <!--
# BEGIN QUESTION
# name: q6a
# points: 1
# -->

# In[22]:


zero_predictor_fp = 0
zero_predictor_fn = sum((Y_train == 1))


# In[23]:


ok.grade("q6a");


# ### Question 6b
# 
# What are the accuracy and recall of `zero_predictor` (classifies every email as ham) on the training set? Do NOT use any `sklearn` functions.
# 
# <!--
# BEGIN QUESTION
# name: q6b
# points: 1
# -->

# In[24]:


zero_predictor_acc = sum((Y_train == 0))/len(X_train)
zero_predictor_recall = 0
zero_predictor_recall


# In[25]:


ok.grade("q6b");


# ### Question 6c
# 
# Provide brief explanations of the results from 6a and 6b. Why do we observe each of these values (FP, FN, accuracy, recall)?
# 
# <!--
# BEGIN QUESTION
# name: q6c
# manual: True
# points: 2
# -->
# <!-- EXPORT TO PDF -->

# We use all these values to evaluate this classifier. If we just use accuracy and FP, we would get the result that the classifier is good, but that's not true.

# ### Question 6d
# 
# Compute the precision, recall, and false-alarm rate of the `LogisticRegression` classifier created and trained in Question 5. Do NOT use any `sklearn` functions.
# 
# <!--
# BEGIN QUESTION
# name: q6d
# points: 2
# -->

# In[26]:


logistic_predictor_precision = sum((Y_train == 1) & (model.predict(X_train) == 1))/sum(model.predict(X_train) == 1)
logistic_predictor_recall = sum((Y_train == 1) & (model.predict(X_train) == 1))/sum(Y_train == 1)
logistic_predictor_far = sum((Y_train == 0) & (model.predict(X_train) == 1))/sum(Y_train == 0)
#sum((Y_train == 0) & (model.predict(X_train) == 1))
#sum((Y_train == 1) & (model.predict(X_train) == 0))


# In[27]:


ok.grade("q6d");


# ### Question 6e
# 
# Are there more false positives or false negatives when using the logistic regression classifier from Question 5?
# 
# <!--
# BEGIN QUESTION
# name: q6e
# manual: True
# points: 1
# -->
# <!-- EXPORT TO PDF -->

# There are more false negatives.

# ### Question 6f
# 
# 1. Our logistic regression classifier got 75.6% prediction accuracy (number of correct predictions / total). How does this compare with predicting 0 for every email?
# 1. Given the word features we gave you above, name one reason this classifier is performing poorly. Hint: Think about how prevalent these words are in the email set.
# 1. Which of these two classifiers would you prefer for a spam filter and why? Describe your reasoning and relate it to at least one of the evaluation metrics you have computed so far.
# 
# <!--
# BEGIN QUESTION
# name: q6f
# manual: True
# points: 3
# -->
# <!-- EXPORT TO PDF -->

# 1.Predicting p for every email has a little bit lower accuracy:0.7447091707706642.
# 
# 2.The words cannot distinguish between spam and ham. (The prevalence are almost the same)
# 
# 3.Their accuracy are similar, logistic classifier got higher recall (0.11>0). Logistic classifier is better.

# # Part II - Moving Forward
# 
# With this in mind, it is now your task to make the spam filter more accurate. In order to get full credit on the accuracy part of this assignment, you must get at least **88%** accuracy on the test set. To see your accuracy on the test set, you will use your classifier to predict every email in the `test` DataFrame and upload your predictions to Kaggle.
# 
# **Kaggle limits you to four submissions per day**. This means you should start early so you have time if needed to refine your model. You will be able to see your accuracy on the entire set when submitting to Kaggle (the accuracy that will determine your score for question 10).
# 
# Here are some ideas for improving your model:
# 
# 1. Finding better features based on the email text. Some example features are:
#     1. Number of characters in the subject / body
#     1. Number of words in the subject / body
#     1. Use of punctuation (e.g., how many '!' were there?)
#     1. Number / percentage of capital letters 
#     1. Whether the email is a reply to an earlier email or a forwarded email
# 1. Finding better words to use as features. Which words are the best at distinguishing emails? This requires digging into the email text itself. 
# 1. Better data processing. For example, many emails contain HTML as well as text. You can consider extracting out the text from the HTML to help you find better words. Or, you can match HTML tags themselves, or even some combination of the two.
# 1. Model selection. You can adjust parameters of your model (e.g. the regularization parameter) to achieve higher accuracy. Recall that you should use cross-validation to do feature and model selection properly! Otherwise, you will likely overfit to your training data.
# 
# ou may use whatever method you prefer in order to create features, but **you are not allowed to import any external feature extraction libraries**. In addition, **you are only allowed to train logistic regression models**. No random forests, k-nearest-neighbors, neural nets, etc.
# 
# We have not provided any code to do this, so feel free to create as many cells as you need in order to tackle this task. However, answering questions 7, 8, and 9 should help guide you.
# 
# ---
# 
# **Note:** *You should use the **validation data** to evaluate your model and get a better sense of how it will perform on the Kaggle evaluation.*
# 
# ---

# In[28]:


def remove_html(data):
    data['email'] = data['email'].str.replace('<html>', '')
    return data


# In[29]:


def choose_words(data):
    data=data.reset_index(drop=True)
    some_words = ["size", "font", "market", "please"]
    words = np.array(words_in_texts(some_words, data['email']))
    data=data.join(pd.DataFrame(words))
    data=data.rename(columns={0:'size',1:'font',2:'market',3:'please'})
    return data


# In[30]:


def num_of_word(data):
    data['num_word']=data['email'].str.split().str.len().sort_index(0)
    return data


# In[31]:


def num_character_mail(data):
    data['num_char_mail']=data['email'].str.len().sort_index(0)
    return data


# In[32]:


def num_of_quesmark(data):
    data['num_quesmark']=data['email'].str.count('!')
    return data


# In[33]:


def num_of_bookmark1(data):
    data['num_bookmark1']=data['email'].str.count('#')
    return data


# In[34]:


def num_of_bookmark(data):
    data['num_bookmark']=data['email'].str.count(':')
    return data


# In[35]:


def occ_uppercase(data):
    data['uppercase']=data['subject'].str.count(r'[A-Z]')
    return data


# In[36]:


def select_columns(data, *columns):
    return data.loc[:, columns]


# In[37]:


train_zy=train.reset_index(drop=True) # We must do this in order to preserve the ordering of emails to labels for words_in_texts
#train_zy=remove_html(train_zy)
df_zy=pd.DataFrame(words_in_texts(['font','size','market','please'],train['email'] )).rename(columns={0: "font", 1:'size',2:'market',3:'please'}).assign(spam=train['spam'])

df_zy=df_zy.assign(num_word=train['email'].str.split().str.len().sort_index(0))
#df_zy=df_zy.assign(num_word_sub=train['subject'].str.split().str.len().sort_index(0))
#df_zy=df_zy.assign(num_char_mail=train['email'].str.len().sort_index(0))
#df_zy=df_zy.assign(num_quesmark=train['email'].str.count('!'))
#df_zy=df_zy.assign(num_bookmark1=train['email'].str.count('#'))
#df_zy=df_zy.assign(num_bookmark=train['email'].str.count(':'))
#df_zy=df_zy.assign(uppercase=train['subject'].str.count(r'[A-Z]'))
#df_zy=df_zy.dropna()


# In[92]:


from sklearn.linear_model import LogisticRegression 
def process_data_fm(data):
    data = (
        data
        .pipe(choose_words)
        .pipe(remove_html)
        .pipe(num_of_word)
        .pipe(num_character_mail)
        .pipe(num_of_quesmark)
        .pipe(num_of_bookmark1) 
        .pipe(num_of_bookmark) 
        .pipe(occ_uppercase) 
        .pipe(select_columns,
             'spam',
             'font',
             'size',
             'market',
             'please',
             'num_char_mail',
             'uppercase',
             'num_word',
             'num_quesmark',
             'num_bookmark',
             'num_bookmark1'
             )
     )
        
    data=data.fillna(0)
    X = data[['font','size','market','please','num_char_mail','uppercase','num_word','num_quesmark','num_bookmark','num_bookmark1']]
    y = data['spam']
    return X,y


# In[93]:


def process_data_fm_x(data):
    data = (
        data
        .pipe(choose_words)
        .pipe(remove_html)
        .pipe(num_of_word)
        .pipe(num_character_mail)
        .pipe(num_of_quesmark)
        .pipe(num_of_bookmark1) 
        .pipe(num_of_bookmark) 
        .pipe(occ_uppercase) 
        .pipe(select_columns,
             'font',
             'size',
             'market',
             'please',
             'num_char_mail',
             'uppercase',
             'num_word',
             'num_quesmark',
             'num_bookmark',
             'num_bookmark1'
             )
     )
        
    data=data.fillna(0)
    X = data[['font','size','market','please','num_char_mail','uppercase','num_word','num_quesmark','num_bookmark','num_bookmark1']]
   
    return X


# In[70]:


model_zy = LogisticRegression(fit_intercept=True,penalty='l2',C=0.2)
train, test1 = train_test_split(original_training_data, test_size=0.2)
x_train, y_train = process_data_fm(train)

x_test, y_test = process_data_fm(test1)
print(f"Training Data Size: {len(x_train)}")
print(f"Test Data Size: {len(x_test)}")


# In[83]:


X_test = process_data_fm_x(test)


# In[86]:


model_zy.fit(x_train, y_train)

train_accuracy = ((model_zy.predict(x_train)==y_train).sum())/len(x_train)
print("Train Accuracy: ", train_accuracy)
test_accuracy = ((model_zy.predict(x_test)==y_test).sum())/len(x_test)
print("Test Accuracy: ", test_accuracy)


# In[73]:


#from sklearn.model_selection import train_test_split
#X_zy=df_zy[['font','size','market','please','num_char_mail','uppercase','num_word','num_quesmark','num_bookmark','num_bookmark1']]
#Y_zy=df_zy['spam']

#X_train_zy, X_test_zy,Y_train_zy,Y_test_zy = train_test_split(X_zy,Y_zy, test_size=0.25, random_state=43)
#X_train_zy = df_zy[['body','html','market','please','num_word','num_char_mail','num_bookmark']]
#Y_train_zy = df_zy['spam']
#print(f"Training Data Size: {len(X_train_zy)}")
#print(f"Test Data Size: {len(X_test_zy)}")


# In[ ]:





# In[74]:


#model_zy = sklearn.linear_model.LogisticRegression(fit_intercept=True,penalty='l2',C=1)
#model_zy.fit(X_train_zy, Y_train_zy)

#train_accuracy = ((model_zy.predict(X_train_zy)==Y_train_zy).sum())/len(X_train_zy)
#print("Train Accuracy: ", train_accuracy)
#test_accuracy = ((model_zy.predict(X_test_zy)==Y_test_zy).sum())/len(X_test_zy)
#print("Test Accuracy: ", test_accuracy)


# ### Question 7: Feature/Model Selection Process
# 
# In this following cell, describe the process of improving your model. You should use at least 2-3 sentences each to address the follow questions:
# 
# 1. How did you find better features for your model?
# 2. What did you try that worked / didn't work?
# 3. What was surprising in your search for good features?
# 
# <!--
# BEGIN QUESTION
# name: q7
# manual: True
# points: 6
# -->
# <!-- EXPORT TO PDF -->

# 1.I tried every feature given, try to either add feature, or substitute another feature and find the one with highest training acuracy. 
# 
# 2.I try the number of words in text and it works. I try number of "<" in text, it doesn't work well.
# 
# 3.Sometimes we do expect that a specific feature (word) is influencial in distinguishing spam and ham, but I am surprised that it improves a lot.

# ### Question 8: EDA
# 
# In the cell below, show a visualization that you used to select features for your model. Include both
# 
# 1. A plot showing something meaningful about the data that helped you during feature / model selection.
# 2. 2-3 sentences describing what you plotted and what its implications are for your features.
# 
# Feel to create as many plots as you want in your process of feature selection, but select one for the response cell below.
# 
# **You should not just produce an identical visualization to question 3.** Specifically, don't show us a bar chart of proportions, or a one-dimensional class-conditional density plot. Any other plot is acceptable, as long as it comes with thoughtful commentary. Here are some ideas:
# 
# 1. Consider the correlation between multiple features (look up correlation plots and `sns.heatmap`). 
# 1. Try to show redundancy in a group of features (e.g. `body` and `html` might co-occur relatively frequently, or you might be able to design a feature that captures all html tags and compare it to these). 
# 1. Visualize which words have high or low values for some useful statistic.
# 1. Visually depict whether spam emails tend to be wordier (in some sense) than ham emails.

# Generate your visualization in the cell below and provide your description in a comment.
# 
# <!--
# BEGIN QUESTION
# name: q8
# manual: True
# format: image
# points: 6
# -->
# <!-- EXPORT TO PDF format:image -->

# In[75]:


# Write your description (2-3 sentences) as a comment here:
# I plot the box plot on (distribution) pf number of words of spam and ham, to compare whether spam emails tend to be wordier
# than ham emails. The visualization shows that the median of spam > median of ham. The difference is obvious.
# So this feature could be used in as feature.
#

# Write the code to generate your visualization here:
new_df=df_zy[['num_word','spam']]
sns.boxplot(x='spam',y='num_word',data=new_df)
plt.ylim((0, 800))


# ### Question 9: ROC Curve
# 
# In most cases we won't be able to get no false positives and no false negatives, so we have to compromise. For example, in the case of cancer screenings, false negatives are comparatively worse than false positives — a false negative means that a patient might not discover a disease until it's too late to treat, while a false positive means that a patient will probably have to take another screening.
# 
# Recall that logistic regression calculates the probability that an example belongs to a certain class. Then, to classify an example we say that an email is spam if our classifier gives it $\ge 0.5$ probability of being spam. However, *we can adjust that cutoff*: we can say that an email is spam only if our classifier gives it $\ge 0.7$ probability of being spam, for example. This is how we can trade off false positives and false negatives.
# 
# The ROC curve shows this trade off for each possible cutoff probability. In the cell below, plot an ROC curve for your final classifier (the one you use to make predictions for Kaggle). Refer to the Lecture 20 notebook to see how to plot an ROC curve.
# 
# 
# 
# <!--
# BEGIN QUESTION
# name: q9
# manual: True
# points: 3
# -->
# <!-- EXPORT TO PDF -->

# In[76]:


from sklearn.metrics import roc_curve

# Note that you'll want to use the .predict_proba(...) method for your classifier
# instead of .predict(...) so you get probabilities, not classes


fpr, tpr, thresholds = roc_curve(y_train, model_zy.predict_proba(x_train)[:, 1])
with sns.axes_style("white"):
    plt.plot(fpr, tpr)

sns.despine()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate');


# In[97]:


X_test2 = process_data_fm_x(test)
final_answer=model_zy.predict(X_test2)


# In[98]:


x_train.head()


# # Question 10: Submitting to Kaggle
# 
# The following code will write your predictions on the test dataset to a CSV, which you can submit to Kaggle. You may need to modify it to suit your needs.
# 
# Save your predictions in a 1-dimensional array called `test_predictions`. *Even if you are not submitting to Kaggle, please make sure you've saved your predictions to `test_predictions` as this is how your score for this question will be determined.*
# 
# Remember that if you've performed transformations or featurization on the training data, you must also perform the same transformations on the test data in order to make predictions. For example, if you've created features for the words "drug" and "money" on the training data, you must also extract the same features in order to use scikit-learn's `.predict(...)` method.
# 
# You should submit your CSV files to https://www.kaggle.com/c/ds100su19
# 
# *The provided tests check that your predictions are in the correct format, but you must submit to Kaggle to evaluate your classifier accuracy.*
# 
# <!--
# BEGIN QUESTION
# name: q10
# points: 15
# -->

# In[99]:


#test_zy=test.reset_index(drop=True)
#test_zy=pd.DataFrame(words_in_texts(['font','html','market','please'],test['email'] )).rename(columns={0: "body", 1:'html',2:'market',3:'please'})
#test_zy=test_zy.assign(num_word=test['email'].str.split().str.len().sort_index(0))
#test_zy=test_zy.assign(num_char_mail=test['email'].str.len().sort_index(0))
#test_zy=test_zy.assign(num_quesmark=test['email'].str.count('!'))
#test_zy=test_zy.assign(num_bookmark1=test['email'].str.count('#'))
#test_zy=test_zy.assign(num_bookmark=test['email'].str.count(':'))
#test_zy=test_zy.assign(uppercase=test['email'].str.count(r'[A-Z]'))
#test_zy=test_zy.dropna()
#test_predictions = model_zy.predict(test_zy[['body','html','market','please','num_char_mail','uppercase','num_word','num_quesmark','num_bookmark','num_bookmark1']])

#test_predictions=model_zy.predict(process_data_fm(test)[0])
#model_zy.predict(process_data_fm(test)[0])
test_predictions=final_answer


# In[100]:


ok.grade("q10");


# The following saves a file to submit to Kaggle.

# In[101]:


from datetime import datetime

# Assuming that your predictions on the test set are stored in a 1-dimensional array called
# test_predictions. Feel free to modify this cell as long you create a CSV in the right format.

# Construct and save the submission:
submission_df = pd.DataFrame({
    "Id": test['id'], 
    "Class": test_predictions,
}, columns=['Id', 'Class'])
timestamp = datetime.isoformat(datetime.now()).split(".")[0]
submission_df.to_csv("submission_{}.csv".format(timestamp), index=False)

print('Created a CSV file: {}.'.format("submission_{}.csv".format(timestamp)))
print('You may now upload this CSV file to Kaggle for scoring.')


# In[ ]:





# # Submit
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output.
# **Please save before submitting!**
# 
# <!-- EXPECT 9 EXPORTED QUESTIONS -->

# In[ ]:


# Save your notebook first, then run this cell to submit.
import jassign.to_pdf
jassign.to_pdf.generate_pdf('proj2.ipynb', 'proj2.pdf')
ok.submit()


# In[ ]:





# In[ ]:




