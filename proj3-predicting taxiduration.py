#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Initialize OK
from client.api.notebook import Notebook
ok = Notebook('proj3.ok')


# # Project 3: Predicting Taxi Ride Duration
# ## Due Date: Thursday 8/13/19, 11:59PM
# 
# **Collaboration Policy**
# 
# Data science is a collaborative activity. While you may talk with others about
# the project, we ask that you **write your solutions individually**. If you do
# discuss the assignments with others please **include their names** at the top
# of your notebook.

# **Collaborators**: *list collaborators here*

# ## Score Breakdown
# Question | Points
# --- | ---
# 1a | 2
# 1b | 2
# 1c | 3
# 1d | 2
# 2a | 1
# 2b | 2
# 3a | 2
# 3b | 1
# 3c | 2
# 3d | 2
# 4a | 2
# 4b | 2
# 4c | 2
# 4d | 2
# 4e | 2
# 4f | 2
# 4g | 4
# Total | 35

# ## This Assignment
# In this project, you will use what you've learned in class to create a regression model that predicts the travel time of a taxi ride in New York. Some questions in this project are more substantial than those of past projects.
# 
# After this project, you should feel comfortable with the following:
# 
# - The data science lifecycle: data selection and cleaning, EDA, feature engineering, and model selection.
# - Using `sklearn` to process data and fit linear regression models.
# - Embedding linear regression as a component in a more complex model.
# 
# First, let's import:

# In[11]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# ## The Data
# Attributes of all [yellow taxi](https://en.wikipedia.org/wiki/Taxicabs_of_New_York_City) trips in January 2016 are published by the [NYC Taxi and Limosine Commission](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
# 
# The full data set takes a long time to download directly, so we've placed a simple random sample of the data into `taxi.db`, a SQLite database. You can view the code used to generate this sample in the `taxi_sample.ipynb` file included with this project (not required).
# 
# Columns of the `taxi` table in `taxi.db` include:
# - `pickup_datetime`: date and time when the meter was engaged
# - `dropoff_datetime`: date and time when the meter was disengaged
# - `pickup_lon`: the longitude where the meter was engaged
# - `pickup_lat`: the latitude where the meter was engaged
# - `dropoff_lon`: the longitude where the meter was disengaged
# - `dropoff_lat`: the latitude where the meter was disengaged
# - `passengers`: the number of passengers in the vehicle (driver entered value)
# - `distance`: trip distance
# - `duration`: duration of the trip in seconds
# 
# Your goal will be to predict `duration` from the pick-up time, pick-up and drop-off locations, and distance.

# ## Part 1: Data Selection and Cleaning
# 
# In this part, you will limit the data to trips that began and ended on Manhattan Island ([map](https://www.google.com/maps/place/Manhattan,+New+York,+NY/@40.7590402,-74.0394431,12z/data=!3m1!4b1!4m5!3m4!1s0x89c2588f046ee661:0xa0b3281fcecc08c!8m2!3d40.7830603!4d-73.9712488)). 

# ### Question 1a
# Use a SQL query to load the `taxi` table from `taxi.db` into a Pandas DataFrame called `all_taxi`. 
# 
# Only include trips that have **both** pick-up and drop-off locations within the boundaries of New York City:
# 
# - Longitude is between -74.03 and -73.75 (inclusive of both boundaries)
# - Latitude is between 40.6 and 40.88 (inclusive of both boundaries)
# 
# *Hint*: Your solution will be shorter if you write Python code to generate the SQL query string. Try not to copy and paste code.
# 
# *The provided tests check that you have constructed `all_taxi` correctly.*
# 
# <!--
# BEGIN QUESTION
# name: q1a
# points: 2
# -->

# In[12]:


import sqlite3

conn = sqlite3.connect('taxi.db')
lon_bounds = [-74.03, -73.75]
lat_bounds = [40.6, 40.88]
query = """
SELECT * from taxi
"""
all_taxi = pd.read_sql(query, conn)

all_taxi = all_taxi[all_taxi['pickup_lon'] .between(-74.03,-73.75, inclusive=True)]
all_taxi = all_taxi[all_taxi['dropoff_lon'] .between(-74.03,-73.75, inclusive=True)]
all_taxi = all_taxi[all_taxi['pickup_lat'] .between(40.6,40.88, inclusive=True)]
all_taxi = all_taxi[all_taxi['dropoff_lat'] .between(40.6,40.88, inclusive=True)]

all_taxi.head()


# In[13]:


ok.grade("q1a");


# A scatter plot of pickup locations shows that most of them are on the island of Manhattan. The empty white rectangle is Central Park; cars are not allowed there.

# In[14]:


def pickup_scatter(t):
    plt.scatter(t['pickup_lon'], t['pickup_lat'], s=2, alpha=0.2)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Pickup locations')
    
plt.figure(figsize=(8, 8))
pickup_scatter(all_taxi)


# The two small blobs outside of Manhattan with very high concentrations of taxi pick-ups are airports.

# ### Question 1b
# 
# Create a DataFrame called `clean_taxi` that only includes trips with a positive passenger count, a positive distance, a duration of at least 1 minute and at most 1 hour, and an average speed of at most 100 miles per hour. Inequalities should not be strict (e.g., `<=` instead of `<`) unless comparing to 0.
# 
# *The provided tests check that you have constructed `clean_taxi` correctly.*
# 
# <!--
# BEGIN QUESTION
# name: q1b
# points: 2
# -->

# In[15]:


clean_taxi = all_taxi[all_taxi['passengers']>0][all_taxi['distance']>0][all_taxi['duration']>=60][all_taxi['duration']<=3600]
clean_taxi=clean_taxi[(all_taxi['distance']/(all_taxi['duration']/3600))<=100]
#[all_taxi['distance']/(all_taxi['duration']/3600)<=100]]


# In[16]:


ok.grade("q1b");


# ### Question 1c (challenging)
# 
# Create a DataFrame called `manhattan_taxi` that only includes trips from `clean_taxi` that start and end within a polygon that defines the boundaries of [Manhattan Island](https://www.google.com/maps/place/Manhattan,+New+York,+NY/@40.7590402,-74.0394431,12z/data=!3m1!4b1!4m5!3m4!1s0x89c2588f046ee661:0xa0b3281fcecc08c!8m2!3d40.7830603!4d-73.9712488).
# 
# The vertices of this polygon are defined in `manhattan.csv` as (latitude, longitude) pairs, which are [published here](https://gist.github.com/baygross/5430626).
# 
# An efficient way to test if a point is contained within a polygon is [described on this page](http://alienryderflex.com/polygon/). There are even implementations on that page (though not in Python). Even with an efficient approach, the process of checking each point can take several minutes. It's best to test your work on a small sample of `clean_taxi` before processing the whole thing. (To check if your code is working, draw a scatter diagram of the (lon, lat) pairs of the result; the scatter diagram should have the shape of Manhattan.)
# 
# *The provided tests check that you have constructed `manhattan_taxi` correctly. It's not required that you implement the `in_manhattan` helper function, but that's recommended. If you cannot solve this problem, you can still continue with the project; see the instructions below the answer cell.*
# 
# <!--
# BEGIN QUESTION
# name: q1c
# points: 3
# -->

# In[17]:


polygon = pd.read_csv('manhattan.csv')

# Recommended: First develop and test a function that takes a position
#              and returns whether it's in Manhattan.
def in_manhattan(x, y):
    """Whether a longitude-latitude (x, y) pair is in the Manhattan polygon."""
    
    
# Recommended: Then, apply this function to every trip to filter clean_taxi.

manhattan_taxi = 


# In[ ]:


ok.grade("q1c");


# If you are unable to solve the problem above, have trouble with the tests, or want to work on the rest of the project before solving it, run the following cell to load the cleaned Manhattan data directly. (Note that you may not solve the previous problem just by loading this data file; you have to actually write the code.)

# In[18]:


manhattan_taxi = pd.read_csv('manhattan_taxi.csv')


# A scatter diagram of only Manhattan taxi rides has the familiar shape of Manhattan Island. 

# In[19]:


plt.figure(figsize=(8, 16))
pickup_scatter(manhattan_taxi)


# ### Question 1d
# 
# Print a summary of the data selection and cleaning you performed. Your Python code should not include any number literals, but instead should refer to the shape of `all_taxi`, `clean_taxi`, and `manhattan_taxi`.
# 
# E.g., you should print something like: "Of the original 1000 trips, 21 anomolous trips (2.1%) were removed through data cleaning, and then the 600 trips within Manhattan were selected for further analysis."
# 
# (Note that the numbers in the example above are not accurate.)
# 
# **Please ensure that your Python code does not contain any very long lines, or we can't grade it.**
# 
# *Your response will be scored based on whether you generate an accurate description and do not include any number literals in your Python expression, but instead refer to the dataframes you have created.*
# 
# <!--
# BEGIN QUESTION
# name: q1d
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# In[52]:


#of the original 100000 trips, 3555 anomolous trips(3.6%)were removed through data cleaning, 
#and then then there are 82800 data of manhattan_taxi."


# ## Part 2: Exploratory Data Analysis
# 
# In this part, you'll choose which days to include as training data in your regression model. 
# 
# Your goal is to develop a general model that could potentially be used for future taxi rides. There is no guarantee that future distributions will resemble observed distributions, but some effort to limit training data to typical examples can help ensure that the training data are representative of future observations.
# 
# January 2016 had some atypical days. New Years Day (January 1) fell on a Friday. MLK Day was on Monday, January 18. A [historic blizzard](https://en.wikipedia.org/wiki/January_2016_United_States_blizzard) passed through New York that month. Using this dataset to train a general regression model for taxi trip times must account for these unusual phenomena, and one way to account for them is to remove atypical days from the training data.

# ### Question 2a
# 
# Add a column labeled `date` to `manhattan_taxi` that contains the date (but not the time) of pickup, formatted as a `datetime.date` value ([docs](https://docs.python.org/3/library/datetime.html#date-objects)). 
# 
# *The provided tests check that you have extended `manhattan_taxi` correctly.*
# 
# <!--
# BEGIN QUESTION
# name: q2a
# points: 1
# -->

# In[53]:


import datetime
manhattan_taxi['date']=pd.to_datetime(manhattan_taxi['pickup_datetime']).dt.date

manhattan_taxi.head()


# In[54]:


ok.grade("q2a");


# ### Question 2b
# 
# Create a data visualization that allows you to identify which dates were affected by the historic blizzard of January 2016. Make sure that the visualization type is appropriate for the visualized data.
# 
# <!--
# BEGIN QUESTION
# name: q2b
# points: 2
# manual: True
# format: image
# -->
# <!-- EXPORT TO PDF -->

# In[55]:


#sns.countplot(x='date',data=manhattan_taxi)
sns.boxplot(x=manhattan_taxi['date'].value_counts().values)
#blizzard_dates=manhattan_taxi['date'].value_counts()<2000
#tru_blizzard_dates=blizzard_dates[blizzard_dates.values==True]
#tru_blizzard_dates

blizzard_dates=manhattan_taxi['date'].value_counts()
arr1=np.percentile(blizzard_dates,[25,75])
boundvalue=arr1[1]-(arr1[1]-arr1[0])*1.5
blizzard_dates=manhattan_taxi['date'].value_counts()<boundvalue
tru_blizzard_dates=blizzard_dates[blizzard_dates.values==True]
tru_blizzard_dates


# Finally, we have generated a list of dates that should have a fairly typical distribution of taxi rides, which excludes holidays and blizzards. The cell below assigns `final_taxi` to the subset of `manhattan_taxi` that is on these days. (No changes are needed; just run this cell.)

# In[56]:


import calendar
import re

from datetime import date

atypical = [1, 2, 3, 18, 23, 24, 25, 26]
typical_dates = [date(2016, 1, n) for n in range(1, 32) if n not in atypical]
typical_dates

print('Typical dates:\n')
pat = '  [1-3]|18 | 23| 24|25 |26 '
print(re.sub(pat, '   ', calendar.month(2016, 1)))

final_taxi = manhattan_taxi[manhattan_taxi['date'].isin(typical_dates)]


# You are welcome to perform more exploratory data analysis, but your work will not be scored. Here's a blank cell to use if you wish. In practice, further exploration would be warranted at this point, but the project is already pretty long.

# In[57]:


# Optional: More EDA here


# ## Part 3: Feature Engineering
# 
# In this part, you'll create a design matrix (i.e., feature matrix) for your linear regression model. You decide to predict trip duration from the following inputs: start location, end location, trip distance, time of day, and day of the week (*Monday, Tuesday, etc.*). 
# 
# You will ensure that the process of transforming observations into a design matrix is expressed as a Python function called `design_matrix`, so that it's easy to make predictions for different samples in later parts of the project.
# 
# Because you are going to look at the data in detail in order to define features, it's best to split the data into training and test sets now, then only inspect the training set.

# In[58]:


import sklearn.model_selection

train, test = sklearn.model_selection.train_test_split(
    final_taxi, train_size=0.8, test_size=0.2, random_state=42)
print('Train:', train.shape, 'Test:', test.shape)


# ### Question 3a
# 
# Create a box plot that compares the distributions of taxi trip durations for each day **using `train` only**. Individual dates shoud appear on the horizontal axis, and duration values should appear on the vertical axis. Your plot should look like this:
# 
# ![](images/3a_duration_by_date.png)
# 
# 
# <!--
# BEGIN QUESTION
# name: q3a
# points: 2
# manual: True
# format: image
# -->
# <!-- EXPORT TO PDF -->

# In[59]:


#train.sort_values(by=['date'])
duration_box=sns.boxplot(x="date", y="duration", data=train.sort_values(by=['date']))
plt.title('Duration by date')
duration_box.set_xticklabels(duration_box.get_xticklabels(), rotation=90);


# ### Question 3b
# 
# In one or two sentences, describe the assocation between the day of the week and the duration of a taxi trip. 
# 
# *Note*: The end of Part 2 showed a calendar for these dates and their corresponding days of the week.
# 
# <!--
# BEGIN QUESTION
# name: q3b
# points: 1
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# The median of duration of taxi trip is lower in weekends than in weekdays.

# Below, the provided `augment` function adds various columns to a taxi ride dataframe. 
# 
# - `hour`: The integer hour of the pickup time. E.g., a 3:45pm taxi ride would have `15` as the hour. A 12:20am ride would have `0`.
# - `day`: The day of the week with Monday=0, Sunday=6.
# - `weekend`: 1 if and only if the `day` is Saturday or Sunday.
# - `period`: 1 for early morning (12am-6am), 2 for daytime (6am-6pm), and 3 for night (6pm-12pm).
# - `speed`: Average speed in miles per hour.
# 
# No changes are required; just run this cell.

# In[60]:


def speed(t):
    """Return a column of speeds in miles per hour."""
    return t['distance'] / t['duration'] * 60 * 60

def augment(t):
    """Augment a dataframe t with additional columns."""
    u = t.copy()
    pickup_time = pd.to_datetime(t['pickup_datetime'])
    u.loc[:, 'hour'] = pickup_time.dt.hour
    u.loc[:, 'day'] = pickup_time.dt.weekday
    u.loc[:, 'weekend'] = (pickup_time.dt.weekday >= 5).astype(int)
    u.loc[:, 'period'] = np.digitize(pickup_time.dt.hour, [0, 6, 18])
    u.loc[:, 'speed'] = speed(t)
    return u
    
train = augment(train)
test = augment(test)
train.iloc[0,:] # An example row


# ### Question 3c
# 
# Use `sns.distplot` to create an overlaid histogram comparing the distribution of average speeds for taxi rides that start in the early morning (12am-6am), day (6am-6pm; 12 hours), and night (6pm-12am; 6 hours). Your plot should look like this:
# 
# ![](images/3c_speeds.png)
# 
# <!--
# BEGIN QUESTION
# name: q3c
# points: 2
# manual: True
# format: image
# -->
# <!-- EXPORT TO PDF -->

# In[61]:


train_copy=train[['distance','duration','pickup_datetime']]
train_copy['speed']=train_copy['distance']/(train_copy['duration']/3600)
train_copy['starttime']=train_copy['pickup_datetime'].str.split(" ").str[1].str.split(':').str[0].astype(int)
sns.distplot(train_copy[(train_copy['starttime']<6) &(train_copy['starttime']>=0)]['speed'],label='Early Morning')
sns.distplot(train_copy[(train_copy['starttime']>=6) &(train_copy['starttime']<18)]['speed'],label='Day')
sns.distplot(train_copy[train_copy['starttime']>=18]['speed'],label='Night')
plt.legend(fontsize=13)


# In[62]:


train


# It looks like the time of day is associated with the average speed of a taxi ride.

# ### Question 3d
# 
# Manhattan can roughly be divided into Lower, Midtown, and Upper regions. Instead of studying a map, let's approximate by finding the first principal component of the pick-up location (latitude and longitude). 
# 
# Add a `region` column to `train` that categorizes each pick-up location as 0, 1, or 2 based on the value of each point's first principal component, such that an equal number of points fall into each region. 
# 
# Read the documentation of [`pd.qcut`](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.qcut.html), which categorizes points in a distribution into equal-frequency bins.
# 
# You don't need to add any lines to this solution. Just fill in the assignment statements to complete the implementation.
# 
# *The provided tests ensure that you have answered the question correctly.*
# 
# <!--
# BEGIN QUESTION
# name: q3d
# points: 2
# -->

# In[63]:


# Find the first principle component
D = train[['pickup_lon','pickup_lat']]
pca_n = train.shape[0]
pca_means = np.mean(D,axis=0)
X = (D - pca_means) / np.sqrt(pca_n)
u, s, vt = np.linalg.svd(X, full_matrices=False)

def add_region(t):
    """Add a region column to t based on vt above."""
    D = t[['pickup_lon','pickup_lat']]
    assert D.shape[0] == t.shape[0], 'You set D using the incorrect table'
    # Always use the same data transformation used to compute vt
    X = (D - pca_means) / np.sqrt(pca_n) 
    first_pc = X @ np.transpose(vt)[0]
    t.loc[:,'region'] = pd.qcut(first_pc, 3, labels=[0, 1, 2])
    #return first_pc
#X @ np.transpose(vt)[0]
#np.linalg.svd(X, full_matrices=False)[2]

#np.asarray(X @ np.transpose(np.linalg.svd(X, full_matrices=False)[2])[0]).ndim
add_region(train)
add_region(test)


# In[64]:


ok.grade("q3d");


# Let's see how PCA divided the trips into three groups. These regions do roughly correspond to Lower Manhattan (below 14th street), Midtown Manhattan (between 14th and the park), and Upper Manhattan (bordering Central Park). No prior knowledge of New York geography was required!

# In[65]:


plt.figure(figsize=(8, 16))
for i in [0, 1, 2]:
    pickup_scatter(train[train['region'] == i])


# ### Questoin 3e (ungraded)
# 
# Use `sns.distplot` to create an overlaid histogram comparing the distribution of speeds for nighttime taxi rides (6pm-12am) in the three different regions defined above. Does it appear that there is an association between region and average speed during the night?

# In[66]:


...


# Finally, we create a design matrix that includes many of these features. Quantitative features are converted to standard units, while categorical features are converted to dummy variables using one-hot encoding. The `period` is not included because it is a linear combination of the `hour`. The `weekend` variable is not included because it is a linear combination of the `day`.  The `speed` is not included because it was computed from the `duration`; it's impossible to know the speed without knowing the duration, given that you know the distance.

# In[67]:


from sklearn.preprocessing import StandardScaler

num_vars = ['pickup_lon', 'pickup_lat', 'dropoff_lon', 'dropoff_lat', 'distance']
cat_vars = ['hour', 'day', 'region']

scaler = StandardScaler()
scaler.fit(train[num_vars])

def design_matrix(t):
    """Create a design matrix from taxi ride dataframe t."""
    scaled = t[num_vars].copy()
    scaled.iloc[:,:] = scaler.transform(scaled) # Convert to standard units
    categoricals = [pd.get_dummies(t[s], prefix=s, drop_first=True) for s in cat_vars]
    return pd.concat([scaled] + categoricals, axis=1)

design_matrix(train)
#.iloc[0,:]  


# ## Part 4: Model Selection
# 
# In this part, you will select a regression model to predict the duration of a taxi ride.
# 
# **Important:** *Tests in this part do not confirm that you have answered correctly. Instead, they check that you're somewhat close in order to detect major errors. It is up to you to calculate the results correctly based on the question descriptions.*

# ### Question 4a
# 
# Assign `constant_rmse` to the root mean squared error on the test set for a constant model that always predicts the mean duration of all training set taxi rides.
# 
# <!--
# BEGIN QUESTION
# name: q4a
# points: 2
# -->

# In[68]:


def rmse(errors):
    """Return the root mean squared error."""
    return np.sqrt(np.mean(errors ** 2))
#train['duration'].mean()
constant_rmse = rmse(test['duration']-train['duration'].mean())
constant_rmse


# In[69]:


ok.grade("q4a");


# ### Question 4b
# 
# Assign `simple_rmse` to the root mean squared error on the test set for a simple linear regression model that uses only the distance of the taxi ride as a feature (and includes an intercept).
# 
# *Terminology Note*: Simple linear regression means that there is only one covariate. Multiple linear regression means that there is more than one. In either case, you can use the `LinearRegression` model from `sklearn` to fit the parameters to data.
# 
# <!--
# BEGIN QUESTION
# name: q4b
# points: 2
# -->

# In[70]:


from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model.fit(train[['distance']],train.loc[:,'duration'])

model.predict(test[['distance']])
#test[['duration']].to_numpy

simple_rmse = rmse(test['duration']-model.predict(test[['distance']]))
simple_rmse
#simple_rmse=rmse(test_y-model.predict(test_x))
#simple_rmse


# In[71]:


ok.grade("q4b");


# ### Question 4c
# 
# Assign `linear_rmse` to the root mean squared error on the test set for a linear regression model fitted to the training set without regularization, using the design matrix defined by the `design_matrix` function from Part 3.
# 
# *The provided tests check that you have answered the question correctly and that your `design_matrix` function is working as intended.*
# 
# <!--
# BEGIN QUESTION
# name: q4c
# points: 2
# -->

# In[72]:


model = LinearRegression()
model.fit(design_matrix(train),train.loc[:,'duration'])
linear_rmse = rmse(test['duration']-model.predict(design_matrix(test)))
#test['duration']-model.predict(design_matrix(test))
#linear_rmse


# In[73]:


ok.grade("q4c");


# In[ ]:





# ### Question 4d
# 
# For each possible value of `period`, fit an unregularized linear regression model to the subset of the training set in that `period`.  Assign `period_rmse` to the root mean squared error on the test set for a model that first chooses linear regression parameters based on the observed period of the taxi ride, then predicts the duration using those parameters. Again, fit to the training set and use the `design_matrix` function for features.
# 
# <!--
# BEGIN QUESTION
# name: q4d
# points: 2
# -->

# In[74]:


model = LinearRegression()
errors = []

#d_matrix_train=design_matrix(train) #new added
#d_matrix_test=design_matrix(test)
for v in np.unique(train['period']):
    model.fit(design_matrix(train[train['period']==v]),train[train['period']==v].loc[:,'duration'])
    errors.append(test[test['period']==v]['duration']-model.predict(design_matrix(test[test['period']==v])))
period_rmse=rmse(pd.concat(errors))
    #errors.append()
#period_rmse=rmse(np.array(errors))
period_rmse
#model.fit(design_matrix(train[train['period']==1]),train[train['period']==1].loc[:,'duration'])
#design_matrix(train[train['period']==1])
#model.fit(design_matrix(train[train['period']==1]),train[train['period']==1].loc[:,'duration'])
#errors.append(test[test['period']==1]['duration']-model.predict(design_matrix(test[test['period']==1])))
#errors


# In[75]:


ok.grade("q4d");


# This approach is a simple form of decision tree regression, where a different regression function is estimated for each possible choice among a collection of choices. In this case, the depth of the tree is only 1.

# ### Question 4e
# 
# In one or two sentences, explain how the `period` regression model could possibly outperform linear regression when the design matrix for linear regression already includes one feature for each possible hour, which can be combined linearly to determine the `period` value.
# 
# <!--
# BEGIN QUESTION
# name: q4e
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# The "period" feature are distinguished by labeling 1,2,3, which influences the parameter because 1,2,3 are just labels. But this method treat 1,2,3 fairly.

# In[76]:


train['speed']


# ### Question 4f
# 
# Instead of predicting duration directly, an alternative is to predict the average *speed* of the taxi ride using linear regression, then compute an estimate of the duration from the predicted speed and observed distance for each ride.
# 
# Assign `speed_rmse` to the root mean squared error in the **duration** predicted by a model that first predicts speed as a linear combination of features from the `design_matrix` function, fitted on the training set, then predicts duration from the predicted speed and observed distance.
# 
# *Hint*: Speed is in miles per hour, but duration is measured in seconds. You'll need the fact that there are 60 * 60 = 3,600 seconds in an hour.
# 
# <!--
# BEGIN QUESTION
# name: q4f
# points: 2
# -->

# In[77]:


model = LinearRegression()
model.fit(design_matrix(train),train.loc[:,'speed'])
speed_rmse=rmse(test['distance']/test['speed']*3600-test['distance']/(model.predict(design_matrix(test)))*3600)
speed_rmse
rmse(test['duration']-test['distance']/(model.predict(design_matrix(test)))*3600)
#linear_rmse = rmse(test['speed']-model.predict(design_matrix(test)))
#speed_rmse = ...
#speed_rmse


# In[78]:


ok.grade("q4f");


# *Optional*: Explain why predicting speed leads to a more accurate regression model than predicting duration directly.

# ### Question 4g
# 
# Finally, complete the function `tree_regression_errors` (and helper function `speed_error`) that combines the ideas from the two previous models and generalizes to multiple categorical variables.
# 
# The `tree_regression_errors` should:
# - Find a different linear regression model for each possible combination of the variables in `choices`;
# - Fit to the specified `outcome` (on train) and predict that `outcome` (on test) for each combination (`outcome` will be `'duration'` or `'speed'`);
# - Use the specified `error_fn` (either `duration_error` or `speed_error`) to compute the error in predicted duration using the predicted outcome;
# - Aggregate those errors over the whole test set and return them.
# 
# You should find that including each of `period`, `region`, and `weekend` improves prediction accuracy, and that predicting speed rather than duration leads to more accurate duration predictions.
# 
# <!--
# BEGIN QUESTION
# name: q4g
# points: 4
# -->

# In[79]:


model = LinearRegression()
choices = ['period', 'region', 'weekend']

def duration_error(predictions, observations):
    """Error between predictions (array) and observations (data frame)"""
    return predictions - observations['duration']

def speed_error(predictions, observations):
    """Duration error between speed predictions and duration observations"""
    return duration_error(observations['distance']/predictions*3600, observations)

def tree_regression_errors(outcome='duration', error_fn=duration_error):
    """Return errors for all examples in test using a tree regression model."""
    errors = []
    for vs in train.groupby(choices).size().index:
        v_train, v_test = train, test
        for v, c in zip(vs, choices):
            v_train = v_train[v_train[c]==v]
            v_test = v_test[v_test[c]==v]
        model.fit(design_matrix(v_train),v_train.loc[:,outcome])
        error=error_fn(model.predict(design_matrix(v_test)),v_test)
        errors.append(error)
    return errors

#tree_regression_errors('duration', duration_error)
errors = tree_regression_errors()
errors_via_speed = tree_regression_errors('speed', speed_error)
tree_rmse=rmse(pd.concat(errors))
tree_speed_rmse=rmse(pd.concat(errors_via_speed))
#tree_rmse = rmse(np.array(errors))
#tree_speed_rmse = rmse(np.array(errors_via_speed))
print('Duration:', tree_rmse, '\nSpeed:', tree_speed_rmse)


# In[80]:


ok.grade("q4g");


# Here's a summary of your results:

# In[81]:


models = ['constant', 'simple', 'linear', 'period', 'speed', 'tree', 'tree_speed']
pd.DataFrame.from_dict({
    'Model': models,
    'Test RMSE': [eval(m + '_rmse') for m in models]
}).set_index('Model').plot(kind='barh');


# **Congratulations**! You've carried out the entire data science lifecycle for a challenging regression problem. 
# 
# In Part 1 on data selection, you solved a domain-specific programming problem relevant to the analysis when choosing only those taxi rides that started and ended in Manhattan.
# 
# In Part 2 on EDA, you used the data to assess the impact of a historical event---the 2016 blizzard---and filtered the data accordingly.
# 
# In Part 3 on feature engineering, you used PCA to divide up the map of Manhattan into regions that roughly corresponded to the standard geographic description of the island.
# 
# In Part 4 on model selection, you found that using linear regression in practice can involve more than just choosing a design matrix. Tree regression made better use of categorical variables than linear regression. The domain knowledge that duration is a simple function of distance and speed allowed you to predict duration more accurately by first predicting speed.
# 
# Hopefully, it is apparent that all of these steps are required to reach a reliable conclusion about what inputs and model structure are helpful in predicting the duration of a taxi ride in Manhattan. 

# ## Future Work
# 
# Here are some questions to ponder:
# 
# - The regression model would have been more accurate if we had used the date itself as a feature instead of just the day of the week. Why didn't we do that?
# - Does collecting this information about every taxi ride introduce a privacy risk? The original data also included the total fare; how could someone use this information combined with an individual's credit card records to determine their location?
# - Why did we treat `hour` as a categorical variable instead of a quantitative variable? Would a similar treatment be beneficial for latitude and longitude?
# - Why are Google Maps estimates of ride time much more accurate than our estimates?
# 
# 
# Here are some possible extensions to the project:
# 
# - An alternative to throwing out atypical days is to condition on a feature that makes them atypical, such as the weather or holiday calendar. How would you do that?
# - Training a different linear regression model for every possible combination of categorical variables can overfit. How would you select which variables to include in a decision tree instead of just using them all?
# - Your models use the observed distance as an input, but the distance is only observed after the ride is over. How could you estimate the distance from the pick-up and drop-off locations?
# - How would you incorporate traffic data into the model?

# In[ ]:


# Save your notebook first, then run this cell to submit.
import jassign.to_pdf
jassign.to_pdf.generate_pdf('proj3.ipynb', 'proj3.pdf')
ok.submit()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




