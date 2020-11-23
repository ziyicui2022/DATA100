#!/usr/bin/env python
# coding: utf-8

# In[270]:


# Initialize OK
from client.api.notebook import Notebook
ok = Notebook('proj1.ok')


# # Project 1: Food Safety 
# ## Cleaning and Exploring Data with Pandas
# ## Due Date: Tuesday 07/02, 11:59 PM
# ## Collaboration Policy
# 
# Data science is a collaborative activity. While you may talk with others about
# the project, we ask that you **write your solutions individually**. If you do
# discuss the assignments with others please **include their names** at the top
# of your notebook.

# **Collaborators**: *list collaborators here*

# 
# ## This Assignment
# <img src="scoreCard.jpg" width=400>
# 
# In this project, you will investigate restaurant food safety scores for restaurants in San Francisco. Above is a sample score card for a restaurant. The scores and violation information have been made available by the San Francisco Department of Public Health. The main goal for this assignment is to understand how restaurants are scored. We will walk through various steps of exploratory data analysis to do this. We will provide comments and insights along the way to give you a sense of how we arrive at each discovery and what next steps it leads to.
# 
# As we clean and explore these data, you will gain practice with:
# * Reading simple csv files
# * Working with data at different levels of granularity
# * Identifying the type of data collected, missing values, anomalies, etc.
# * Applying probability sampling techniques
# * Exploring characteristics and distributions of individual variables
# 
# ## Score Breakdown
# Question | Points
# --- | ---
# 1a | 1
# 1b | 0
# 1c | 0
# 1d | 3
# 1e | 1
# 2a | 1
# 2b | 2
# 3a | 2
# 3b | 0
# 3c | 2
# 3d | 1
# 3e | 1
# 3f | 1
# 4a | 1
# 4b | 1
# 4c | 1
# 4d | 1
# 4e | 1
# 4f | 1
# 4g | 2
# 4h | 1
# 4i | 1
# 5a | 2
# 5b | 3
# 6a | 1
# 6b | 1
# 6c | 1
# 7a | 2
# 7b | 3
# 7c | 3
# 8a | 2
# 8b | 2
# 8c | 6
# 8d | 2
# 8e | 3
# Total | 56

# To start the assignment, run the cell below to set up some imports and the automatic tests that we will need for this assignment:
# 
# In many of these assignments (and your future adventures as a data scientist) you will use `os`, `zipfile`, `pandas`, `numpy`, `matplotlib.pyplot`, and optionally `seaborn`.  
# 
# 1. Import each of these libraries `as` their commonly used abbreviations (e.g., `pd`, `np`, `plt`, and `sns`).  
# 1. Don't forget to include `%matplotlib inline` which enables [inline matploblib plots](http://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-matplotlib). 
# 1. If you want to use `seaborn`, add the line `sns.set()` to make your plots look nicer.

# In[271]:


...


# In[272]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zipfile 
import seaborn as sns
sns.set()


# ## Downloading the Data
# 
# For this assignment, we need this data file: http://www.ds100.org/sp19/assets/datasets/proj1-SFBusinesses.zip
# 
# We could write a few lines of code that are built to download this specific data file, but it's a better idea to have a general function that we can reuse for all of our assignments. Since this class isn't really about the nuances of the Python file system libraries, we've provided a function for you in ds100_utils.py called `fetch_and_cache` that can download files from the internet.
# 
# This function has the following arguments:
# - data_url: the web address to download
# - file: the file in which to save the results
# - data_dir: (default="data") the location to save the data
# - force: if true the file is always re-downloaded 
# 
# The way this function works is that it checks to see if `data_dir/file` already exists. If it does not exist already or if `force=True`, the file at `data_url` is downloaded and placed at `data_dir/file`. The process of storing a data file for reuse later is called caching. If `data_dir/file` already and exists `force=False`, nothing is downloaded, and instead a message is printed letting you know the date of the cached file.
# 
# The function returns a `pathlib.Path` object representing the location of the file ([pathlib docs](https://docs.python.org/3/library/pathlib.html#basic-use)). 

# In[273]:


import ds100_utils
source_data_url = 'http://www.ds100.org/sp19/assets/datasets/proj1-SFBusinesses.zip'
target_file_name = 'data.zip'

# Change the force=False -> force=True in case you need to force redownload the data
dest_path = ds100_utils.fetch_and_cache(
    data_url=source_data_url, 
    data_dir='.', 
    file=target_file_name, 
    force=True)


# After running the cell above, if you list the contents of the directory containing this notebook, you should see `data.zip`.

# In[274]:


get_ipython().system('ls')


# ---
# ## 0. Before You Start
# 
# For all the assignments with programming practices, please write down your answer in the answer cell(s) right below the question. 
# 
# We understand that it is helpful to have extra cells breaking down the process towards reaching your final answer. If you happen to create new cells below your answer to run codes, **NEVER** add cells between a question cell and the answer cell below it. It will cause errors in running Autograder, and sometimes fail to generate the PDF file.
# 

# ## 1: Loading Food Safety Data
# 
# We have data, but we don't have any specific questions about the data yet, so let's focus on understanding the structure of the data. This involves answering questions such as:
# 
# * Is the data in a standard format or encoding?
# * Is the data organized in records?
# * What are the fields in each record?
# 
# Let's start by looking at the contents of `data.zip`. It's not just a single file, but a compressed directory of multiple files. We could inspect it by uncompressing it using a shell command such as `!unzip data.zip`, but in this project we're going to do almost everything in Python for maximum portability.

# ### Question 1a: Looking Inside and Extracting the Zip Files
# 
# Assign `my_zip` to a `Zipfile.zipfile` object representing `data.zip`, and assign `list_files` to a list of all the names of the files in `data.zip`.
# 
# *Hint*: The [Python docs](https://docs.python.org/3/library/zipfile.html) describe how to create a `zipfile.ZipFile` object. You might also look back at the code from lecture and lab. It's OK to copy and paste code from previous assignments and demos, though you might get more out of this exercise if you type out an answer.
# 
# <!--
# BEGIN QUESTION
# name: q1a
# points: 1
# -->

# In[275]:


my_zip = zipfile.ZipFile(dest_path, 'r')
list_names =[f.filename for  f in my_zip.filelist]
list_names


# In[276]:


ok.grade("q1a");


# In your answer above, if you have written something like `zipfile.ZipFile('data.zip', ...)`, we suggest changing it to read `zipfile.ZipFile(dest_path, ...)`. In general, we **strongly suggest having your filenames hard coded as string literals only once** in a notebook. It is very dangerous to hard code things twice, because if you change one but forget to change the other, you can end up with bugs that are very hard to find.

# Now display the files' names and their sizes.
# 
# If you're not sure how to proceed, read about the attributes of a `ZipFile` object in the Python docs linked above.

# In[277]:


my_zip.infolist()


# Often when working with zipped data, we'll never unzip the actual zipfile. This saves space on our local computer. However, for this project, the files are small, so we're just going to unzip everything. This has the added benefit that you can look inside the csv files using a text editor, which might be handy for understanding what's going on. The cell below will unzip the csv files into a subdirectory called `data`. Just run it.

# In[278]:


from pathlib import Path
data_dir = Path('data')
my_zip.extractall(data_dir)
get_ipython().system('ls {data_dir}')


# The cell above created a folder called `data`, and in it there should be four CSV files. Open up `legend.csv` to see its contents. Click on 'Jupyter' in the top left, then navigate to su19/proj/proj1/data/ and click on `legend.csv`. The file will open up in another tab. You should see something that looks like:
# 
#     "Minimum_Score","Maximum_Score","Description"
#     0,70,"Poor"
#     71,85,"Needs Improvement"
#     86,90,"Adequate"
#     91,100,"Good"

# ### Question 1b: Programatically Looking Inside the Files

# The `legend.csv` file does indeed look like a well-formed CSV file. Let's check the other three files. Rather than opening up each file manually, let's use Python to print out the first 5 lines of each. The `ds100_utils` library has a method called `head` that will allow you to retrieve the first N lines of a file as a list. For example `ds100_utils.head('data/legend.csv', 5)` will return the first 5 lines of "data/legend.csv". Try using this function to print out the first 5 lines of all four files that we just extracted from the zipfile.

# In[279]:


for i in list_names:
   print(ds100_utils.head('data/'+i, 5))


# ### Question 1c: Reading in the Files
# 
# Based on the above information, let's attempt to load `businesses.csv`, `inspections.csv`, and `violations.csv` into pandas data frames with the following names: `bus`, `ins`, and `vio` respectively.
# 
# *Note:* Because of character encoding issues one of the files (`bus`) will require an additional argument `encoding='ISO-8859-1'` when calling `pd.read_csv`. One day you should read all about [character encodings](https://www.diveinto.org/python3/strings.html).

# In[280]:


# path to directory containing data
dsDir = Path('data')

bus = pd.read_csv('data/businesses.csv', encoding='ISO-8859-1')
ins = pd.read_csv('data/inspections.csv')
vio = pd.read_csv('data/violations.csv')


# Now that you've read in the files, let's try some `pd.DataFrame` methods ([docs](https://pandas.pydata.org/pandas-docs/version/0.21/generated/pandas.DataFrame.html)).
# Use the `DataFrame.head` method to show the top few lines of the `bus`, `ins`, and `vio` dataframes. To show multiple return outputs in one single cell, you can use `display()`. Use `Dataframe.describe` to learn about the numeric columns.

# In[281]:


display(bus.head(3),ins.head(3),vio.head(3))


# In[282]:


display(bus.describe(),ins.describe(),vio.describe())


# The `DataFrame.describe` method can also be handy for computing summaries of various statistics of our dataframes. Try it out with each of our 3 dataframes.

# In[283]:


display(bus.describe(include='all'),ins.describe(include='all'),vio.describe(include='all'))


# Now, we perform some sanity checks for you to verify that you loaded the data with the right structure. Run the following cells to load some basic utilities (you do not need to change these at all):

# First, we check the basic structure of the data frames you created:

# In[284]:


assert all(bus.columns == ['business_id', 'name', 'address', 'city', 'state', 'postal_code',
                           'latitude', 'longitude', 'phone_number'])
assert 6400 <= len(bus) <= 6420

assert all(ins.columns == ['business_id', 'score', 'date', 'type'])
assert 14210 <= len(ins) <= 14250

assert all(vio.columns == ['business_id', 'date', 'description'])
assert 39020 <= len(vio) <= 39080


# Next we'll check that the statistics match what we expect. The following are hard-coded statistical summaries of the correct data.

# In[285]:


bus_summary = pd.DataFrame(**{'columns': ['business_id', 'latitude', 'longitude'],
 'data': {'business_id': {'50%': 68294.5, 'max': 94574.0, 'min': 19.0},
  'latitude': {'50%': 37.780435, 'max': 37.824494, 'min': 37.668824},
  'longitude': {'50%': -122.41885450000001,
   'max': -122.368257,
   'min': -122.510896}},
 'index': ['min', '50%', 'max']})

ins_summary = pd.DataFrame(**{'columns': ['business_id', 'score'],
 'data': {'business_id': {'50%': 61462.0, 'max': 94231.0, 'min': 19.0},
  'score': {'50%': 92.0, 'max': 100.0, 'min': 48.0}},
 'index': ['min', '50%', 'max']})

vio_summary = pd.DataFrame(**{'columns': ['business_id'],
 'data': {'business_id': {'50%': 62060.0, 'max': 94231.0, 'min': 19.0}},
 'index': ['min', '50%', 'max']})

from IPython.display import display

print('What we expect from your Businesses dataframe:')
display(bus_summary)
print('What we expect from your Inspections dataframe:')
display(ins_summary)
print('What we expect from your Violations dataframe:')
display(vio_summary)


# The code below defines a testing function that we'll use to verify that your data has the same statistics as what we expect. Run these cells to define the function. The `df_allclose` function has this name because we are verifying that all of the statistics for your dataframe are close to the expected values. Why not `df_allequal`? It's a bad idea in almost all cases to compare two floating point values like 37.780435, as rounding error can cause spurious failures.

# ## Question 1d: Verifying the data
# 
# Now let's run the automated tests. If your dataframes are correct, then the following cell will seem to do nothing, which is a good thing! However, if your variables don't match the correct answers in the main summary statistics shown above, an exception will be raised.
# 
# <!--
# BEGIN QUESTION
# name: q1d
# points: 3
# -->

# In[286]:


"""Run this cell to load this utility comparison function that we will use in various
tests below (both tests you can see and those we run internally for grading).

Do not modify the function in any way.
"""


def df_allclose(actual, desired, columns=None, rtol=5e-2):
    """Compare selected columns of two dataframes on a few summary statistics.
    
    Compute the min, median and max of the two dataframes on the given columns, and compare
    that they match numerically to the given relative tolerance.
    
    If they don't match, an AssertionError is raised (by `numpy.testing`).
    """    
    # summary statistics to compare on
    stats = ['min', '50%', 'max']
    
    # For the desired values, we can provide a full DF with the same structure as
    # the actual data, or pre-computed summary statistics.
    # We assume a pre-computed summary was provided if columns is None. In that case, 
    # `desired` *must* have the same structure as the actual's summary
    if columns is None:
        des = desired
        columns = desired.columns
    else:
        des = desired[columns].describe().loc[stats]

    # Extract summary stats from actual DF
    act = actual[columns].describe().loc[stats]

    return np.allclose(act, des, rtol)


# In[287]:


ok.grade("q1d");


# ### Question 1e: Identifying Issues with the Data

# Use the `head` command on your three files again. This time, describe at least one potential problem with the data you see. Consider issues with missing values and bad data.
# 
# <!--
# BEGIN QUESTION
# name: q1e
# manual: True
# points: 1
# -->
# <!-- EXPORT TO PDF -->

# In bus table, there are some values that are NAN. We're not sure if they're misiing or lost.We should dive deeper and refine these data.
# Some zip codes are not specific zip code but display "Ca".
# 

# We will explore each file in turn, including determining its granularity and primary keys and exploring many of the variables individually. Let's begin with the businesses file, which has been read into the `bus` dataframe.

# ---
# ## 2: Examining the Business Data
# 
# From its name alone, we expect the `businesses.csv` file to contain information about the restaurants. Let's investigate the granularity of this dataset.
# 
# **Important note: From now on, the local autograder tests will not be comprehensive. You can pass the automated tests in your notebook but still fail tests in the autograder.** Please be sure to check your results carefully.

# ### Question 2a
# 
# Examining the entries in `bus`, is the `business_id` unique for each record that is each row of data? Your code should compute the answer, i.e. don't just hard code `True` or `False`.
# 
# Hint: use `value_counts()` or `unique()` to determine if the `business_id` series has any duplicates.
# 
# <!--
# BEGIN QUESTION
# name: q2a
# points: 1
# -->

# In[288]:


unique_value=len(bus['business_id'].unique())
total_value=bus['business_id'].count()
is_business_id_unique = (unique_value==total_value)
is_business_id_unique


# In[289]:


ok.grade("q2a");


# ### Question 2b
# 
# With this information, you can address the question of granularity. Answer the questions below.
# 
# 1. What does each record represent (e.g., a business, a restaurant, a location, etc.)?  
# 1. What is the primary key?
# 1. What would you find by grouping by the following columns: `business_id`, `name`, `address` each individually?
# 
# Please write your answer in the markdown cell below. You may create new cells below your answer to run code, but **please never add cells between a question cell and the answer cell below it.**
# 
# <!--
# BEGIN QUESTION
# name: q2b
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# 1. A restaurant
# 2. business_id
# 3. The groups are the same, a group represents a restaurant. But the groups appear in different orders.

# ---
# ## 3: Zip Codes
# 
# Next, let's  explore some of the variables in the business table. We begin by examining the postal code.
# 
# ### Question 3a
# 
# Answer the following questions about the `postal code` column in the `bus` data frame?  
# 1. Are ZIP codes quantitative or qualitative? If qualitative, is it ordinal or nominal? 
# 1. What data type is used to represent a ZIP code?
# 
# *Note*: ZIP codes and postal codes are the same thing.
# 
# <!--
# BEGIN QUESTION
# name: q3a
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# 1. They are qualitative,nominal.
# 2. String

# In[ ]:





# ### Question 3b
# 
# How many restaurants are in each ZIP code? 
# 
# In the cell below, create a series where the index is the postal code and the value is the number of records with that postal code in descending order of count. 94110 should be at the top with a count of 596. You may want to use `.size()` or `.value_counts()`. 
# 
# <!--
# BEGIN QUESTION
# name: q3b
# points: 0
# -->

# In[290]:


zip_counts=bus['postal_code'].value_counts().sort_values(ascending=False)
zip_counts.head()


# Did you take into account that some businesses have missing ZIP codes?

# In[291]:


print('zip_counts describes', sum(zip_counts), 'records.')
print('The original data have', len(bus), 'records')


# Missing data is extremely common in real-world data science projects. There are several ways to include missing postal codes in the `zip_counts` series above. One approach is to use the `fillna` method of the series, which will replace all null (a.k.a. NaN) values with a string of our choosing. In the example below, we picked "?????". When you run the code below, you should see that there are 240 businesses with missing zip code.

# In[292]:


zip_counts = bus.fillna("?????").groupby("postal_code").size().sort_values(ascending=False)
zip_counts.head(15)


# An alternate approach is to use the DataFrame `value_counts` method with the optional argument `dropna=False`, which will ensure that null values are counted. In this case, the index will be `NaN` for the row corresponding to a null postal code.

# In[293]:


bus["postal_code"].value_counts(dropna=False).sort_values(ascending = False).head(15)


# Missing zip codes aren't our only problem. There are also some records where the postal code is wrong, e.g., there are 3 'Ca' and 3 'CA' values. Additionally, there are some extended postal codes that are 9 digits long, rather than the typical 5 digits. We will dive deeper into problems with postal code entries in subsequent questions. 
# 
# For now, let's clean up the extended zip codes by dropping the digits beyond the first 5. Rather than deleting or replacing the old values in the `postal_code` columnm, we'll instead create a new column called `postal_code_5`.
# 
# The reason we're making a new column is that it's typically good practice to keep the original values when we are manipulating data. This makes it easier to recover from mistakes, and also makes it more clear that we are not working with the original raw data.

# In[294]:


bus['postal_code_5'] = bus['postal_code'].str[:5]
bus.head()


# ### Question 3c : A Closer Look at Missing ZIP Codes
# 
# Let's look more closely at records with missing ZIP codes. Describe why some records have missing postal codes.  Pay attention to their addresses. You will need to look at many entries, not just the first five.
# 
# *Hint*: The `isnull` method of a series returns a boolean series which is true only for entries in the original series that were missing.
# 
# <!--
# BEGIN QUESTION
# name: q3c
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# There are restaurants that have various locations, while some are off the grid.

# In[295]:


bus[bus['postal_code'].isnull()]


# ### Question 3d: Incorrect ZIP Codes

# This dataset is supposed to be only about San Francisco, so let's set up a list of all San Francisco ZIP codes.

# In[296]:


all_sf_zip_codes = ["94102", "94103", "94104", "94105", "94107", "94108", 
                    "94109", "94110", "94111", "94112", "94114", "94115", 
                    "94116", "94117", "94118", "94119", "94120", "94121", 
                    "94122", "94123", "94124", "94125", "94126", "94127", 
                    "94128", "94129", "94130", "94131", "94132", "94133", 
                    "94134", "94137", "94139", "94140", "94141", "94142", 
                    "94143", "94144", "94145", "94146", "94147", "94151", 
                    "94158", "94159", "94160", "94161", "94163", "94164", 
                    "94172", "94177", "94188"]


# Set `weird_zip_code_businesses` equal to a new dataframe showing only rows corresponding to ZIP codes that are not valid - either not 5-digit long or not a San Francisco zip code - and not missing. Use the `postal_code_5` column.
# 
# *Hint*: The `~` operator inverts a boolean array. Use in conjunction with `isin`.
# 
# <!--
# BEGIN QUESTION
# name: q3d1
# points: 0
# -->

# In[297]:


#bus[(~ bus['postal_code_5'].isin (all_sf_zip_codes)|
weird_zip_code_businesses =bus[((~ bus['postal_code'].isin(bus['postal_code_5']))| (~ bus['postal_code_5'].isin (all_sf_zip_codes))) & (~ bus['postal_code'].isnull())]
#bus[~bus['postal_code'].isin(bus['postal_code_5'])]
#weird_zip_code_businesses = ...
weird_zip_code_businesses


# If we were doing very serious data analysis, we might indivdually look up every one of these strange records. Let's focus on just two of them: ZIP codes 94545 and 94602. Use a search engine to identify what cities these ZIP codes appear in. Try to explain why you think these two ZIP codes appear in your dataframe. For the one with ZIP code 94602, try searching for the business name and locate its real address.
# <!--
# BEGIN QUESTION
# name: q3d2
# points: 1
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# Hayward, Oakland.
# The one with ZIP codes 94545 has "various locations(17)" in its address. Maybe it's easy to mess up with zip code of same restaurant in different location.
# For the second one, There are streets called "1900 MARKET ST" in both SF and Oakland.
# Orbit Room:1900 Market St, San Francisco, CA 94102

# In[298]:


bus[bus['postal_code']=='94602']


# In[299]:


bus


# ### Question 3e
# 
# We often want to clean the data to improve our analysis. This cleaning might include changing values for a variable or dropping records.
# 
# The value 94602 is wrong. Change it to the most reasonable correct value, using all information you have available. Modify the `postal_code_5` field using `bus['postal_code_5'].str.replace` to replace 94602.
# 
# <!--
# BEGIN QUESTION
# name: q3e
# points: 1
# -->

# In[300]:


# WARNING: Be careful when uncommenting the line below, it will set the entire column to NaN unless you 
# put something to the right of the ellipses.

bus['postal_code_5'] = bus['postal_code_5'].str.replace('94602','94102')


# In[301]:


ok.grade("q3e");


# ### Question 3f
# 
# Now that we have corrected one of the weird postal codes, let's filter our `bus` data such that only postal codes from San Francisco remain. While we're at it, we'll also remove the businesses that are missing a postal code. As we mentioned in question 3d, filtering our postal codes in this way may not be ideal. (Fortunately, this is just a course assignment.) Use the `postal_code_5` column.
# 
# Assign `bus` to a new dataframe that has the same columns but only the rows with ZIP codes in San Francisco.
# 
# <!--
# BEGIN QUESTION
# name: q3f
# points: 1
# -->

# In[302]:


#bus = bus['postal_code_5'].isin (all_sf_zip_codes)
bus=bus[ bus['postal_code_5'].isin (all_sf_zip_codes)]
bus.head(5)


# In[303]:


len(bus)


# In[304]:


ok.grade("q3f");


# ## 4: Sampling from the Business Data
# We can now sample from the business data using the cleaned ZIP code data. Make sure to use `postal_code_5` instead of `postal_code` for all parts of this question.

# ### Question 4a
# 
# First, complete the following function `sample`, which takes an arguments a series, `series`, and a sample size, `n`, and returns a simple random sample (SRS) of size `n` from the series. Recall that in SRS, sampling is performed **without** replacement. 
# 
# The result should be a **list** of the `n` values that are in the sample.
# 
# *Hint*: Consider using [`np.random.choice`](https://docs.scipy.org/doc/numpy-1.14.1/reference/generated/numpy.random.choice.html).
# 
# <!--
# BEGIN QUESTION
# name: q4a
# points: 1
# -->

# In[305]:


def sample(series, n):
    # Do not change the following line of code in any way!
    # In case you delete it, it should be "np.random.seed(40)"
    np.random.seed(40)
    #sample_lst=[]
    #for i in np.arange(n):
    #    sample_lst=sample_lst+np.random.choice(series, replace=False)
    arr=np.random.choice(series,n, replace=False)
    
    return arr.tolist()


# In[306]:


ok.grade("q4a");


# ### Question 4b
# Suppose we take a SRS of 5 businesses from the business data. What is the probability that the business named AMERICANA GRILL & FOUNTAIN is in the sample?
# <!--
# BEGIN QUESTION
# name: q4b
# points: 1
# -->

# In[307]:


q4b_answer =5/len(bus)
q4b_answer


# In[308]:


ok.grade("q4b");


# In[309]:


#bus[bus['name']=='AMERICANA GRILL & FOUNTAIN']
len(bus)


# ### Question 4c
# 
# **New content: Stratified Sampling**
# 
# In simple random sampling (SRS), every member or set of members has an equal chance to be selected in the final sample. We often use this method when we don’t have any kind of prior information about the target population.
# 
# Here, we actually do have a good amount of information about the population - address, coordinates, phone number, and postal code, etc. Let's try to use one of these information in our new sampling, by grouping the members via a specific factor/piece of information. 
# 
# Members of the population are first partitioned into groups, called **strata**, by their postal codes. Then, within each group (**stratum**), members are randomly selected into the final probability sample, which is often a simple random sample (SRS). This method is called **stratified sampling**. 
# 
# **EXAMPLE:**
# In Spring 2019, there were 800 students enrolled in Data 100, each of whom signed up for 1 of the 35 sections. Now we would like to survey 120 students to hear their thoughts on the midterm exam. One of the TAs proposed to do a stratified sampling; he grouped students by their standings - freshman, sophomore, junior, senior, graduate (5 **strata** in total) - and randomly chose 24 students in each group (**stratum**), and survey these 120 students. 
# 
# Now let's try to collect a stratified random sample of business names, where each stratum consists of a postal code. Collect one business name per stratum. Assign `bus_strat_sample` to a series of business names selected by this sampling procedure. Your output should be a series with the individual business names (not lists of one element each) as the values.
# 
# Hint: You can use the `sample` function you defined earlier. Also consider using `lambda x` when applying a function to a group. 
# 
# <!--
# BEGIN QUESTION
# name: q4c
# points: 1
# -->

# In[310]:


#bus['name'].groupby(bus['postal_code_5'])
def f1(x):
    v1=sample(x,1)[0]
    return v1
#bus['name'].groupby(bus['postal_code_5']).agg(f1)
#bus.groupby("postal_code")
#(lambda x: sample(x, i))
bus_strat_sample =bus['name'].groupby(bus['postal_code_5']).agg(f1)
bus_strat_sample.head()


# In[311]:


ok.grade("q4c");


# ### Question 4d
# 
# What is the probability that AMERICANA GRILL & FOUNTAIN is selected as part of this stratified random sampling procedure?
# <!--
# BEGIN QUESTION
# name: q4d
# points: 1
# -->

# In[312]:


n1=bus[bus['postal_code']=='94121']
alen=len(n1.index)
q4d_answer = 1/alen
q4d_answer


# In[313]:


ok.grade("q4d");


# In[314]:


bus[bus['name']=='AMERICANA GRILL & FOUNTAIN']


# In[315]:


n1=bus[bus['postal_code']=='94121']
len(n1.index)


# ### Question 4e
# 
# **New content: Cluster Sampling**
# 
# Different from stratified sampling, in some cases we may not need a member from each group (stratum). Another way to utilize the information we have about the population is cluster sampling. 
# 
# In cluster sampling, the population is also first divided into groups, called **clusters**, based on prior known information. Note that in cluster sampling, every member of the population is assigned to one, and only one, cluster. A sample of clusters is then chosen, using a probability method (often simple random sampling). All members of the selected clusters will be in the final probability sample. 
# 
# **EXAMPLE:**
# In Spring 2019, there were 800 students enrolled in Data 100, each of whom signed up for 1 of the 35 sections. Another TA proposed to do a cluster sampling; there were 35 sections that each has 25 seats. She randomly selected 5 sections (clusters); she didn't know how many students were there in each of these 5 sections (clusters). She ended up surveying 119 students. 
# 
# 
# Now, let's try collect a cluster sample of business IDs, where each cluster is a postal code, with 5 clusters in the sample. Assign `bus_cluster_sample` to a series of business IDs selected by this sampling procedure. Reminder: Use the `postal_code_5` column.
# 
# Hint: Consider using [`isin`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.isin.html).
# 
# <!--
# BEGIN QUESTION
# name: q4e
# points: 1
# -->

# In[316]:


#bus['postal_code_5'].value_count()
id_lst=list(bus.groupby('postal_code_5').groups.keys())
random_id_lst=np.random.choice(id_lst,5)
#movies.groupby('genre').groups.keys()
#bus[bus['postal_code_5'].isin (random_id_lst)]['business_id']
#['business_id'].groupby(bus['postal_code_5']).
bus_cluster_sample = bus[bus['postal_code_5'].isin (random_id_lst)]['business_id']
bus_cluster_sample.head()


# In[317]:


ok.grade("q4e");


# ### Question 4f
# What is the probability that AMERICANA GRILL & FOUNTAIN is selected as part of this cluster sampling procedure?
# <!--
# BEGIN QUESTION
# name: q4f
# points: 1
# -->

# In[318]:


q4f_answer = 5/len(bus.groupby('postal_code_5'))
q4f_answer


# In[319]:


ok.grade("q4f");


# In[320]:


len(bus['postal_code_5'].value_counts())


# ### Question 4g
# In the context of this question, what are the benefit(s) you can think of performing SRS over stratified sampling? what about stratified sampling over cluster sampling? Why would you consider performing one sampling method over another? Compare the strengths and weaknesses of these three sampling techniques.
# <!--
# BEGIN QUESTION
# name: q4g
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# Compared to SRS, it's worth noting that stratified sampling may be biased. The restaruants with same postal code vary a lot. One may not entirely represent whole group.
# Cluster sampling simply overlook all samples in clusters that are not choosen.It matters because there may be locations is one of determining factors for the restaurants.But it's easier to apply cluster smapling in real life.
# I'd prefer SRS, for every data has the same chance to be chosen.Sample is representative of all the restaurants.

# ### Question 4h
# Collect a multi-stage sample. First, take a SRS of 5 postal codes. You should have 5 unique postal codes after this. Then, collect an SRS of one business name per selected postal code. Assign `bus_multi_sample` to a series of names selected by this procedure. You may need to sort your result by `postal_code_5` in an ascending order. 
# 
# Similar to 4c, try using the individual businesses names as the values of the series instead of lists of one business name each.
# 
# <!--
# BEGIN QUESTION
# name: q4h
# points: 1
# -->

# In[321]:


np.random.seed(40) # Do not touch this!
postal_lst=list(bus.groupby('postal_code_5').groups.keys())
random_postal_lst=np.random.choice(postal_lst,5)
n1_bus=bus[bus['postal_code_5'].isin(random_postal_lst)].sort_values('postal_code_5',ascending=True)
#n1_bus['name'].groupby(n1_bus['postal_code_5']).agg(f1)
bus_multi_sample = n1_bus['name'].groupby(n1_bus['postal_code_5']).agg(f1)
bus_multi_sample.head()


# In[322]:


ok.grade("q4h");


# ### Question 4i
# What is the probability that AMERICANA GRILL & FOUNTAIN is chosen in the multi-stage sample (from 4h)?
# 
# <!--
# BEGIN QUESTION
# name: q4i
# points: 1
# -->

# In[323]:


q4i_answer =(1/6)*(1/160)
q4i_answer


# In[324]:


ok.grade("q4i");


# In[325]:


bus[]


# ---
# ## 5: Latitude and Longitude
# 
# Let's also consider latitude and longitude values in the `bus` data frame and get a sense of how many are missing.
# 
# ### Question 5a
# 
# How many businesses are missing longitude values?
# 
# *Hint*: Use `isnull`.
# 
# <!--
# BEGIN QUESTION
# name: q5a1
# points: 1
# -->

# In[326]:


null_bus=bus[bus['longitude'].isnull()]
#len(null_bus.index)
num_missing_longs = len(null_bus.index)
num_missing_longs


# In[327]:


ok.grade("q5a1");


# As a somewhat contrived exercise in data manipulation, let's try to identify which ZIP codes are missing the most longitude values.

# Throughout problems 5a and 5b, let's focus on only the "dense" ZIP codes of the city of San Francisco, listed below as `sf_dense_zip`.

# In[328]:


sf_dense_zip = ["94102", "94103", "94104", "94105", "94107", "94108",
                "94109", "94110", "94111", "94112", "94114", "94115",
                "94116", "94117", "94118", "94121", "94122", "94123", 
                "94124", "94127", "94131", "94132", "94133", "94134"]


# In the cell below, create a series where the index is `postal_code_5`, and the value is the number of businesses with missing longitudes in that ZIP code. Your series should be in descending order (the values should be in descending order). The first two rows of your answer should include postal code 94103 and 94110. Only businesses from `sf_dense_zip` should be included. 
# 
# *Hint: Start by making a new dataframe called `bus_sf` that only has businesses from `sf_dense_zip`.*
# 
# *Hint: Use `len` or `sum` to find out the output number.*
# 
# *Hint: Create a custom function to compute the number of null entries in a series, and use this function with the `agg` method.*
# <!--
# BEGIN QUESTION
# name: q5a2
# points: 1
# -->

# In[329]:


bus_sf=bus[bus['postal_code_5']. isin(sf_dense_zip)]
def f2(series):
    new_series=series[series.isnull()]
    return len(new_series.index)
#bus_sf['longitude'].groupby(bus_sf['postal_code_5']).agg(f2)
num_missing_in_each_zip = bus_sf['longitude'].groupby(bus_sf['postal_code_5']).agg(f2).sort_values(ascending=False)
num_missing_in_each_zip.head()


# In[330]:


ok.grade("q5a2");


# In[331]:


new_bb=bus['longitude'][bus['longitude'].isnull()]
len(new_bb.index)


# ### Question 5b
# 
# In question 5a, we counted the number of null values per ZIP code. Reminder: we still only use the zip codes found in `sf_dense_zip`. Let's now count the proportion of null values of longitudinal coordinates.
# 
# Create a new dataframe of counts of the null and proportion of null values, storing the result in `fraction_missing_df`. It should have an index called `postal_code_5` and should also have 3 columns:
# 
# 1. `count null`: The number of missing values for the zip code.
# 2. `count non null`: The number of present values for the zip code.
# 3. `fraction null`: The fraction of values that are null for the zip code.
# 
# Your data frame should be sorted by the fraction null in descending order. The first two rows of your answer should include postal code 94107 and 94124.
# 
# Recommended approach: Build three series with the appropriate names and data and then combine them into a dataframe. This will require some new syntax you may not have seen. You already have code from question 4a that computes the `null count` series.
# 
# To pursue this recommended approach, you might find these two functions useful and you aren't required to use these two:
# 
# * `rename`: Renames the values of a series.
# * `pd.concat`: Can be used to combine a list of Series into a dataframe. Example: `pd.concat([s1, s2, s3], axis=1)` will combine series 1, 2, and 3 into a dataframe. Be careful about `axis=1`. 
# 
# *Hint*: You can use the divison operator to compute the ratio of two series.
# 
# *Hint*: The - operator can invert a boolean array. Or alternately, the `notnull` method can be used to create a boolean array from a series.
# 
# *Note*: An alternate approach is to create three aggregation functions and pass them in a list to the `agg` function.
# <!--
# BEGIN QUESTION
# name: q5b
# points: 3
# -->

# In[332]:


c1=num_missing_in_each_zip.rename('count null')
allcolumn=bus_sf.groupby('postal_code_5').size()
c2=(allcolumn-c1).rename('count non null')
c3=(c1/allcolumn).rename('fraction null')
raw_fraction=pd.concat([c1,c2,c3],axis=1,sort=True)
#raw_fraction.sort_values('fraction null', ascending=False)
fraction_missing_df = raw_fraction.sort_values('fraction null', ascending=False)
# make sure to use this name for your dataframe 
fraction_missing_df.index.names =['postal_code_5']
fraction_missing_df.head()


# In[333]:


ok.grade("q5b");


# ## Summary of the Business Data
# 
# Before we move on to explore the other data, let's take stock of what we have learned and the implications of our findings on future analysis. 
# 
# * We found that the business id is unique across records and so we may be able to use it as a key in joining tables. 
# * We found that there are some errors with the ZIP codes. As a result, we dropped the records with ZIP codes outside of San Francisco or ones that were missing. In practive, however, we could take the time to look up the restaurant address online and fix these errors.   
# * We found that there are a huge number of missing longitude (and latitude) values. Fixing would require a lot of work, but could in principle be automated for records with well-formed addresses. 

# ---
# ## 6: Investigate the Inspection Data
# 
# Let's now turn to the inspection DataFrame. Earlier, we found that `ins` has 4 columns named `business_id`, `score`, `date` and `type`.  In this section, we determine the granularity of `ins` and investigate the kinds of information provided for the inspections. 

# Let's start by looking again at the first 5 rows of `ins` to see what we're working with.

# In[334]:


ins.head(5)


# ### Question 6a
# From calling `head`, we know that each row in this table corresponds to a single inspection. Let's get a sense of the total number of inspections conducted, as well as the total number of unique businesses that occur in the dataset.
# <!--
# BEGIN QUESTION
# name: q6a
# points: 1
# -->

# In[335]:


# The number of rows in ins
rows_in_table=len(ins.index)
# The number of unique business IDs in ins.
unique_ins_ids = len(ins.groupby('business_id'))
unique_ins_ids


# In[336]:


ok.grade("q6a");


# ### Question 6b
# 
# Next, let us examine the Series in the `ins` dataframe called `type`. From examining the first few rows of `ins`, we see that `type` takes string value, one of which is `'routine'`, presumably for a routine inspection. What other values does the inspection `type` take? How many occurrences of each value is in `ins`? What can we tell about these values? Can we use them for further analysis? If so, how?
# 
# <!--
# BEGIN QUESTION
# name: q6b
# points: 1
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# Complaint.(1), Routine(14221)
# 
# Complaint suggests that the restaurant has some issues.
# 
# We can further study it by comparing the same business' inspection with 'routine' and with 'complaint'. Observe whether there is improvement after the they receive complaints.

# In[337]:


ins.groupby('type').size()


# ### Question 6c
# 
# In this question, we're going to try to figure out what years the data span. The dates in our file are formatted as strings such as `20160503`, which are a little tricky to interpret. The ideal solution for this problem is to modify our dates so that they are in an appropriate format for analysis. 
# 
# In the cell below, we attempt to add a new column to `ins` called `new_date` which contains the `date` stored as a datetime object. This calls the `pd.to_datetime` method, which converts a series of string representations of dates (and/or times) to a series containing a datetime object.

# In[338]:


ins['new_date'] = pd.to_datetime(ins['date'])
ins.head(5)


# As you'll see, the resulting `new_date` column doesn't make any sense. This is because the default behavior of the `to_datetime()` method does not properly process the passed string. We can fix this by telling `to_datetime` how to do its job by providing a format string.

# In[339]:


ins['new_date'] = pd.to_datetime(ins['date'], format='%Y%m%d')
ins.head(5)


# This is still not ideal for our analysis, so we'll add one more column that is just equal to the year by using the `dt.year` property of the new series we just created.

# In[340]:


ins['year'] = ins['new_date'].dt.year
ins.head(5)


# Now that we have this handy `year` column, we can try to understand our data better.
# 
# What range of years is covered in this data set? Are there roughly the same number of inspections each year? Provide your answer in text only in the markdown cell below. If you would like show your reasoning with codes, make sure you put your code cells **below** the markdown answer cell. 
# 
# <!--
# BEGIN QUESTION
# name: q6c
# points: 1
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# It spans from 2015 to 2018. There are roughly same in 2016 and 2017. 2015 is a little less. 2018 is tremendously less.

# In[341]:


ins['year'].value_counts()


# ---
# ## 7: Explore Inspection Scores

# ### Question 7a
# Let's look at the distribution of inspection scores. As we saw before when we called `head` on this data frame, inspection scores appear to be integer values. The discreteness of this variable means that we can use a barplot to visualize the distribution of the inspection score. Make a bar plot of the counts of the number of inspections receiving each score. 
# 
# It should look like the image below. It does not need to look exactly the same (e.g., no grid), but make sure that all labels and axes are correct.
# 
# *Hint*: Use `plt.bar()` for plotting. See [PyPlot tutorial](http://data100.datahub.berkeley.edu/hub/user-redirect/git-sync?repo=https://github.com/DS-100/su19&subPath=lab/lab01/pyplot.ipynb) from Lab01 for other references, such as labeling.
# 
# *Note*: If you use seaborn `sns.countplot()`, you may need to manually set what to display on xticks. 
# 
# <img src="q7a.png" width=500>
# 
# <!--
# BEGIN QUESTION
# name: q7a
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# In[342]:


new_ins=ins.sort_values('score',ascending=True)
data=new_ins.groupby('score').size()
index_list=data.index
value_list=data.values
plt.xlabel('Score')
plt.ylabel('Count')
plt.title('Distribution of Inspection Scores')
plt.bar(index_list,value_list)
#plt.bar(data)
#x=np.linspace(0, 100, 100)
#plt.xticks(np.arange(0, 100, 10)


# In[ ]:





# ### Question 7b
# 
# Describe the qualities of the distribution of the inspections scores based on your bar plot. Consider the mode(s), symmetry, tails, gaps, and anamolous values. Are there any unusual features of this distribution? What do your observations imply about the scores?
# 
# <!--
# BEGIN QUESTION
# name: q7b
# points: 3
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# 1.The mode is 100.It's not symmetric.The tails are on the left.(around 65).Tail is short.There are gaps, but they are most obvious from 95-100. No outliers.
# 
# 2.From score 85-100, scores in odd number is obviously less compared to those with even number.
# 
# 
# 3.Large proportion of restaurants are scored really high (above 90).

# ### Question 7c

# Let's figure out which restaurants had the worst scores ever (single lowest score). Let's start by creating a new dataframe called `ins_named`. It should be exactly the same as `ins`, except that it should have the name and address of every business, as determined by the `bus` dataframe. If a `business_id` in `ins` does not exist in `bus`, the name and address should be given as NaN.
# 
# *Hint*: Use the merge method to join the `ins` dataframe with the appropriate portion of the `bus` dataframe. See the official [documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html) on how to use `merge`.
# 
# *Note*: For quick reference, a pandas 'left' join keeps the keys from the left frame, so if ins is the left frame, all the keys from ins are kept and if a set of these keys don't have matches in the other frame, the columns from the other frame for these "unmatched" key rows contains NaNs.
# 
# <!--
# BEGIN QUESTION
# name: q7c1
# points: 1
# -->

# In[343]:


n3_bus=bus[['business_id', 'name', 'address']]
ins_named = ins.merge(n3_bus,how='left', left_on='business_id',right_on='business_id')
ins_named.head()


# In[344]:


ok.grade("q7c1");


# Using this data frame, identify the restaurant with the lowest inspection scores ever. Head to yelp.com and look up the reviews page for this restaurant. Copy and paste anything interesting you want to share.
# 
# <!--
# BEGIN QUESTION
# name: q7c2
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# DA cafe.
# "It is grimy, traditional, and a local. Older asians fill up the joint -- no hipster, trendy, 1+ hr wait times. Best of all, cheap"

# Just for fun you can also look up the restaurants with the best scores. You'll see that lots of them aren't restaurants at all!

# In[345]:


#ins.sort_values('score')
bus[bus['business_id']==86647]


# ---
# ## 8: Restaurant Ratings Over Time

# Let's consider various scenarios involving restaurants with multiple ratings over time.

# In[346]:


ins[ins['business_id']==19]


# ### Question 8a

# Let's see which restaurant has had the most extreme improvement in its rating, aka scores. Let the "swing" of a restaurant be defined as the difference between its highest-ever and lowest-ever rating. **Only consider restaurants with at least 3 ratings, aka rated for at least 3 times (3 scores)!** Using whatever technique you want to use, assign `max_swing` to the name of restaurant that has the maximum swing.
# 
# *Note*: The "swing" is of a specific business. There might be some restaurants with multiple locations; each location has its own "swing".
# 
# <!--
# BEGIN QUESTION
# name: q8a1
# points: 2
# -->

# In[347]:


#ins[ins[]]
def f4(subframe):
    results=subframe['business_id'].value_counts()>3
    return results
res_3ratings=ins.groupby(ins['business_id']).filter(f4)
min_score=res_3ratings['score'].groupby(ins['business_id']).min()
max_score=res_3ratings['score'].groupby(ins['business_id']).max()
max_series=max_score-min_score
bus_index=max_series.sort_values(ascending=False).index[0]
#bus[bus['business_id']==bus_index].loc[:,'name'].values[0]
#bus[bus['business_id']==77532]
#.sort_values(ascending=False)
#ins[usiness_id'].value_counts()>3).index
#n3_series[n3_series.value>=3]
max_swing = bus[bus['business_id']==bus_index].loc[:,'name'].values[0]
max_swing


# In[350]:


ok.grade("q8a1");


# In[351]:


res_3ratings


# ### Question 8b
# 
# To get a sense of the number of times each restaurant has been inspected, create a multi-indexed dataframe called `inspections_by_id_and_year` where each row corresponds to data about a given business in a single year, and there is a single data column named `count` that represents the number of inspections for that business in that year. The first index in the MultiIndex should be on `business_id`, and the second should be on `year`.
# 
# An example row in this dataframe might look tell you that business_id is 573, year is 2017, and count is 4.
# 
# *Hint: Use groupby to group based on both the `business_id` and the `year`.*
# 
# *Hint: Use rename to change the name of the column to `count`.*
# 
# <!--
# BEGIN QUESTION
# name: q8b
# points: 2
# -->

# In[352]:



inspections_by_id_and_year = ins.groupby([ins['business_id'],ins['year']]).size().to_frame()
inspections_by_id_and_year.columns=['count']
inspections_by_id_and_year.head()
#inspections_by_id_and_year.head()


# In[353]:


ok.grade("q8b");


# You should see that some businesses are inspected many times in a single year. Let's get a sense of the distribution of the counts of the number of inspections by calling `value_counts`. There are quite a lot of businesses with 2 inspections in the same year, so it seems like it might be interesting to see what we can learn from such businesses.

# In[354]:


inspections_by_id_and_year['count'].value_counts()


# In[355]:


ins['score'].count()


# In[356]:


ins


# ### Question 8c
# 
# What's the relationship between the first and second scores for the businesses with 2 inspections in a year? Do they typically improve? For simplicity, let's focus on only 2016 for this problem, using `ins2016` data frame that will be created for you below. 
# 
# First, make a dataframe called `scores_pairs_by_business` indexed by `business_id` (containing only businesses with exactly 2 inspections in 2016).  This dataframe contains the field `score_pair` consisting of the score pairs **ordered chronologically**  `[first_score, second_score]`. 
# 
# Plot these scores. That is, make a scatter plot to display these pairs of scores. Include on the plot a reference line with slope 1. 
# 
# You may find the functions `sort_values`, `groupby`, `filter` and `agg` helpful, though not all necessary. 
# 
# The first few rows of the resulting table should look something like:
# 
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>score_pair</th>
#     </tr>
#     <tr>
#       <th>business_id</th>
#       <th></th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>24</th>
#       <td>[96, 98]</td>
#     </tr>
#     <tr>
#       <th>45</th>
#       <td>[78, 84]</td>
#     </tr>
#     <tr>
#       <th>66</th>
#       <td>[98, 100]</td>
#     </tr>
#     <tr>
#       <th>67</th>
#       <td>[87, 94]</td>
#     </tr>
#     <tr>
#       <th>76</th>
#       <td>[100, 98]</td>
#     </tr>
#   </tbody>
# </table>
# 
# The scatter plot should look like this:
# 
# <img src="q8c2.png" width=500>
# 
# *Note: Each score pair must be a list type; numpy arrays will not pass the autograder.*
# 
# *Hint: Use the `filter` method from lecture 3 to create a new dataframe that only contains restaurants that received exactly 2 inspections.*
# 
# *Hint: Our answer is a single line of code that uses `sort_values`, `groupby`, `filter`, `groupby`, `agg`, and `rename` in that order. Your answer does not need to use these exact methods.*
# 
# <!--
# BEGIN QUESTION
# name: q8c1
# points: 3
# -->

# In[357]:


# Create the dataframe here
def f5(subframe):
        return len(subframe.index)==2
def f6(series):
    return series.iloc[0]
def f7(series):
    return series.iloc[1]
#scores_pairs_by_business = ...
ins2016 = ins[ins['year'] == 2016]
ins2016_2checks=ins2016.sort_values('date')
#ins2016.groupby(ins2016['business_id'])

#.sort_values('date')
n1_ins2016_2checks=ins2016_2checks.groupby(ins2016['business_id']).filter(f5).set_index('business_id')
n2_ins2016_2checks=n1_ins2016_2checks['score'].groupby('business_id')
first_score=n2_ins2016_2checks.agg(f6)
second_score=n2_ins2016_2checks.agg(f7)
first_score_list=first_score.apply(lambda x: [x])
second_score_list=second_score.apply(lambda x: [x])
scores_pairs_by_business=(first_score_list+second_score_list).to_frame()
scores_pairs_by_business.columns=['score_pair']
scores_pairs_by_business



# In[358]:


ok.grade("q8c1");


# Now, create your scatter plot in the cell below. It does not need to look exactly the same (e.g., no grid) as the above sample, but make sure that all labels, axes and data itself are correct.
# 
# *Hint*: Use `plt.plot()` for the reference line, if you are using matplotlib.
# 
# *Hint*: Use `facecolors='none'` to make circle markers.
# 
# *Hint*: Use `zip()` function to unzip scores in the list.
# <!--
# BEGIN QUESTION
# name: q8c2
# points: 3
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# In[359]:


nn1=first_score.to_frame()
nn1.columns=['First Score']
nn2=second_score.to_frame()
nn2.columns=['Second Score']
frame_for_scatter=nn1.merge(nn2,left_on='business_id',right_on='business_id')
plt.scatter(first_score,second_score,s=38,edgecolors='b',facecolors='none')
#sns.lmplot(x='First Score',y='Second Score',data=frame_for_scatter,ci=False)
x=np.linspace(55,100,101) 
plt.title('First Inspection Score vs. Second Inspection Score')
plt.xlim((55,100))
plt.ylim((55,100))
plt.xticks(np.arange(55, 105,5 ))
plt.plot( [0,1],[0,1] )
plt.plot(x,x,'k-',color='r')
#frame_for_scatter
#['first_score']
#sns.lmplot(x=)


# ### Question 8d
# 
# Another way to compare the scores from the two inspections is to examine the difference in scores. Subtract the first score from the second in `scores_pairs_by_business`. Make a histogram of these differences in the scores. We might expect these differences to be positive, indicating an improvement from the first to the second inspection.
# 
# The histogram should look like this:
# 
# <img src="q8d.png" width=500>
# 
# *Hint*: Use `second_score` and `first_score` created in the scatter plot code above.
# 
# *Hint*: Convert the scores into numpy arrays to make them easier to deal with.
# 
# *Hint*: Use `plt.hist()` Try changing the number of bins when you call `plt.hist()`.
# 
# <!--
# BEGIN QUESTION
# name: q8d
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# In[360]:


diff_series=second_score-first_score
#plt.plot(diff_series)
plt.hist(diff_series,bins=35)
plt.xlim((-30,35))
plt.ylim((0,220))
plt.yticks(np.arange(0, 250,50 ))
plt.title('Distribution of Score Differences')
plt.xlabel('Score Difference (Second Score - First Score)')
plt.ylabel('Count')
#diff_series.hist()


# ### Question 8e
# 
# If a restaurant's score improves from the first to the second inspection, what do you expect to see in the scatter plot that you made in question 8c? What do you see?
# 
# If a restaurant's score improves from the first to the second inspection, how would this be reflected in the histogram of the difference in the scores that you made in question 8d? What do you see?
# 
# <!--
# BEGIN QUESTION
# name: q8e
# points: 3
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# 1.I expect to see the point is above the line. From my obseravation, many of the points are above the line.
# 
# 2.It would be distributed on the right side on the figure. I see a little bit more restaurants are on the right side of the figure (compare to 0).

# ## Summary of the Inspections Data
# 
# What we have learned about the inspections data? What might be some next steps in our investigation? 
# 
# * We found that the records are at the inspection level and that we have inspections for multiple years.   
# * We also found that many restaurants have more than one inspection a year. 
# * By joining the business and inspection data, we identified the name of the restaurant with the worst rating and optionally the names of the restaurants with the best rating.
# * We identified the restaurant that had the largest swing in rating over time.
# * We also examined the relationship between the scores when a restaurant has multiple inspections in a year. Our findings were a bit counterintuitive and may warrant further investigation. 
# 

# ## Congratulations!
# 
# You are finished with Project 1. You'll need to make sure that your PDF exports correctly to receive credit. Run the following cell and follow the instructions.

# In[ ]:





# # Submit
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output.
# **Please save before submitting!**
# 
# <!-- EXPECT 14 EXPORTED QUESTIONS -->

# In[ ]:





# In[ ]:


# Save your notebook first, then run this cell to submit.
import jassign.to_pdf
jassign.to_pdf.generate_pdf('proj1.ipynb', 'proj1.pdf')
ok.submit()


# In[ ]:




