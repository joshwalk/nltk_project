
# coding: utf-8

# # SI 330: Homework 3 - To tokenize, or not to tokenize, that is the question: Natural language processing of Shakespearean text with NLTK
# 
# 
# ## Due: Friday, February 2, 2018,  11:59:00pm
# 
# ### Submission instructions
# After completing this homework, you will turn in three files via Canvas ->  Assignments -> Lab 3:
# Your Notebook, named si330-hw3-YOUR_UNIQUE_NAME.ipynb and
# the HTML file, named si330-hw3-YOUR_UNIQUE_NAME.html
# 
# ### Name:  Joshua Walker
# ### Uniqname: joshwalk
# ### People you worked with: I worked by myself.
# 
# ## Top-Level Goal
# To use NLP techniques to determine which characters in Shakespeare's play "Hamlet" are most similar to each other, based on their spoken lines.
# 
# ## Learning Objectives
# After completing this Lab, you should know how to:
# * use NLTK to normalize and tokenize text data
# * calculate type-token ratios (TTR)
# * use NLTK to extract n-grams
# * calculate document similarity using cosines
# 
# 
# ### Note: Suggestions for going "Above and Beyond" 80% are highlighted throughout this notebook.

# ### Outline of Steps For Analysis
# Here's an overview of the steps that you'll need to do to complete this lab.
# 1. Load the raw text
# 2. Iterate through the text, extracting the character and the lines that they say and tokenize the spoken lines.
# 3. Normalize the text
# 4. Remove stopwords
# 5. Calculate type-token ration for each character
# 6. (Only when repeating for bigrams and trigrams): Generate n-grams
# 7. Calculate cosine similarity between each character (we supply the functions to do this)
# 8. List the top 10 most similar characters, based on cosine similarity of the lines they say
# 9. Visualize the results (we supply the code)
# 10. Repeat steps 4 to 7 with bigrams (n-grams where n=2) and trigrams (n-grams where n=3)
# 
# Each of these steps is detailed below.

# Before we start the analysis, let's load the libraries that we'll need.  You should recognize re, nltk, and defaultdict (from collections).  We're also going to peek into the future of this course and use some functionality from pandas (which some of you have already used) and we're going to do some plotting so we'll use matplotlib and Seaborn.  You'll learn more about those later in the course; for now, we're just going to give you code that uses those libraries.

# **NOTE: If the next code cell fails because of missing libraries, install them like you installed nltk (look at the lab for instructions on how to do that)**

# In[160]:

import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from collections import defaultdict
get_ipython().magic('matplotlib inline')


# ## Step 1: Load the text
# 
# Just like we did in the lab, we're going to load the text from one of NLTK's corpora (in this case, the tragic play "Hamlet") into a variable and print the first 500 or so characters.
# 
# **NOTE: If you print substantially more than 500 characters, make sure you clear your output before saving your
# notebook.  Failing to do so may make your notebook unopenable in future sessions.**

# In[161]:

raw_text = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')

# Print the first 500 characters of the text.
print(raw_text[:500])


# Next, we will retrieve the lines of a all the characters from the play and store in a dictionary.  You should store each sentence as a list. The value of your dictionary should be a list of lists for the character.
# 
# ## Step 2: Split lines into character names and spoken lines and tokenize the spoken lines into 
# 
# 
# 
# We suggest you use a dictionary with the key being the character name and the value being a list of lists that correspond the sentences (the outer list) and the tokenized words (the inner lists)</font>
# So you might get a dictionary with keys and values that look something like:
# 
# ```
# {'Bap': [['say', 'what', 'is', 'horatio', 'there'],
#               ['welcome', 'horatio', 'welcome', 'good', 'marcellus'],
#               ['i', 'haue', 'seene', 'nothing']]}
# ```
# ### <font color="magenta">Copy and modify the code you created in the lab to split the character name from their lines.</font>

# In[163]:

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
tokenizer = RegexpTokenizer(r'^\s{2,}([A-Z][a-z]+)\.\s+(.*?)\n(?=\s+)')
char_lines = tokenizer.tokenize(raw_text)

abbrev_dict = {
    'Al': 'All',
    'All': 'All',
    'Amb': 'Ambassador',
    'Bap': 'Baptista',
    'Bapt': 'Baptista',
    'Bar': 'Barnardo',
    'Barn': 'Barnardo',
    'Barnardo': 'Barnardo',
    'Both': 'Both',
    'Cap': 'Captain',
    'Clo': 'Clown',
    'Clown': 'Clown',
    'For': 'Fortinbras',
    'Fortin': 'Fortinbras',
    'Fra': 'Francisco',
    'Fran': 'Francisco',
    'Gen': 'General',
    'Gho': 'Ghost',
    'Ghost': 'Ghost',
    'Guil': 'Guild',
    'Guild': 'Guild',
    'Ha': 'Hamlet',
    'Ham': 'Hamlet',
    'Hamlet': 'Hamlet',
    'Hor': 'Horatio',
    'Hora': 'Horatio',
    'Horat': 'Horatio',
    'Kin': 'King',
    'King': 'King',
    'La': 'Laer',
    'Laer': 'Laer',
    'Lucian': 'Lucian',
    "Mar": 'Marcell',
    'Marcell': 'Marcell',
    'Mes': 'Messenger',
    'Ophe': 'Ophelia',
    'Ophel': 'Ophelia',
    'Osr': 'Osricke',
    'Osricke': 'Osricke',
    'Other': 'Other',
    "Play": 'Player',
    'Player': 'Player',
    'Pol': 'Polon',
    'Polon': 'Polon',
    'Priest': 'Priest',
    'Qu': 'Queen',
    'Queen': 'Queen',
    'Queene': 'Queen',
    'Reynol': 'Reynol',
    'Ro': 'Rosin',
    'Rosin': 'Rosin',
    'Say': 'Saylor',
    'Ser': 'Servant',
    'Volt': 'Volt'

}

char_lines_dict = defaultdict(list)
for char, line in char_lines:
    char_full = abbrev_dict[char]
    tokenized_words = RegexpTokenizer(r'\w+').tokenize(line.lower())
    char_lines_dict[char_full].append(RegexpTokenizer(r'\w+').tokenize(line.lower()))


# When you do this, you will notice that there seem to be a number of different abbreviations used for the same
# characters.  For example, "Clown" is referred to as "Clo", "Clown" and "Clowne".  You should merge those multiple
# abbreviations into the same character, using your best judgement about which ones are the same.  It would be best (**Above and Beyond**) to look up the actual list of characters (use Google) and map the abbreviations onto the
# full names of the characters.  You should also use this opportunity to eliminate stage directions that managed
# to slip through your regular expression and any other questionable lines.
# 
# One way to do this is to create a dictionary that maps abbreviations onto the canonical names and look up the values for each abbreviation.
# 
# ### <font color="magenta">Write code to combine lines from different abbreviations of the same character name.</font>

# In[164]:

# completed above -- I did look up an actual list of characters to make list as accurate as possible


# ## Step 3: Normalize text
# You know how to do this: use ```str.lower()```.
# ### <font color="magenta">Use str.lower() to normalize the spoken lines

# In[165]:

# completed above


# ## Step 4: Remove stopwords
# 
# You used stopwords in an earlier assignment.
# **NOTE: When you work with bigrams and trigrams (see step 10), you should remove bigrams that
# contain stopwords.  For example ('a','zebra') would be removed because 'a' is in the stopword list**.
# 
# 
# ### <font color="magenta">Remove stopwords from your list of words.</font>

# In[166]:

from nltk.corpus import stopwords
dict_no_stop = {}
stop_words = set(stopwords.words('english'))
for k,v in char_lines_dict.items():
    flat_list = [item for sublist in v for item in sublist] # puts the lists of lists of words into one list for each key(character)
    filtered_sentence = [w for w in flat_list if not w in stop_words] # removes stopwords
    dict_no_stop[k] = filtered_sentence

print(dict_no_stop)
    
    


# ## Step 5:  Calculate type-token ratios for each character

# We would like to compare the Type-Token Ratio (TTR) for the different cast members.
# ### <font color="magenta">Use the dictionary created previously and calculate total number of word types, word tokens, and type-token ratio for each character.</font>
# 
# Print the results in a readable, attractive format.</font>

# In[167]:

dict_flat = {} # puts the lists of lists of words into one list for each key(character)
for k,v in char_lines_dict.items():
    flat_list = [item for sublist in v for item in sublist]
    dict_flat[k] = flat_list

for k,v in dict_flat.items():
    word_type_count = len(set(v))
    word_token_count = len(v)
    ttr = word_type_count/word_token_count
    print("{}: types={}, tokens={}, TTR={}".format(k, word_type_count, word_token_count, ttr))


# ## Step 6: Generate n-grams (only when doing Step 10, see below)
# 
# The ```nltk.ngrams(words,n)``` function takes a list of words and a value of n, where n=2 for bigrams and n=3 for trigrams, and return a list of n-tuples.  You can use it to generate bigrams and trigrams.
# ### <font color="magenta">Use ```nltk.ngrams()``` to generate bigrams and trigrams as appropriate.

# In[168]:

bigram_dict = {}
for k,v in dict_flat.items():
    count_dict = defaultdict(int)
    for item in nltk.ngrams(v, 2):
        if item[0] not in stop_words and item[1] not in stop_words:
            count_dict[item] += 1
        bigram_dict[k] = count_dict

trigram_dict = {}
for k,v in dict_flat.items():
    count_dict = defaultdict(int)
    for item in nltk.ngrams(v, 3):
        if item[0] not in stop_words and item[1] not in stop_words:
            count_dict[item] += 1
        trigram_dict[k] = count_dict

    


# ## Step 7: Cosine similarity
# We want to compare the similarity of two n-gram vectors.  A document is a collection of words so we can create a _vector_ of that document with as many dimensions as there are words, and a value (the word count) that represents the length of the vector on that axis.
# 
# Consider the following text:
# >"This course is awesome."
# 
# We could create a vector representation of the normalized (i.e. lowercased) version of it:
# 
# |Word|Count|
# |---|---|
# |this | 1 | 
# | course | 1 | 
# | is | 1 |
# | awesome | 1 |
# 
# Now let's take another text:
# 
# >"This course is a lot of work."
# 
# The vector representation of that (normalized) text would be:
# 
# | Word | Count | 
# | --- | --- |
# | this | 1 |
# | course | 1 |
# | is | 1 | 
# | a | 1 |
# | lot | 1 | 
# | of | 1 |
# | work | 1 |
# 
# If we "align" the two vectors, we get:
# 
# | Word | Count(D1) | Count(D2) |
# | --- | --- | --- |
# | this | 1 | 1 |
# | course | 1 | 1 |
# | is | 1 | 1 |
# | awesome | 1 | 0 |
# | a | 0 | 1 |
# | lot | 0 | 1 | 
# | of | 0 | 1 |
# | work | 0 | 1 |
# 
# To calculate the cosine similarity, we take the dot product (also known as the inner product) of the two
# vectors and "normalize" it to the length of the two vectors.
# 
# #### $cos(\theta)  = {\mathbf{A} \cdot \mathbf{B} \over \|\mathbf{A}\| \|\mathbf{B}\|}$
# 
# Here D1.D2 is the inner product. Let's say we have two word_count dictionaries, D1 = {"and": 3,"of": 2,"the": 5} and D2 =  {"and": 4,"in": 1,"of": 1,"this": 2}. The inner product for D1 and D2 = 14.0 
# 
# #### $cos(\theta) = 1$ means that the two documents are identical
# #### $cos(\theta) = 0$ means that the two documents have no words in common
# 
# We have created the function <b>```cosine_similarity```</b> for you which will calculate the similarity measure. You will need to pass two dictionaries into the function, where the key is the n-gram and the value is the count for the n-gram.

# In[169]:

import math

def cosine_similarity(D1,D2):
    """
    The input is a list of (word,freq) pairs.
    Return the angle between these two vectors.
    """
    numerator = inner_product(D1,D2)
    denominator = math.sqrt(inner_product(D1,D1)*inner_product(D2,D2))
    #return (math.acos(numerator/denominator)/math.pi) * 180
    return (numerator/denominator)

def inner_product(D1,D2):
    """
    Inner product between two vectors, where vectors
    are represented as dictionaries of (word,freq) pairs.
    Example: inner_product({"and":3,"of":2,"the":5},
                           {"and":4,"in":1,"of":1,"this":2}) = 14.0 
    """
    sum = 0.0
    for key in D1:
        if key in D2:
            sum += D1[key] * D2[key]
    return sum


# Using the example from above:

# In[170]:

d1 = {"this": 1, "course": 1, "is": 1, "awesome": 1, "a": 0, "lot": 0, "of": 0, "work": 0}
d2 = {"this": 1, "course": 1, "is": 1, "awesome": 0, "a": 1, "lot": 1, "of": 1, "work": 1}
cosine_similarity(d1,d2)


# If we have two vectors that point in the same direction but are different lengths, the
# angle between them is 0 (i.e. they are identical):

# In[171]:

d1 = {'a': 20, 'b': 30, 'c': 44}
d2 = {'a': 10, 'b': 15, 'c': 22}
cosine_similarity(d1,d2)


# ### <font color="magenta">Use the cosine_similarity function to generate a dictionary whose keys are 2-tuples of characters and values are the cosine similarity between them</font>
# Elements should look something like:
# ```
# (('King', 'Polonius'), 0.8332402081352969)
# ```

# In[172]:

# the following lines create all possible pairs from the list of characters
from itertools import combinations
char_list = [x for x in bigram_dict.keys()]
char_combination_pair = list(combinations(char_list,2))

word_count_dict = {}
for k,v in dict_no_stop.items():
    count_dict = defaultdict(int)
    for item in v:
        count_dict[item] += 1
    word_count_dict[k] = count_dict

similarities = {}
for combo in char_combination_pair:
    if word_count_dict[combo[0]] and word_count_dict[combo[1]]:
        similarities[combo] = cosine_similarity(word_count_dict[combo[0]],word_count_dict[combo[1]])

    


# ## Step 8: Print the top 10 most similar characters, based on cosine similarity of what they said

# In[173]:

sorted_word_count_cs = (sorted(similarities.items(), key=lambda x:x[1], reverse=True))

for item in sorted_word_count_cs[:10]:
    print("{} {}".format(item[0],item[1]))


# ## Step 9: Visualize the similarity matrix
# We're jumping ahead a bit in this step, but we're supplying the code for you.  Assuming your data is in the
# format specified in Step 7 (above), you should be able to simply run the next code block to generate a 
# heatmap of the correlation matrix.

# In[174]:

# You should not have to modify the following code
ser = pd.Series(list(similarities.values()),
                  index=pd.MultiIndex.from_tuples(similarities.keys()))
df = ser.unstack().fillna(0)
df.shape
sns.set(rc={"figure.figsize": (16, 16)})
sns.heatmap(df,cmap="Blues");


# ## Step 10: Repeat Steps 6-9 two more times, once using bigrams and once using trigrams (instead of single words)

# **NOTE: You can copy the code and run it  or you can go _Above and Beyond_ and refactor your code above so that you create functions that take parameters (line the n for n-grams) and then call those functions**

# In[175]:

# bigram similarity
similarities = {}
for combo in char_combination_pair:
    if bigram_dict[combo[0]] and bigram_dict[combo[1]]:
        similarities[combo] = cosine_similarity(bigram_dict[combo[0]],bigram_dict[combo[1]])


# In[176]:

# bigram similarity
sorted_bigrams = (sorted(similarities.items(), key=lambda x:x[1], reverse=True))

for item in sorted_bigrams[:10]:
    print("{} {}".format(item[0],item[1]))


# In[177]:

# bigram visualization
ser = pd.Series(list(similarities.values()),
                  index=pd.MultiIndex.from_tuples(similarities.keys()))
df = ser.unstack().fillna(0)
df.shape
sns.set(rc={"figure.figsize": (16, 16)})
sns.heatmap(df,cmap="Blues");


# In[178]:

# trigram similarity
similarities = {}
for combo in char_combination_pair:
    if trigram_dict[combo[0]] and trigram_dict[combo[1]]:
        similarities[combo] = cosine_similarity(trigram_dict[combo[0]],trigram_dict[combo[1]])


# In[179]:

# trigram similarity
sorted_trigrams = (sorted(similarities.items(), key=lambda x:x[1], reverse=True))

for item in sorted_trigrams[:10]:
    print("{} {}".format(item[0],item[1]))


# In[180]:

# trigram visualization
ser = pd.Series(list(similarities.values()),
                  index=pd.MultiIndex.from_tuples(similarities.keys()))
df = ser.unstack().fillna(0)
df.shape
sns.set(rc={"figure.figsize": (16, 16)})
sns.heatmap(df,cmap="Blues");


# ### Above and Beyond Possibility
# You might want to consider running the above analyses on other Shakespearean tragedies and compare the results to these.

# ## Macbeth analysis
# I ran the analyses on Macbeth, which I have attached as a separate file.
# 
# Comparing the first calculated cosine similarities, Hamlet’s were much higher— the top 10 were almost all higher than Macbeth’s #1. As far as bigram similarity comparison, Macbeth’s were much higher. The Lords were in the top 5 for Macbeth; they must not have had too many lines/words. It is difficult to compare visualizations because seaborne placed them on different scales. Trigram cosine similarities were < .1 for both, where for Macbeth only 9 were above 0.

# In[ ]:



