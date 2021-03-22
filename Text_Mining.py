import pandas as pd
from collections import Counter
import numpy as np

DATA_DIR  = r'.\data'
DATA_FILE = r'\Corona_NLP_train.csv'

data = pd.read_csv(DATA_DIR+DATA_FILE)

sentiment = data.iloc[:, 5]
senti_count = dict(Counter(sentiment))
# count each type of sentiment in data
print(senti_count)

#####################################
# Question 1
#####################################
print('\n')
print("Question1:")
# count possibility of each type of sentiment in data
print("The possible sentiments:",
      {key: format((value/len(sentiment)), '.4f') for key, value in senti_count.items() if value > 1})

# the second most popular sentiment in the tweets
sort = sorted(senti_count.items(), key=lambda d: d[1])
print("the second most popular sentiment in the tweets:",
      sort[len(sort)-2])

# the date with the greatest number of extremely positive tweets
date = []
for key, value in sentiment.items():
    if ('Extremely Positive' == value):
        date.append(data.iloc[key, 3])
date_count = max(date, key=date.count)
print("the date with the greatest number of extremely positive tweets:", date_count)

# transform the data
message = data.iloc[:, 4]

# message = str.lower(str(message))
message = [s.lower() for s in message]

import re
alphal = re.compile("[^a-z|^A-Z]")
for i in range(len( message )):
    message[i] = alphal.sub(" ", message[i].strip())
    message[i] = ' '.join(message[i].split())
message  # after blank combined

#####################################
# Question 2
#####################################
print('\n')
print("Question2:")

# # Tokenize(self-made)
# tokenized_message = []
# for i in range(len( message )):
#     tokenized_message.append( message[i].split() )
# # print(list_message)

# Regular Tokenize
word_tokens = []
from nltk.tokenize import word_tokenize
for i in range(len( message )):
    word_tokens.append(word_tokenize(message[i]))

# Count words number
words_count = dict(Counter(word for sentence in message for word in sentence.split()))
print("Words count: ", sum(words_count.values()))

# Count words with no repetition
words_count_norep = dict(Counter(words_count.keys()))
print("Words count with no repetition: ", len(words_count_norep))

# The 10 most frequent words in the corpus
word_sorted = sorted(words_count.items(), key=lambda x: x[1], reverse=True)
print("The 10 most frequent words:")
for i in range(0, 10):
    print("[", i+1, "]: ", word_sorted[i])

# remove stop words
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

OAGTokensWOStop = []
for item in word_tokens:
    temp = []
    for tweet in item:
        if tweet not in stop_words:
            temp.append(tweet)
    OAGTokensWOStop.append(temp)

# remove the word that character less and equal to 2
TokenAndStopAnd2 = [[] for i in range(len(message))]
# for i in range(len(OAGTokensWOStop)):
#     TokenAndStopAnd2.append(re.sub(r'\b\w{1,2}\b', '', (OAGTokensWOStop[i])))
for i in range(len(OAGTokensWOStop)):
    for item in range(len(OAGTokensWOStop[i])):
        if len(OAGTokensWOStop[i][item]) >= 3:
            TokenAndStopAnd2[i].append(OAGTokensWOStop[i][item])

# Count words number
words_count2 = dict(Counter(word for sentence in TokenAndStopAnd2 for word in sentence))
# words_count2 = dict()
# for i in range(len(TokenAndStopAnd2)):
#     words_count2[i] = (len( TokenAndStopAnd2[i] ))
print("[Removed]Words count: ", sum(words_count2.values()))

# Count words with no repetition
words_count_norep2 = dict(Counter(words_count2.keys()))
print("[Removed]Words count with no repetition: ", len(words_count_norep2))

# The 10 most frequent words in the corpus
word_sorted2 = sorted(words_count2.items(), key=lambda x: x[1], reverse=True)
print("[Removed]The 10 most frequent words:")
for i in range(0, 10):
    print("[", i+1, "]: ", word_sorted2[i])


#####################################
# Question 3
#####################################

# Plot line chart [x - words][y - words frequencies]
import matplotlib.pyplot as plt

# x = words_count2.keys()
# y = {k: v / total for total in (sum(words_count2.values()),) for k, v in words_count2.items()}
sorted_tweet = sorted(words_count2.items(), key=lambda x : x[1])
x = [i[0] for i in sorted_tweet[79250:]]
y = [i[1] for i in sorted_tweet[79250:]]

plt.plot(x, y)

plt.xticks(rotation=60)
plt.show()


#####################################
# Question 4
#####################################
print("v")
