import pandas as pd
from collections import Counter
import numpy as np

DATA_DIR  = r'.\data'
DATA_FILE = r'\Corona_NLP_train.csv'

data = pd.read_csv(DATA_DIR+DATA_FILE)

sentiment = data.iloc[:, 5]
# count each type of sentiment in data
senti_count = dict(Counter(sentiment))
print('sentiment count:', senti_count)

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
# alphal = re.compile(r'http://[a-zA-Z0-9.?/&=:]*|https://[a-zA-Z0-9.?/&=:]*', re.S)
alphal = re.compile("[^a-z|^A-Z]")
url_reg = r'[a-z]*[:.]+\S+'
at_reg = r'[@]+\S+'
# result   = re.sub(url_reg, '', message)
for i in range(len(message)):
    message[i] = re.sub(url_reg, ' ', message[i].strip() )  # - url
    message[i] = re.sub(at_reg, ' ', message[i] )    # - @xxx
    message[i] = re.sub(alphal, " ", message[i])    # keep words
    message[i] = ' '.join(message[i].split())
message  # after blank combined

#####################################
# Question 2
#####################################
print('\n')
print("Question2:")

# # Tokenize(self-made function, but i choose the more formal function below)
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
print('\n')
print("Question3:")

# Plot line chart [x - words][y - words frequencies]
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
# messageLOW=message.lower()
message_rebuild = [[] for i in range(len(message))]
message_list = [[] for i in range(len(message))]

# Here, we delete the words with less than 2 words and stopwords again
# because tokenize was used in the front, which cannot be used in CountVectorizer
for i in range(len(message)):
    for item in message[i].split():
        if item not in stop_words and len( item ) > 2:
            message_rebuild[i].append(item)
            tmp = (' '.join(str(z) for z in message_rebuild[i]))
            message_list[i] = tmp
    # this step is to avoid list in CountVectorizer, else it will report error
    if len(message_list[i]) == 0:
        message_list[i] = str(' ')

count_vec = CountVectorizer()  # Creating a vectorizer object
array = np.array(message_list)
# vector = count_vec.fit_transform([message_list[i] for i in range(len(message_list))])
message_vector = count_vec.fit_transform(array)

# Sum the item in message
message_vector_sum = sum(message_vector.toarray() > 0).tolist()
message_sum = sorted( zip( message_vector_sum, count_vec.get_feature_names() ), key=lambda x: x[0] )

# here we change the position of x_sorted and y_sorted in message_sum
x_sorted = [x[1] for x in message_sum]
y_sorted = [x[0] / len( message ) for x in message_sum]

plt.figure()
plt.plot(x_sorted, y_sorted)
plt.xticks([])
plt.title('The frequencies of words in Tweets')
plt.xlabel('Words')
plt.ylabel('Frequencies')
plt.savefig('figure/words_frequencies.png')

# # ↓↓↓ this code is the first version of plot, but it didn't use CountVectorizer and with very slow speed
# sorted_tweet = sorted(words_count2.items(), key=lambda x : x[1])
# x = [i[0] for i in sorted_tweet if i[1] > 100]
# y = [i[1]/sum(words_count2.values()) for i in sorted_tweet if i[1] > 100]
# plt.plot(x, y)
#
# plt.xticks([])
# # plt.xticks(rotation=60)
# plt.savefig('figure/Line Chart(100).png')
# plt.show()
print("PLOT COMPLETED...")

#####################################
# Question 4
#####################################
print('\n')
print("Question4:")

X = np.array(message)
Y = np.array(sentiment)

cv = CountVectorizer()
X = cv.fit_transform(X)

# Apply SKlearn-MultinomialNB
from sklearn.naive_bayes import MultinomialNB

MNB = MultinomialNB()
MNB.fit(X, Y)

# Predict
predict_sentiment = MNB.predict(X)

# Score the accuracy of predict
from sklearn import metrics
score = metrics.accuracy_score(Y, predict_sentiment)
print("the Error rate of MultinomialNB: ", 1-score)
