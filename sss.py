import re
import pandas as pd

d = pd.read_csv(r'.\data\Corona_NLP_train.csv')
d = d.iloc[:, 4]

alphal = re.compile("[^a-z|^A-Z]")
for i in range(len( d )):
    d[i] = alphal.sub(" ", d[i].strip())
    d[i] = ' '.join(d[i].split())
d = list(d)  # after blank combined

word_tokens = []
from nltk.tokenize import word_tokenize
for i in range(len( d )):
    word_tokens.append(word_tokenize(d[i]))

TokenAndStopAnd2 = []
for i in range(len(word_tokens)):
    TokenAndStopAnd2.append(re.sub(r'\b\w{1,2}\b', '', str(word_tokens[i])))
w=str(word_tokens[1])
q=word_tokens[1][1]
print(TokenAndStopAnd2)