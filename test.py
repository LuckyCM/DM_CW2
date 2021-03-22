from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
texts=["dog cat fish","dog cat cat","fish bird", 'bird'] # “dog cat fish” 为输入列表元素,即代表一个文章的字符串
texts = np.array(texts)
cv = CountVectorizer()#创建词袋数据结构
cv_fit=cv.fit_transform(texts)
print(cv_fit.toarray())