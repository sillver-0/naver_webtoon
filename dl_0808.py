#!/usr/bin/env python
# coding: utf-8

# In[1]:


#-*- coding:utf-8 -*-
import json
import os
from pprint import pprint
from konlpy.tag import Okt
import numpy as np
import nltk
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow import keras
okt = Okt()


# In[2]:


def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data


# In[3]:


def read_comment(filename):
    with open(filename, 'r', encoding = 'utf-8') as f:
        data = [line for line in f.read().splitlines()]
        data = data[1:]
    return data


# In[4]:


train_data = read_data('comment_list/comment_train.txt')
test_data = read_data('comment_list/comment_test.txt')


# In[5]:


print(len(train_data))
print(len(train_data[0]))
print(len(test_data))
print(len(test_data[0]))


# In[6]:


train_data


# In[6]:


def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]


# In[7]:


if os.path.isfile('train_docs.json'):
    with open('train_docs.json', encoding='UTF-8') as f:
        train_docs = json.load(f)
    with open('test_docs.json', encoding='UTF-8') as f:
        test_docs = json.load(f)
else:
    train_docs = [(tokenize(row[0]), row[1]) for row in train_data]
    test_docs = [(tokenize(row[0]), row[1]) for row in test_data]
    # JSON 파일로 저장
    with open('train_docs.json', 'w', encoding = 'UTF-8') as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent="\t")
    with open('test_docs.json', 'w', encoding = 'UTF-8') as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent="\t")


# In[18]:


type(train_docs[0])


# In[8]:


pprint(train_docs[0])


# In[9]:


tokens = [t for d in train_docs for t in d[0]]
print(len(tokens))
print(tokens[:10])


# In[10]:


text = nltk.Text(tokens, name='NMSC')

# 전체 토큰의 개수
print(len(text.tokens))

# 중복을 제외한 토큰의 개수
print(len(set(text.tokens)))            

# 출현 빈도가 높은 상위 토큰 10개
pprint(text.vocab().most_common(10))


# In[11]:


selected_words = [f[0] for f in text.vocab().most_common(4400)]


# In[12]:


def term_frequency(doc):
    return [doc.count(word) for word in selected_words]


# In[13]:


train_x = [term_frequency(d) for d, _ in train_docs]
test_x = [term_frequency(d) for d, _ in test_docs]
train_y = [c for _, c in train_docs]
test_y = [c for _, c in test_docs]
print(type(train_y[0]))
x_train = np.asarray(train_x).astype('float32')
x_test = np.asarray(test_x).astype('float32')

y_train = np.asarray(train_y).astype('float32')
y_test = np.asarray(test_y).astype('float32')


# In[14]:


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(4400,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[15]:


model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
             loss = losses.binary_crossentropy,
             metrics = [metrics.binary_accuracy])


# In[16]:


model.fit(x_train, y_train, epochs = 6, batch_size = 256)
results = model.evaluate(x_test, y_test)
print(results)


# In[17]:


def predict_pos_neg(review):
    token = tokenize(review)
    tf = term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    
    return score


# In[19]:


pos = 0
neg = 0
file = read_comment('comment_list/label/a/후덜덜덜남극전자.txt')

for k in file:
    score = predict_pos_neg(k)
    if (score > 0.7):
        pos += 1
    elif (score > 0.5 and score <= 0.7):
        pos += 0.5
    elif (score <= 0.5 and score > 0.3):
        neg += 0.5
    elif (score <= 0.3):
        neg += 1
        
all = pos + neg
pos_per = pos / all * 100
neg_per = neg / all * 100

print("긍정확률: {:.2f}%".format(pos_per))
print("부정확률: {:.2f}%".format(neg_per))


# In[25]:


predict_pos_neg("너무 재밌어요!!")
predict_pos_neg("개노잼")

