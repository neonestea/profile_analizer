# -*- coding: utf-8 -*-
"""groups.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vIcOOKZAZU2XA_rc1oF3oKT792-qiymO
"""

import pandas as pd

name1 = "2024-02-29_11_18_40.xlsx"
name2 = "2024-02-29_16_47_03.xlsx"
name3 = "2024-03-03_12_40_01.xlsx"

df1 = pd.read_excel(name1, header = 1)

df2 = pd.read_excel(name2, header = 1)

df3 = pd.read_excel(name3, header = 1)

df = pd.concat([df1, df2])
df = pd.concat([df, df3])

groups = df['Владелец'].unique()

groups

openness = df[df['Владелец'] == 'Хабр']

not_openness = df[df['Владелец'] == 'Консерватор']

conscientiousness = df[df['Владелец'] == 'ПБК ЦСКА']
not_conscientiousness = df[df['Владелец'] == 'Паша и его прокрастинация']
extroversion =  df[df['Владелец'] == 'Странная Планета / Strange Planet']
not_extroversion =  df[df['Владелец'] == 'Грусть']
agreeableness = df[df['Владелец'] == 'Благотворительный фонд Дети-ждут.рф']
not_agreeableness = df[df['Владелец'] == 'Черный список | Рынок Садовод']
neuroticism = df[df['Владелец'] == 'Типичный невротичный (ТЫ)']
not_neuroticism = df[df['Владелец'] == 'Psychology|Психология']

texts = ' '.join(agreeableness['Текст'].tolist())

!pip install keybert

comments = pd.read_csv('not_extroversioncomments.csv', delimiter=';')

texts += (str(comments['text'].tolist()))
#openness_texts += (str(not_open_comments['text'].tolist()))

# script.py
from keybert import KeyBERT

kw_model = KeyBERT()

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
russian_stopwords = stopwords.words("russian")

# Commented out IPython magic to ensure Python compatibility.
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# %matplotlib inline

result = kw_model.extract_keywords(texts, keyphrase_ngram_range=(1, 1), stop_words=russian_stopwords, top_n=30)

text_raw = ''
for k, v in result:
    text_raw += ' '
    text_raw += k
wordcloud = WordCloud().generate(text_raw)
plt.figure(figsize = (5, 5), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()