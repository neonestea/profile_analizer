# -*- coding: utf-8 -*-
"""comments_collector.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Bv4Rs6iJYU5Yebu7dlwwfe1BtT-zLOB-
"""

import warnings
warnings.filterwarnings("ignore")

name = 'not_conscientiousness.csv'

import pandas as pd

def prepare_df(filename):
  df = pd.read_csv(filename, delimiter = ';', encoding = 'utf-8')
  df = df.drop(columns=['Unnamed: 0'])
  return df

df = prepare_df(name)

df.head()

!pip install vk_api

import vk_api
from datetime import datetime
#from .models import ProfileInfo, GroupInfo
import uuid
#from .custom_logger import configure_logger, set_stdout_handler

session = vk_api.VkApi(token="vk1.a.Gi9VLR0O3w9yu6Mf30_rFTibl20zp6XMcjmn_sodhGHSH1tWW8_nYW9NZ9DavAhtQFnxA1sVeZ5b5754Rtzio8J8sDoCB6XFN4MX0QgbiN548Wc8xKFnqvv2Avqjb7O5xps2J8QSEpIwfR93NNq2aBKymo0KCdqiTSMXfKB36AT-x7xcnBjxTmvXxnaW4WVLniAlDvn2nVjDOCGcrjFS5Q")
vk = session.get_api()
#date_format = "%d.%m.%Y"
#logger = configure_logger()
#logger = set_stdout_handler(logger)

def get_comments_for_post(group_id, post_id, df = None):
  comments = []
  wall = None
  try:
    comments = vk.wall.getComments(owner_id = group_id, post_id = post_id)
    comments = comments['items']
  except vk_api.exceptions.ApiError:
    return df
  #print(len(comments))
  for comment in comments:
    user_id = comment['from_id']
    text = comment['text']
    result = {'user_id': user_id, 'text': text}
    #print(result)
    df = df.append(result, ignore_index = True)
  return df

def get_all_comments(df):
  users_comments = pd.DataFrame(
        [],
        columns=["user_id", "text"]
    )
  for index, row in df.iterrows():
    group_and_post = row['ID поста']
    group_id, post_id = group_and_post.split('_')
    users_comments = get_comments_for_post(group_id, post_id, users_comments)

  return users_comments

users_comments = get_all_comments(df)

users_comments.head()

users_comments.to_csv(name.split('.')[0] + 'comments.csv', sep=';', encoding='utf-8')