from .result_builder import build_result
from .analizer import analize
import pandas as pd
from .custom_logger import configure_logger, set_stdout_handler

logger = configure_logger()
logger = set_stdout_handler(logger)

def analize_hse(search):
    df = pd.read_pickle("C:\hse_processed (5).pkl") 

    count = df.shape[0]
    count_cons = 0
    count_neur = 0
    count_agree = 0
    count_open = 0
    count_extr = 0
    full_interests = []
    for _, row in df.iterrows():
        interests = list(row['interests'])
        full_interests.extend(interests)
        cons = row['cons']
        if cons == 1:
            count_cons += 1
        agree = row['agree']
        if agree == 1:
            count_agree += 1
        neur = row['neur']
        if neur == 1:
            count_neur += 1
        openn = row['open']
        if openn == 1:
            count_open += 1
        extr = row['extr']
        if extr == 1:
            count_extr += 1
    full_interests = set(full_interests)
    countries = df['country'].unique()
    country = ', '.join(countries)
    cities = df['city'].unique()
    city = ', '.join(cities)
    openn = str(round(((count_open / count) * 100), 2)) + "%"
    cons =str( round(((count_cons / count) * 100), 2))+ "%"
    neur = str(round(((count_neur / count) * 100), 2))+ "%"
    agree = str(round(((count_agree / count) * 100), 2))+ "%"
    extr = str(round(((count_extr / count) * 100), 2))+ "%"
    df.loc[df["age"] == "Не указано", "age"] = 0

    df["age"] = df["age"].astype("int64")

    min_age = df[df["age"] != 0]["age"].min()
    max_age = df[df["age"] != 0]["age"].max()
    age = str(min_age) + " - " + str(max_age)
    interests = full_interests
    build_result(
        search,
        '-',
        '-',
        age,
        country,
        city,
        openn,
        cons,
        neur,
        agree,
        extr,
        interests,
    )

def analize_miem(search):
    
    df = pd.read_pickle("C:\processed (6).pkl") 
    #df = pd.read_pickle("C:\miem_preprocessed.pkl") 
    #df = prepare_df('C:\miem_preprocessed.csv')
    count = df.shape[0]
    count_cons = 0
    count_neur = 0
    count_agree = 0
    count_open = 0
    count_extr = 0
    full_interests = []
    for _, row in df.iterrows():
        interests = list(row['interests'])
        full_interests.extend(interests)
        cons = row['cons']
        if cons == 1:
            count_cons += 1
        agree = row['agree']
        if agree == 1:
            count_agree += 1
        neur = row['neur']
        if neur == 1:
            count_neur += 1
        openn = row['open']
        if openn == 1:
            count_open += 1
        extr = row['extr']
        if extr == 1:
            count_extr += 1
    full_interests = set(full_interests)
    countries = df['country'].unique()
    country = ', '.join(countries)
    cities = df['city'].unique()
    city = ', '.join(cities)
    openn = str(round(((count_open / count) * 100), 2)) + "%"
    cons =str( round(((count_cons / count) * 100), 2))+ "%"
    neur = str(round(((count_neur / count) * 100), 2))+ "%"
    agree = str(round(((count_agree / count) * 100), 2))+ "%"
    extr = str(round(((count_extr / count) * 100), 2))+ "%"
    df["age"] = df["age"].astype("int64")
    min_age = df[df["age"] != 0]["age"].min()
    max_age = df[df["age"] != 0]["age"].max()
    age = str(min_age) + " - " + str(max_age)
    interests = full_interests
    build_result(
        search,
        '-',
        '-',
        age,
        country,
        city,
        openn,
        cons,
        neur,
        agree,
        extr,
        interests,
    )
