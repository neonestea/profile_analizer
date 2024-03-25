'''Module to analyze information'''
from .models import ProfileInfo, Result
from datetime import datetime
from .custom_logger import configure_logger, set_stdout_handler
from .result_builder import build_result
import pandas as pd
import emoji
from .tokenization import RegexTokenizer
from .dostoevsky_models import FastTextSocialNetworkModel
import nltk
import string
import re

from pymystem3 import Mystem
from string import punctuation
from nltk.corpus import stopwords
from autocorrect import Speller
import gensim
import gensim.corpora as corpora
from gensim import models


logger = configure_logger()
logger = set_stdout_handler(logger)
'''logger'''
nltk.download("stopwords")
nltk.download("punkt")
spell = Speller(lang="ru")
'''speller'''
russian_stopwords = stopwords.words("russian")
'''stopwords'''


def preprocess(profiles):
    pass


def get_open(profile):
    pass


def get_neur(profile):
    pass


def get_cons(profile):
    pass


def get_agree(profile):
    pass


def get_extr(profile):
    pass


def create_line(profile):
    """Creates dictionary from one profile to pass to personality traits classificators.
    
    :param profile:      Profile to create dictionary.
    :type profile:       ProfileInfo

    :return: Returns dict from profile to add to dataset
    :rtype: dict
    """
    line = {}
    # line['user_id'] = '1'
    line["country"] = profile.country
    line["city"] = profile.city
    line["bdate"] = profile.bdate
    line["interests"] = profile.interests
    line["books"] = profile.books
    line["tv"] = profile.tv
    line["games"] = profile.games
    line["movies"] = profile.movies
    line["activities"] = profile.activities
    line["music"] = profile.music
    line["status"] = profile.status
    line["military"] = profile.military
    line["university_name"] = profile.university_name
    line["faculty"] = profile.faculty
    line["home_town"] = profile.home_town
    line["relation"] = profile.relation
    line["sex"] = profile.sex
    line["about"] = profile.about
    line["friends_count"] = profile.friends_count
    line["followers_count"] = profile.followers_count
    line["groups"] = profile.groups
    line["posts"] = profile.posts
    line["photos"] = profile.photos_count
    line["comments"] = profile.comments
    line["comments_of_others"] = profile.comments_of_other_users

    line["smoking_0"] = 0
    line["smoking_1"] = 0
    line["smoking_2"] = 0
    line["smoking_3"] = 0
    line["smoking_4"] = 0
    line["smoking_5"] = 0
    line["alcohol_0"] = 0
    line["alcohol_1"] = 0
    line["alcohol_2"] = 0
    line["alcohol_3"] = 0
    line["alcohol_4"] = 0
    line["alcohol_5"] = 0
    line["life_main_0"] = 0
    line["life_main_1"] = 0
    line["life_main_2"] = 0
    line["life_main_3"] = 0
    line["life_main_4"] = 0
    line["life_main_5"] = 0
    line["life_main_6"] = 0
    line["life_main_7"] = 0
    line["life_main_8"] = 0
    line["people_main_0"] = 0
    line["people_main_1"] = 0
    line["people_main_2"] = 0
    line["people_main_3"] = 0
    line["people_main_4"] = 0
    line["people_main_5"] = 0
    line["people_main_6"] = 0
    line["relation_0"] = 0
    line["relation_1"] = 0
    line["relation_2"] = 0
    line["relation_3"] = 0
    line["relation_4"] = 0
    line["relation_5"] = 0
    line["relation_6"] = 0
    line["relation_7"] = 0
    line["relation_8"] = 0
    line["political_0"] = 0
    line["political_1"] = 0
    line["political_2"] = 0
    line["political_3"] = 0
    line["political_4"] = 0
    line["political_5"] = 0
    line["political_6"] = 0
    line["political_7"] = 0
    line["political_8"] = 0
    line["political_9"] = 0
    line["political_10"] = 0
    line["sex_1.0"] = 0
    line["sex_2.0"] = 0
    line["sex_"] = 0
    line["military_"] = 0
    line["military_0"] = 0
    line["military_False"] = 0
    line["military_True"] = 0
    line["religion"] = profile.religion

    line["smoking_" + str(profile.smoking)] = 1
    line["alcohol_" + str(profile.alcohol)] = 1
    line["life_main_" + str(profile.life_main)] = 1
    line["people_main_" + str(profile.people_main)] = 1
    line["relation_" + str(profile.relation)] = 1
    line["sex_" + str(profile.sex)] = 1
    line["military_" + str(profile.military)] = 1
    line["political_" + str(profile.political)] = 1
    # print(line)
    return line


def count_sentiments(texts):
    """Counts sentiment texts.
    
    :param texts:      Texts to check sentiments.
    :type texts:       list

    :return: Returns count of sentiments
    :rtype: tuple
    """
    tokenizer = RegexTokenizer()
    model = FastTextSocialNetworkModel(tokenizer=tokenizer)

    positive_count = 0
    negative_count = 0
    neutral_count = 0
    speech_count = 0
    uncertain_count = 0
    emojis = 0
    results = model.predict(texts, k=2)
    for message, sentiment in zip(texts, results):
        emojis += emoji.emoji_count(message)
        v = list(sentiment.values())
        k = list(sentiment.keys())
        sentiment = k[v.index(max(v))]
        if sentiment == "speech":
            speech_count += 1
        elif sentiment == "positive":
            positive_count += 1
        elif sentiment == "negative":
            negative_count += 1
        elif sentiment == "neutral":
            neutral_count += 1
        else:
            uncertain_count += 1
    return (
        positive_count,
        negative_count,
        neutral_count,
        speech_count,
        uncertain_count,
        emojis,
    )


def add_count_sentiment_columns(df):
    """Adds sentiment count columns.
    
    :param df:      Input Dataframe.
    :type df:       DataFrame

    :return: Returns Dataframe with new columns.
    :rtype: DataFrame
    """
    result_df = df
    positives = []
    negatives = []
    neutrals = []
    speechs = []
    uncertains = []
    emojis = []
    full_texts = []
    for index, row in df.iterrows():
        # user_id = row['user_id']
        texts = row["comments"]
        texts = texts.split("', '")
        # print(row['comments'].split('\,')[0])
        # print(texts)
        texts.extend(row["posts"].split("', '"))
        texts.extend(row["comments_of_others"].split("', '"))
        texts.append(row["religion"])
        texts.append(row["status"])
        texts.append(row["interests"])
        texts.append(row["books"])
        texts.append(row["tv"])
        texts.append(row["games"])
        texts.append(row["movies"])
        texts.append(row["activities"])
        texts.append(row["music"])
        texts.append(row["about"])

        # print(texts)
        # break
        full_text = " ".join(t for t in texts)
        full_texts.append(full_text)
        # print(texts)
        # print(full_text)

        (
            positive_count,
            negative_count,
            neutral_count,
            speech_count,
            uncertain_count,
            emoji,
        ) = count_sentiments(texts)

        positives.append(positive_count)
        negatives.append(negative_count)
        neutrals.append(neutral_count)
        speechs.append(speech_count)
        uncertains.append(uncertain_count)
        emojis.append(emoji)

    # print(len(positives))  #break
    result_df = result_df.assign(positives=positives)
    result_df = result_df.assign(negatives=negatives)
    result_df = result_df.assign(neutrals=neutrals)
    result_df = result_df.assign(speechs=speechs)
    result_df = result_df.assign(uncertains=uncertains)
    result_df = result_df.assign(emojis=emojis)
    result_df = result_df.assign(full_texts=full_texts)
    return result_df


def preprocess_text(text):
    """Preprocesses text.
    
    :param text:      Input text.
    :type text:       str

    :return: Returns preprocessed text.
    :rtype: str
    """
    mystem = Mystem()
    pattern_reply = "\[id[0-9]*\|[a-zA-Zа-я--Я]*\], "
    text = re.sub(pattern_reply, "", text)
    # print(text)
    # text = re.sub('[^a-zA-Zа-яА-Я]+ ', '', text)

    text = text.lower()
    text = text.translate(string.punctuation)
    tokens = mystem.lemmatize(text)
    tokens = [re.sub("[^a-zA-Zа-яА-Я]+", "", t) for t in tokens]
    tokens = [
        spell(token)
        for token in tokens
        if token not in russian_stopwords  # and not token.isalpha()
        and token != " "
        and token != "nan"
        and "..." not in token
        and token.strip() not in punctuation
    ]
    # print(tokens)
    text = " ".join(tokens)

    return text


def make_prepr(df):
    """Preprocesses texts in dataframe.
    
    :param df:      Input Dataframe.
    :type df:       DataFrame

    :return: Returns Dataframe with preprocessed texts.
    :rtype: DataFrame
    """
    # counter = 0
    result_df = df
    new_texts = []
    for index, row in df.iterrows():
        text = row["full_texts"]
        new_text = preprocess_text(text)
        new_texts.append(new_text)
    # print(new_text)
    # print(counter)
    # break
    # counter += 1
    result_df = result_df.assign(full_texts=new_texts)
    return result_df


def preprocess_df(new_df):
    """Cleanes dataset and processes columns.
    
    :param new_df:      Input Dataframe.
    :type new_df:       DataFrame

    :return: Returns Dataframe with new columns.
    :rtype: DataFrame
    """
    df = new_df
    df = df.replace("Не указано", "")
    df = df.fillna("")
    df.relation[df.relation == ""] = "0"
    df["groups"] = df["groups"].astype("int64")
    df.photos[df.photos == ""] = "0"

    df["photos"] = df["photos"].astype("int64")

    # df = pd.get_dummies(df, columns=['smoking', 'alcohol', 'life_main', 'people_main', 'relation', 'political', 'sex', 'military'], drop_first= False )

    df = add_count_sentiment_columns(df)
    # df = df.drop('user_id', axis=1)
    df = make_prepr(df)
    df = df.drop("comments_of_others", axis=1)
    df = df.drop("status", axis=1)
    df = df.drop("posts", axis=1)
    df = df.drop("comments", axis=1)
    df = df.drop("religion", axis=1)
    df = df.drop("interests", axis=1)
    df = df.drop("books", axis=1)
    df = df.drop("tv", axis=1)
    df = df.drop("games", axis=1)
    df = df.drop("movies", axis=1)
    df = df.drop("activities", axis=1)
    df = df.drop("music", axis=1)
    df = df.drop("about", axis=1)
    df = df.replace("", 0)
    df = df.fillna(0)
    df["bdate"] = df["bdate"].astype("int64")
    return df


def get_topic_by_id(id):
    """Returns topic by id.
    
    :param id:      Input id.
    :type id:       int

    :return: Returns name of topic.
    :rtype: str
    """
    if id == 0:
        return ""
    elif id == 1:
        return "Авто"
    elif id == 2:
        return "IT"
    elif id == 3:
        return "Красота"
    elif id == 4:
        return "Новости"
    elif id == 5:
        return "Бизнес"
    elif id == 6:
        return "Мероприятия"
    elif id == 7:
        return ""
    elif id == 8:
        return "Факты"
    elif id == 9:
        return "Отдых и туризм"
    elif id == 10:
        return "Отношения между людьми"
    elif id == 11:
        return "Искусство"
    elif id == 12:
        return "Обучение"
    elif id == 13:
        return "Недвижимость"
    elif id == 14:
        return "Семья и дом"
    elif id == 15:
        return "Новости"
    elif id == 16:
        return "Товары и услуги"
    elif id == 17:
        return "Факты"
    elif id == 18:
        return "Животные"
    elif id == 19:
        return "Фото"


def get_interests(profile):
    """Defines interests of a person.
    
    :param profile:      Input Profile.
    :type profile:       ProfileInfo

    :return: Returns interests.
    :rtype: list
    """
    interests_data = []
    to_process = []
    if len(profile.interests) != 0:
        to_process.append(profile.interests)
    if len(profile.books) != 0:
        interests_data.append("КниГи")
    if len(profile.tv) != 0:
        interests_data.append("ТВ")
    if len(profile.games) != 0:
        interests_data.append("Игры")
    if len(profile.movies) != 0:
        interests_data.append("Фильмы")
    if len(profile.music) != 0:
        interests_data.append("Музыка")
    for item in profile.group_infos.split("==="):
        descr_act = item.split("Activity: ")
        if len(descr_act) != 0 and len(descr_act[0]) != 0:
            activity = descr_act[1]
            interests_data.append(activity)
        if len(item) != 0:

            to_process.append(item)

    # print("Interests_data: ", interests_data)
    data_norm = [preprocess_text(t).split() for t in to_process]
    lda_model = gensim.models.ldamodel.LdaModel.load("C:\LDA_model.model")
    loaded_dict = corpora.Dictionary.load("C:\DictionaryInterests.sav")
    # print("data_norm: ", data_norm)

    for text in data_norm:
        query = [loaded_dict.doc2bow(text)]
        vector = lda_model[query]
        vector = sorted(vector[0], key=lambda x: (x[1]), reverse=True)
        dominant_topic = 0
        for j, (topic_num, topic_contrib) in enumerate(vector):
            if j == 0:  # => dominant topic
                wp = lda_model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                topic_num = int(topic_num)
                dominant_topic = int(topic_num)
        topic = get_topic_by_id(dominant_topic)
        if len(topic) != 0:

            interests_data.append(topic)

    return interests_data


def analize(search):
    """Analize search.
    
    :param search:      Input search.
    :type search:       Search
    """
    profiles = [el for el in ProfileInfo.objects.filter(connected_search=search)]

    if len(profiles) == 1:
        logger.debug("analize: one person")
        line = create_line(profiles[0])
        users_info = pd.DataFrame(line, index=[0])
        # users_info = users_info.append(line_df, ignore_index=True)
        users_info = preprocess_df(users_info)
        print(users_info.columns)
        print(users_info.values)

        first_name = profiles[0].first_name
        last_name = profiles[0].last_name
        age = datetime.now().year - profiles[0].bdate
        city = profiles[0].city
        country = profiles[0].country
        open = "NO"
        cons = "NO"
        neur = "NO"
        agree = "NO"
        extr = "NO"
        interests = "NOT IMPLEMENTED"

        interests = set(get_interests(profiles[0]))

    else:
        logger.debug("analize: many people")
        users_info = None
        full_interests = []
        for profile in profiles:
            line = create_line(profile)
            users_info = pd.DataFrame(line, index=[0])
            users_info = preprocess_df(users_info)
            print(users_info.columns)
            print(users_info.values)
            interests = get_interests(profile)
            full_interests.extend(interests)
        full_interests = set(full_interests)
        count = len(profiles)
        first_name = "-"
        last_name = "-"
        ages = [datetime.now().year - int(el.bdate) for el in profiles]
        min_age = min(ages)
        max_age = max(ages)
        if min_age == max_age:
            age = str(min_age)
        else:
            age = str(min_age) + " - " + str(max_age)
        # TODO traits and interests
        countries = set([el.country for el in profiles])
        cities = set([el.city for el in profiles])
        country = ", ".join(countries)
        city = ", ".join(cities)
        open = "1%"
        cons = "1%"
        neur = "1%"
        agree = "1%"
        extr = "1%"
        interests = full_interests
    build_result(
        search,
        first_name,
        last_name,
        age,
        country,
        city,
        open,
        cons,
        neur,
        agree,
        extr,
        interests,
    )
