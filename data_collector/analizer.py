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
import pickle

from pymystem3 import Mystem
from category_encoders.hashing import HashingEncoder
from string import punctuation
from nltk.corpus import stopwords
from autocorrect import Speller
import gensim
import gensim.corpora as corpora
from gensim import models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from catboost import CatBoostClassifier


logger = configure_logger()
logger = set_stdout_handler(logger)
'''logger'''
nltk.download("stopwords")
nltk.download("punkt")
spell = Speller(lang="ru")
'''speller'''
russian_stopwords = stopwords.words("russian")
'''stopwords'''

open_classifier = pickle.load(open("C:\saved_models\open_classifier.pickle", "rb"))
neur_classifier = pickle.load(open("C:\saved_models\\neur_classifier.pickle", "rb"))
extr_classifier = pickle.load(open("C:\saved_models\extr_classifier.pickle", "rb"))
agree_classifier = pickle.load(open("C:\saved_models\\agree_classifier.pickle", "rb"))
cons_classifier = pickle.load(open("C:\saved_models\cons_classifier.pickle", "rb"))
extr_tfidfconverter = pickle.load(open("C:\saved_models\extr_tf_idf.pickle", "rb"))
open_tfidfconverter = pickle.load(open("C:\saved_models\open_tf_idf.pickle", "rb"))
neur_tfidfconverter = pickle.load(open("C:\saved_models\\neur_tf_idf.pickle", "rb"))
agree_tfidfconverter = pickle.load(open("C:\saved_models\\agree_tf_idf.pickle", "rb"))
cons_vectorizer = pickle.load(open("C:\saved_models\cons_count_vectorizer.pickle", "rb"))
cons_tfidfconverter = pickle.load(open("C:\saved_models\cons_tf_idf.pickle", "rb"))
interests_tokenizer = RegexTokenizer()
interests_model = FastTextSocialNetworkModel(tokenizer=interests_tokenizer)


def encode_with_hashing(x_df, trait):
    """Classifies extraversion.
    
    :param x_df:      Input dataframe.
    :type x_df:       DataFrame
    :param trait:      Trait short name.
    :type trait:       str

    :return: Returns label
    :rtype: int
    """
    country_encoder = pickle.load(open("C:\saved_models\\" + str(trait) +  "_country_encoder.pickle", "rb"))
    city_encoder = pickle.load(open("C:\saved_models\\" + str(trait) +  "_city_encoder.pickle", "rb"))
    university_name_encoder = pickle.load(open("C:\saved_models\\" + str(trait) +  "_university_name_encoder.pickle", "rb"))
    faculty_encoder = pickle.load(open("C:\saved_models\\" + str(trait) +  "_faculty_encoder.pickle", "rb"))
    home_town_encoder = pickle.load(open("C:\saved_models\\" + str(trait) +  "_home_town_encoder.pickle", "rb"))
    countries = country_encoder.transform(x_df['country'])
    countries = countries.rename(columns={'col_0': 'country0', 'col_1': 'country1', 'col_2': 'country2', 'col_3': 'country3', 'col_4': 'country4',
                        'col_5': 'country5', 'col_6': 'country6', 'col_7': 'country7'})
    city = city_encoder.transform(x_df['city'])
    city = city.rename(columns={'col_0': 'city0', 'col_1': 'city1', 'col_2': 'city2', 'col_3': 'city3', 'col_4': 'city4',
                        'col_5': 'city5', 'col_6': 'city6', 'col_7': 'city7'})

    university_name = university_name_encoder.transform(x_df['university_name'])
    university_name = university_name.rename(columns={'col_0': 'university_name0', 'col_1': 'university_name1', 'col_2': 'university_name2', 'col_3': 'university_name3', 'col_4': 'university_name4',
                        'col_5': 'university_name5', 'col_6': 'university_name6', 'col_7': 'university_name7'})
    faculty = faculty_encoder.transform(x_df['faculty'])
    faculty = faculty.rename(columns={'col_0': 'faculty0', 'col_1': 'faculty1', 'col_2': 'faculty2', 'col_3': 'faculty3', 'col_4': 'faculty4',
                        'col_5': 'faculty5', 'col_6': 'faculty6', 'col_7': 'faculty7'})

    home_town = home_town_encoder.transform(x_df['home_town'])
    home_town = home_town.rename(columns={'col_0': 'home_town0', 'col_1': 'home_town1', 'col_2': 'home_town2', 'col_3': 'home_town3', 'col_4': 'home_town4',
                        'col_5': 'home_town5', 'col_6': 'home_town6', 'col_7': 'home_town7'})
    
    tmp_df = countries.join(city)
    tmp_df = tmp_df.join(university_name)
    tmp_df = tmp_df.join(faculty)
    tmp_df = tmp_df.join(home_town)
    x_df = tmp_df.join(x_df)
    x_df = x_df.drop('country', axis = 1)
    x_df = x_df.drop('city', axis = 1)
    x_df = x_df.drop('home_town', axis = 1)
    x_df = x_df.drop('faculty', axis = 1)
    x_df = x_df.drop('university_name', axis = 1)
    X = x_df.values
    return X

def get_open(df):
    """Classifies oppenness.
    
    :param df:      Input dataframe.
    :type df:       DataFrame

    :return: Returns label
    :rtype: int
    """
    logger.debug("get_open: start")
    df_texts = df[['full_texts']]
    X = df_texts.values
    
    X = open_tfidfconverter.transform(X.ravel()).toarray()
    tf_idf_df = pd.DataFrame(X)
    #print(df.columns)
    x_df = df.join(tf_idf_df)
    x_df = x_df.drop('full_texts', axis = 1)
    
    X = encode_with_hashing(x_df, 'open') 
    y = open_classifier.predict(X)
    return int(y[0])


def get_neur(df):
    """Classifies neurotism.
    
    :param df:      Input dataframe.
    :type df:       DataFrame

    :return: Returns label
    :rtype: int
    """
    logger.debug("get_neur: start")
    df_texts = df[['full_texts']]
    X = df_texts.values
    
    X = neur_tfidfconverter.transform(X.ravel()).toarray()
    tf_idf_df = pd.DataFrame(X)
    x_df = df.join(tf_idf_df)
    x_df = x_df.drop('full_texts', axis = 1)
    x_df = x_df.drop('sex_', axis = 1)
    x_df = x_df.drop('military_', axis = 1)
    x_df = x_df.drop('political_10', axis = 1)
    
    
    X = encode_with_hashing(x_df, 'neur') 
    y = neur_classifier.predict(X)
    return int(y[0])


def get_cons(df):
    """Classifies cons.
    
    :param df:      Input dataframe.
    :type df:       DataFrame

    :return: Returns label
    :rtype: int
    """
    logger.debug("get_cons: start")
    df_texts = df[['full_texts']]
    X = df_texts.values
   
    X = cons_vectorizer.transform(X.ravel())
    X = cons_tfidfconverter.transform(X).toarray()
    tf_idf_df = pd.DataFrame(X)
    x_df = df.join(tf_idf_df)
    x_df = x_df.drop('full_texts', axis = 1)
    x_df = x_df.drop('sex_', axis = 1)
    x_df = x_df.drop('military_', axis = 1)
    x_df = x_df.drop('political_10', axis = 1)
    
    X = encode_with_hashing(x_df, 'cons')
    y = cons_classifier.predict(X)
    return int(y[0])

def get_agree(df):
    """Classifies agreeableness.
    
    :param df:      Input dataframe.
    :type df:       DataFrame

    :return: Returns label
    :rtype: int
    """
    logger.debug("get_agree: start")
    df_texts = df[['full_texts']]
    X = df_texts.values
    
    
    X = agree_tfidfconverter.transform(X.ravel()).toarray()
    tf_idf_df = pd.DataFrame(X)
    x_df = df.join(tf_idf_df)
    x_df = x_df.drop('full_texts', axis = 1)
    x_df = x_df.drop('sex_', axis = 1)
    x_df = x_df.drop('military_', axis = 1)
    x_df = x_df.drop('political_10', axis = 1)
    x_df = x_df.drop('political_9', axis = 1)
    
    X = encode_with_hashing(x_df, 'agree')
    y = agree_classifier.predict(X)
    return int(y[0])


def get_extr(df):
    """Classifies extraversion.
    
    :param df:      Input dataframe.
    :type df:       DataFrame

    :return: Returns label
    :rtype: int
    """
    logger.debug("get_extr: start")
    df_texts = df[['full_texts']]
    X = df_texts.values
    
    
    X = extr_tfidfconverter.transform(X.ravel()).toarray()
    tf_idf_df = pd.DataFrame(X)
    x_df = df.join(tf_idf_df)
    x_df = x_df.drop('full_texts', axis = 1)
    x_df = x_df.drop('sex_', axis = 1)
    x_df = x_df.drop('military_', axis = 1)
    x_df = x_df.drop('political_10', axis = 1)
    
    X = encode_with_hashing(x_df, 'extr')
    y = extr_classifier.predict(X)
    return int(y[0])


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
    line["sex_" + str(profile.sex) + '.0'] = 1
    #line["military_" + str(profile.military)] = 1
    line["political_" + str(profile.political)] = 1
    if str(profile.military) == '0':
        line["military_False"] = 1
    else:
        line["military_True"] = 1
    return line


def count_sentiments(texts):
    """Counts sentiment texts.
    
    :param texts:      Texts to check sentiments.
    :type texts:       list

    :return: Returns count of sentiments
    :rtype: tuple
    """
    

    positive_count = 0
    negative_count = 0
    neutral_count = 0
    speech_count = 0
    uncertain_count = 0
    emojis = 0
    results = interests_model.predict(texts, k=2)
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

        full_text = " ".join(t for t in texts)
        if len(full_text) == 0:
            full_text = 'а'
        full_texts.append(full_text)


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

    text = " ".join(tokens)
    if len(text) == 0:
        text = 'Не указано'

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
    #df.relation[df.relation == ""] = "0"
    df.loc[df["relation"] == "", "relation"] = "0"
    df["groups"] = df["groups"].astype("int64")
    df.loc[df["photos"] == "", "photos"] = "0"
    #df.photos[df.photos == ""] = "0"

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
    df = df.drop("military", axis=1)
    df = df.drop("relation", axis=1)
    df = df.drop("sex", axis=1)
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
    logger.debug("get_interests: start")
    interests_data = []
    to_process = []
    if len(profile.interests) != 0:
        to_process.append(profile.interests)
    if len(profile.books) != 0:
        interests_data.append("Книги")
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
        if len(descr_act) == 2 and len(descr_act[0]) != 0:
            activity = descr_act[1]
            if 'Данный материал заблокирован' not in activity:
                interests_data.append(activity)
        if len(item) != 0:

            to_process.append(item)


    data_norm = [preprocess_text(t).split() for t in to_process]
    lda_model = gensim.models.ldamodel.LdaModel.load("C:\saved_models\LDA_model.model")
    loaded_dict = corpora.Dictionary.load("C:\saved_models\DictionaryInterests.sav")


    for text in data_norm:
        query = [loaded_dict.doc2bow(text)]
        vector = lda_model[query]
        vector = sorted(vector[0], key=lambda x: (x[1]), reverse=True)
        dominant_topic = 0
        for j, (topic_num, topic_contrib) in enumerate(vector):
            if j == 0:  # => dominant topic
                #wp = lda_model.show_topic(topic_num)
                #topic_keywords = ", ".join([word for word, prop in wp])
                #topic_num = int(topic_num)
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
    

        first_name = profiles[0].first_name
        last_name = profiles[0].last_name
        age = 'Не указано'
        if str(profiles[0].bdate) != '0':
            age = datetime.now().year - profiles[0].bdate
        city = profiles[0].city
        country = profiles[0].country
        openn = "No" if get_open(users_info) == 0 else "Yes"
        cons = "No" if get_cons(users_info) == 0 else "Yes"
        neur = "No" if get_neur(users_info) == 0 else "Yes"
        agree = "No" if get_agree(users_info) == 0 else "Yes"
        extr = "No" if get_extr(users_info) == 0 else "Yes"

        interests = set(get_interests(profiles[0]))

    else:
        count_cons = 0
        count_neur = 0
        count_agree = 0
        count_open = 0
        count_extr = 0
        logger.debug("analize: many people")
        users_info = None
        full_interests = []
        count = len(profiles)
        for profile in profiles:
            try:
                line = create_line(profile)
                users_info = pd.DataFrame(line, index=[0])
                users_info = preprocess_df(users_info)
                #print(users_info.columns)
                #print(users_info.values)
                interests = get_interests(profile)
                full_interests.extend(interests)
                cons = get_cons(users_info)
                if cons == 1:
                    count_cons += 1
                agree = get_agree(users_info)
                if agree == 1:
                    count_agree += 1
                neur = get_neur(users_info)
                if neur == 1:
                    count_neur += 1
                openn = get_open(users_info)
                if openn == 1:
                    count_open += 1
                extr = get_extr(users_info)
                if extr == 1:
                    count_extr += 1
            except:
                continue

        full_interests = set(full_interests)
        
        first_name = "-"
        last_name = "-"
        ages_values = [int(el.bdate) for el in profiles if str(el.bdate) != '0']
        ages = 'Не указано'
        if len(ages_values) != 0:
            ages = [datetime.now().year - el for el in ages_values]
            min_age = min(ages)
            max_age = max(ages)
            if min_age == max_age:
                age = str(min_age)
            else:
                age = str(min_age) + " - " + str(max_age)
        countries = set([el.country for el in profiles])
        cities = set([el.city for el in profiles])
        country = ", ".join(countries)
        city = ", ".join(cities)
        openn = str(round(((count_open / count) * 100), 2)) + "%"
        cons =str( round(((count_cons / count) * 100), 2))+ "%"
        neur = str(round(((count_neur / count) * 100), 2))+ "%"
        agree = str(round(((count_agree / count) * 100), 2))+ "%"
        extr = str(round(((count_extr / count) * 100), 2))+ "%"
        interests = full_interests
    build_result(
        search,
        first_name,
        last_name,
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
