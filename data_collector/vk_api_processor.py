'''Module to get information by VK API'''
import vk_api
from datetime import datetime
from .models import ProfileInfo
from .analizer import analize
import uuid
from .custom_logger import configure_logger, set_stdout_handler

session = vk_api.VkApi(
    token="vk1.a.Gi9VLR0O3w9yu6Mf30_rFTibl20zp6XMcjmn_sodhGHSH1tWW8_nYW9NZ9DavAhtQFnxA1sVeZ5b5754Rtzio8J8sDoCB6XFN4MX0QgbiN548Wc8xKFnqvv2Avqjb7O5xps2J8QSEpIwfR93NNq2aBKymo0KCdqiTSMXfKB36AT-x7xcnBjxTmvXxnaW4WVLniAlDvn2nVjDOCGcrjFS5Q"
)
'''vk api session'''
vk = session.get_api()
'''api'''
date_format = "%d.%m.%Y"
'''date formatter'''
logger = configure_logger()
logger = set_stdout_handler(logger)


def start_collecting_info(search, links):
    """Starts collecting info.
    
    :param search:      Input Search.
    :type search:       Search
    :param links:      Input links as text.
    :type links:       str

    """

    logger.debug("START start_collecting_info")
    links = [el.strip() for el in links.split("\n")]
    users = []
    for link in links:
        users.extend(parse_profile(link, search))
    logger.debug("users count = %s", len(users))
    for user in users:
        logger.debug("user = %s, link = %s", str(user[0]["id"]), user[1])
        parse_profile_info(user[0], user[1], search)
    analize(search)


def parse_profile(url, search):
    """Initiates profile processing.
    
    :param url:      Input url.
    :type url:       str
    :param search:      Input Search.
    :type search:       Search

    """
    profile = check_user(url)
    logger.debug("START parse_profile")
    result = []
    if profile is not None:
        result.append([profile, url])
    else:
        profile = check_group(url)
        members = vk.groups.getMembers(group_id=profile["id"])
        for member in members["items"]:
            usr_link = "https://vk.com/id" + str(member)
            logger.debug("Result link = %s", usr_link)
            member_profile = check_user(usr_link)
            result.append([member_profile, usr_link])
    return result


def parse_profile_info(profile, url, search):
    """Creates new ProfileInfo.
    
    :param profile:      Input Profile from VK API.
    :type profile:       dict
    :param url:      Input url.
    :type url:       str
    :param search:      Input search.
    :type search:       Search

    """
    logger.debug("START parse_profile_info")
    profile_id = str(profile["id"])
    new_profile_info = ProfileInfo()
    new_profile_info.connected_search = search
    new_profile_info.link = url
    new_profile_info.country = (
        profile["country"]["title"] if "country" in profile else "Не указано"
    )
    new_profile_info.city = (
        profile["city"]["title"] if "city" in profile else "Не указано"
    )
    new_profile_info.first_name = (
        profile["first_name"] if "first_name" in profile else "Не указано"
    )
    new_profile_info.last_name = (
        profile["last_name"] if "last_name" in profile else "Не указано"
    )
    bdate = "Не указано"
    if "bdate" in profile:
        # print(profile['bdate'])
        try:
            bdate = datetime.strptime(profile["bdate"], date_format)
        # print(bdate)
        except:
            bdate = "Не указано"

    new_profile_info.bdate = bdate.year if bdate != "Не указано" else 0
    new_profile_info.interests = (
        profile["interests"] if "interests" in profile else "Не указано"
    )
    new_profile_info.books = profile["books"] if "books" in profile else "Не указано"
    new_profile_info.tv = profile["tv"] if "tv" in profile else "Не указано"
    new_profile_info.games = profile["games"] if "games" in profile else "Не указано"
    new_profile_info.movies = profile["movies"] if "movies" in profile else "Не указано"
    new_profile_info.activities = (
        profile["activities"] if "activities" in profile else "Не указано"
    )
    new_profile_info.music = profile["music"] if "music" in profile else "Не указано"
    new_profile_info.status = profile["status"] if "status" in profile else "Не указано"
    new_profile_info.military = (
        len(profile["military"]) != 0 if "military" in profile else 0
    )
    new_profile_info.university_name = (
        profile["university_name"] if "university_name" in profile else "Не указано"
    )
    new_profile_info.faculty = "Не указано"
    new_profile_info.posts_count = get_user_posts(profile_id)
    new_profile_info.photos_count = get_user_photos(profile_id)
    if "university" in profile and "faculty" in profile:
        university_id = int(profile["university"])
        if university_id != 0:
            faculties = vk.database.getFaculties(university_id=university_id)
            for el in faculties["items"]:
                if el["id"] == int(profile["faculty"]):
                    new_profile_info.faculty = el["title"]

    # info['graduation'] = profile['graduation'] if 'graduation' in profile else None
    new_profile_info.home_town = (
        profile["home_town"] if "home_town" in profile else "Не указано"
    )
    new_profile_info.relation = (
        int(profile["relation"]) if "relation" in profile else "0"
    )
    # info['schools'] = profile['schools'] if 'schools' in profile else None
    new_profile_info.sex = int(profile["sex"]) if "sex" in profile else "2.0"
    new_profile_info.about = profile["about"] if "about" in profile else "Не указано"
    # info['career'] = profile['career'] if 'career' in profile else None
    new_profile_info.country = (
        profile["country"]["title"] if "country" in profile else "Не указано"
    )
    new_profile_info.city = (
        profile["city"]["title"] if "city" in profile else "Не указано"
    )
    # new_profile_info.timezone = profile['timezone'] if 'timezone' in profile else "Не указано"
    new_profile_info.friends_count = get_user_friends(profile_id)
    new_profile_info.followers_count = get_user_followers(profile_id)
    new_profile_info.group_infos, new_profile_info.groups = get_user_groups(profile_id)
    (
        new_profile_info.posts,
        new_profile_info.comments,
        new_profile_info.comments_of_other_users,
    ) = get_all_user_comments(profile_id)
    # new_profile_info.occupation = profile['occupation'] if 'occupation' in profile else "Не указано"
    if "personal" in profile:
        pers = profile["personal"]
        new_profile_info.alcohol = int(pers["alcohol"]) if "alcohol" in pers else "0"
        new_profile_info.life_main = (
            int(pers["life_main"]) if "life_main" in pers else "0"
        )
        new_profile_info.people_main = (
            int(pers["people_main"]) if "people_main" in pers else "0"
        )
        new_profile_info.political = (
            int(pers["political"]) if "political" in pers else "0"
        )
        new_profile_info.religion = pers["religion"] if "religion" in pers else "0"
        new_profile_info.smoking = int(pers["smoking"]) if "smoking" in pers else "0"

    else:
        new_profile_info.alcohol = "0"
        new_profile_info.life_main = "0"
        new_profile_info.people_main = "0"
        new_profile_info.political = "0"
        new_profile_info.religion = "0"
        new_profile_info.smoking = "0"
    new_profile_info.save()


def check_user(url):
    """Gets user from VK API by url.

    :param url:      Input url.
    :type url:       str

    :return: Returns user from vk api.
    :rtype: dict
    """
    token = url.split("/")[-1]
    profile = vk.users.get(
        user_ids=(token),
        fields="activities, about, books, bdate, city, country, education, followers_count, home_town, sex, status, games, interests, military, movies, music, personal, relation, relatives, timezone, tv, universities",
    )
    if len(profile) == 0:
        return None
    return profile[0]


def check_group(url):
    """Gets group from VK API by url.

    :param url:      Input url.
    :type url:       str

    :return: Returns group from vk api.
    :rtype: dict
    """
    token = url.split("/")[-1]
    profile = vk.groups.getById(group_id=token)
    if len(profile) == 0:
        return None
    return profile[0]


def get_user_posts(user_id):
    """Gets number of posts of user.

    :param user_id:      Input user id.
    :type user_id:       str

    :return: Returns number of posts.
    :rtype: int
    """
    posts_count = 0
    try:
        wall = vk.wall.get(owner_id=user_id)
        posts_count = wall["count"]
        return posts_count
    except vk_api.exceptions.ApiError:
        return posts_count


def get_user_photos(user_id):
    """Gets number of user photos from VK API user id.

    :param user_id:      Input user id.
    :type user_id:       str

    :return: Returns number of user photos.
    :rtype: dict
    """
    photos_count = 0
    try:
        photos = vk.photos.getAll(owner_id=user_id)
        photos_count = photos["count"]
        return photos_count
    except vk_api.exceptions.ApiError:
        return photos_count


def get_user_groups(user_id):
    """Gets user groups and user_groups count from VK API user id.

    :param user_id:      Input user id.
    :type user_id:       str

    :return: Returns user group infos and count.
    :rtype: tuple
    """
    logger.debug("START get_user_groups")
    groups_count = 0
    result = ""
    try:
        groups = vk.users.getSubscriptions(user_id=user_id)
        groups = groups["groups"]["items"]
        groups_count = len(groups)
        groups = groups[0:50]
        for item in groups:
            try:
                group = vk.groups.getById(
                    group_id=item, fields="description, activity, status"
                )
                # logger.debug("Group = %s", group)
                st = group[0]["status"] if "status" in group[0] else ''
                descr = group[0]["description"] if "description" in group[0] else ''
                act = group[0]["activity"] if "activity" in group[0] else ''
                result += (
                    " "
                    + descr
                    + " "
                    + st
                    + " Activity: "
                    + act
                    + "==="
                )

            except vk_api.exceptions.ApiError:
                continue
        return result, groups_count
    except vk_api.exceptions.ApiError:
        return result, groups_count



def get_user_followers(user_id):
    """Gets user followers count from VK API user id.

    :param user_id:      Input user id.
    :type user_id:       str

    :return: Returns user followers count.
    :rtype: int
    """
    try:
        followers = vk.users.getFollowers(user_id=user_id)
        return followers["count"]
    except vk_api.exceptions.ApiError:
        return 0


def get_user_friends(user_id):
    """Gets user friends count from VK API user id.

    :param user_id:      Input user id.
    :type user_id:       str

    :return: Returns user friends count.
    :rtype: int
    """
    try:
        friends = vk.friends.get(user_id=user_id)
        return friends["count"]
    except vk_api.exceptions.ApiError:
        return 0


def get_photo_comments(user_id):
    """Gets photo comments of user.

    :param user_id:      Input user id.
    :type user_id:       str

    :return: Returns user photo comments count and comments from other people count.
    :rtype: tuple
    """
    logger.debug("START get_photo_comments")
    user_comments = []
    others_comments = []
    try:
        photos = vk.photos.getAll(owner_id=user_id, count=10)
    except vk_api.exceptions.ApiError:
        return user_comments, others_comments
    #logger.debug("Photos size = %s", photos["count"])
    for photo in photos["items"][:10]:
        comms = []
        try:
            comms = vk.photos.getComments(owner_id=user_id, photo_id=photo["id"])
            comms = comms["items"]
        except vk_api.exceptions.ApiError:
            comms = comms
        #logger.debug("Comms size = %s", len(comms))
        for com in comms:
            if len(com["text"]) != 0:
                if str(com["from_id"]) == str(user_id):
                    user_comments.append(com["text"])
                else:
                    others_comments.append(com["text"])

    return user_comments, others_comments


def get_all_user_comments(user_id):
    """Gets all comments per user.

    :param user_id:      Input user id.
    :type user_id:       str

    :return: Returns user posts, user comments, comments from other users.
    :rtype: tuple
    """
    logger.debug("START get_all_user_comments")
    all_comments, others_comments = get_photo_comments(user_id)

    posts, usr_comms, oth_comms = get_user_wall(user_id)
    all_comments.extend(usr_comms)
    others_comments.extend(oth_comms)
    # for group in get_user_groups(user_id):
    #    usr_comms = get_user_in_group_comments(group, user_id)
    #   all_comments.extend(usr_comms)

    return posts, all_comments, others_comments


def get_user_wall(user_id):
    """Gets user wall.

    :param user_id:      Input user id.
    :type user_id:       str

    :return: Returns user posts, user comments, comments from other users.
    :rtype: tuple
    """
    logger.debug("START get_user_wall")
    user_comments = []
    posts = []
    others_comments = []
    wall = None
    try:
        wall = vk.wall.get(owner_id=user_id, count=10)
    except vk_api.exceptions.ApiError:
        return posts, user_comments, others_comments

    #logger.debug("wall size = %s", wall["count"])
    for post in wall["items"][:10]:
        if len(post["text"]) != 0:
            posts.append(post["text"])
        if "copy_history" in post:
            hist = post["copy_history"]

            if len(hist[0]["text"]) != 0:
                # logger.debug("post = %s", hist['text'])
                posts.append(hist[0]["text"])
        comms = []
        try:
            comms = vk.wall.getComments(owner_id=user_id, post_id=post["id"])
            comms = comms["items"]
            #logger.debug("comms size = %s", len(comms))
        except vk_api.exceptions.ApiError:
            # print("ERROR")
            comms = comms
        for com in comms:
            if len(com["text"]) != 0:
                if str(com["from_id"]) == str(user_id):
                    user_comments.append(com["text"])
                else:
                    others_comments.append(com["text"])

    return posts, user_comments, others_comments