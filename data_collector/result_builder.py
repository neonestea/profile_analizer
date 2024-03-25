'''Module to build result'''
from .models import ProfileInfo, Result, Search
from datetime import datetime
from .custom_logger import configure_logger, set_stdout_handler

logger = configure_logger()
'''logger'''
logger = set_stdout_handler(logger)
'''logger'''


def build_result(
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
):
    """Builds result of the search.

    :param search:      Search to show.
    :type search:       Search
    :param first_name:      First name of the person.
    :type first_name:       str
    :param last_name:      Last name of the person.
    :type last_name:       str
    :param age:     Age or ages.
    :type age:      str
    :param country:     Country or countries.
    :type country:      str
    :param city: City or cities.
    :type city:  str
    :param open: Openess or openess percentage.
    :type open: str
    :param cons: Conscientiousness or conscientiousness percentage.
    :type cons: str
    :param neur: Neurotism or neurotism percentage.
    :type neur: str
    :param agree: Agreeableness or agreeableness percentage.
    :type agree: str
    :param extr: Extraversion or extraversion percentage.
    :type extr: str
    :param interests: Interests.
    :type interests: str
    """
    logger.debug("build_result: start")
    new_result = Result()
    new_result.connected_search = search
    new_result.first_name = first_name
    new_result.last_name = last_name
    new_result.age = age
    new_result.city = city
    new_result.country = country
    new_result.open = open
    new_result.cons = cons
    new_result.neur = neur
    new_result.agree = agree
    new_result.extr = extr
    new_result.interests = interests
    new_result.save()
    # search = Search.objects.get(id=search)
    search.ready = 1
    search.save()
