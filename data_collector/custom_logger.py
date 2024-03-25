'''Logger'''
import logging


def configure_logger():
    """Creates new logger.

    :return: Returns new logger.
    :rtype: logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("cache.log")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def set_stdout_handler(custom_logger):
    """Configures stdout handler for logger.
    
    :param custom_logger:      Input Logger.
    :type custom_logger:       logger

    :return: Returns logger.
    :rtype: logger
    """
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("STREAM %(asctime)s\t%(levelname)s\t%(message)s")
    stdout_handler.setFormatter(formatter)
    custom_logger.addHandler(stdout_handler)
    return custom_logger
