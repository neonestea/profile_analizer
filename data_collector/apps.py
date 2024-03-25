'''Data collector apps'''
from django.apps import AppConfig


class DataCollectorConfig(AppConfig):
    '''Data collector config'''
    default_auto_field = "django.db.models.BigAutoField"
    name = "data_collector"
