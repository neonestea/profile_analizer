'''Data collector models'''
import uuid
from django.db import models
from django.contrib.auth.models import User
from django.conf import settings


class Search(models.Model):
    """
    Search model
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_by = models.ForeignKey(
        User, null=True, blank=True, on_delete=models.SET_NULL
    )
    name = models.CharField("Name", max_length=50, default="NewSearch")
    link = models.TextField("Link")
    date = models.DateTimeField("Date of search", auto_now_add=True)
    ready = models.IntegerField("Ready", default=0)

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return f"/searches/{self.id}"

    def link_lines(self):
        return [el.strip() for el in self.link.split("\n")]

    def save_model(self, request, obj, form, change):
        obj.created_by = request.user
        super().save_model(request, obj, form, change)

    class Meta:
        verbose_name = "Search"
        verbose_name_plural = "Searches"


class ProfileInfo(models.Model):
    '''ProfileInfo model'''
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    connected_search = models.ForeignKey(
        Search, null=True, blank=True, on_delete=models.SET_NULL
    )
    link = models.CharField("Link", max_length=50)
    first_name = models.CharField("First Name", max_length=50)
    last_name = models.CharField("Last Name", max_length=50)
    bdate = models.IntegerField("Birth Date")
    interests = models.TextField("Interests")
    books = models.TextField("Books")
    tv = models.TextField("TV")
    games = models.TextField("Games")
    movies = models.TextField("Movies")
    activities = models.TextField("Activities")
    music = models.TextField("Music")
    status = models.TextField("Status")
    military = models.IntegerField("Military")
    # university = models.TextField('University')
    university_name = models.TextField("University Name")
    faculty = models.TextField("Faculty")
    # graduation = models.TextField('Graduation')
    home_town = models.CharField("Home Town", max_length=50)
    relation = models.TextField("Relation")
    # schools = models.TextField('Schools')
    sex = models.IntegerField("Sex")
    about = models.TextField("About")
    # career = models.TextField('Career')
    country = models.CharField("Country", max_length=50)
    city = models.CharField("City", max_length=50)
    friends_count = models.IntegerField("Friends count")
    followers_count = models.IntegerField("Followers count")
    groups = models.IntegerField("Groups count")
    group_infos = models.TextField("Groups")
    posts = models.TextField("Posts")
    comments = models.TextField("Comments")
    comments_of_other_users = models.TextField("Comments of other users")
    alcohol = models.CharField("Alcohol", max_length=50)
    life_main = models.CharField("Life Main", max_length=50)
    people_main = models.CharField("People Main", max_length=50)
    political = models.CharField("Political", max_length=50)
    religion = models.CharField("Religion", max_length=50)
    smoking = models.CharField("Smoking", max_length=50)
    photos_count = models.IntegerField("Photos count")
    posts_count = models.IntegerField("Posts count")

    class Meta:
        verbose_name = "ProfileInfo"
        verbose_name_plural = "ProfileInfos"


class Result(models.Model):
    '''Result model'''
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    connected_search = models.ForeignKey(
        Search, null=True, blank=True, on_delete=models.SET_NULL
    )
    first_name = models.CharField("First Name", max_length=50)
    last_name = models.CharField("Last Name", max_length=50)
    age = models.CharField("Age(s)", max_length=20)
    city = models.TextField("City(s)")
    country = models.TextField("Country(s)")
    open = models.CharField("openness", max_length=20)
    cons = models.CharField("cons", max_length=20)
    neur = models.CharField("neur", max_length=20)
    agree = models.CharField("agree", max_length=20)
    extr = models.CharField("extr", max_length=20)
    interests = models.TextField("Interests")
