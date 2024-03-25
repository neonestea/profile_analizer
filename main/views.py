'''Main views'''
from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    '''
    Function to open main page
    '''
    return render(request, 'main/index.html')

def about(request):
    '''
    Function to open page about project
    '''
    #return HttpResponse("About project")
    return render(request, 'main/about.html')

def contacts(request):
    '''
    Function to open page with contacts
    '''
    #return HttpResponse("About project")
    return render(request, 'main/contacts.html')

