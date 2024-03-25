'''Users views'''
from django.shortcuts import render, redirect
from django.contrib.auth import logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import NewUserForm
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, authenticate

def register_request(request):
	'''Register user view'''
	error = ''
	if request.method == "POST":
		form = NewUserForm(request.POST)
		if form.is_valid():
			user = form.save()
			login(request, user)
			error = "Registration successful."
			return redirect("home")
		error = list(form.errors.values())[0]
	
	form = NewUserForm()
	data = {
        'form' : form,
        'error': error
    }
	return render (request=request, template_name="users/register.html", context=data)


@login_required
def profile(request):
    return render(request, 'users/profile.html')

@login_required
def logout_view(request):
    logout(request)
    return redirect('/')

def login_request(request):
	'''Login view'''
	error = ''
	if request.method == "POST":
		form = AuthenticationForm(request, data=request.POST)
		if form.is_valid():
			username = form.cleaned_data.get('username')
			password = form.cleaned_data.get('password')
			user = authenticate(username=username, password=password)
			if user is not None:
				login(request, user)
				return redirect("search_home")
		error = list(form.errors.values())[0]
		
	form = AuthenticationForm()
	data = {
        'login_form' : form,
        'error': error
    }
	return render(request=request, template_name="users/login.html", context=data)