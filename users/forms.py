'''Users forms'''
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User



class NewUserForm(UserCreationForm):
	'''Form to create user'''
	#email = forms.EmailField(required=True)

	class Meta:
		model = User
		#fields = ("username", "email", "password1", "password2")
		fields = ("username", "password1", "password2")

	def save(self, commit=True):
		user = super(NewUserForm, self).save(commit=False)
		#user.email = self.cleaned_data['email']
		if commit:
			user.save()
		return user