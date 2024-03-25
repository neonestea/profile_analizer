'''Data collector forms'''
from .models import Search
from django.forms import ModelForm, TextInput, Textarea


def validate_name(obj):
    '''Validates name'''
    print(obj.created_by)


class SearchForm(ModelForm):
    '''Form to create search'''
    def __init__(self, **kwargs):
        self.created_by = kwargs.pop("created_by", None)
        super(SearchForm, self).__init__(**kwargs)

    def save(self, commit=True):
        obj = super(SearchForm, self).save(commit=False)
        obj.created_by = self.created_by
        if commit:
            obj.save()
        return obj

    class Meta:
        model = Search
        fields = ["name", "link"]
        # readonly_fields = ['created_by', 'date']

        widgets = {
            "name": TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "Type a name of a search",
                }
            ),
            "link": Textarea(
                attrs={
                    "class": "form-control",
                    "placeholder": "Type link(s) to sarch, one per line",
                }
            ),
        }


class SearchFormUpdate(ModelForm):
    '''Form to update search'''
    class Meta:
        model = Search
        fields = ["name"]
        readonly_fields = ["user_id", "link", "date"]

        widgets = {
            "name": TextInput(
                attrs={
                    "class": "form-control",
                    "placeholder": "Type a name of a search",
                }
            )
        }
