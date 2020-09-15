from django.contrib.auth.models import User
from django import forms
from django.forms import ModelForm
from .models import Uploadpics
from django.contrib.auth.forms import UserCreationForm


class UserForm(UserCreationForm):
    # password= forms.CharField(widget= forms.PasswordInput)
    email = forms.EmailField(max_length=200, help_text='Required')

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']


class login_form(forms.Form):
    username = forms.CharField(label='Your name', max_length=100)
    password = forms.CharField(widget=forms.PasswordInput)


class Upload_content_style(ModelForm):
    # url_field = forms.CharField(max_length=100)
    class Meta:
        model = Uploadpics
        fields = ['Url_field', 'Style_num', 'Contentfile']

        widgets = {
            'Url_field': forms.HiddenInput(attrs={'id': 'url_field', 'name': 'url_field'}),
            'Style_num': forms.HiddenInput(attrs={'id': 'style_num', 'name': 'style_num'}),
            'Contentfile': forms.FileInput(
                attrs={'id': 'file_in', 'style:"height=200px;"' 'name': 'file_in', 'placeholder': 'asddvef',
                       'class': 'smoothscroll btn btn--stroke'}),

        }
