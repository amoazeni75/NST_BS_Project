from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from .forms import *
from django.contrib import messages
from .models import Styles
from style_transfer_backend import Tools, StyleTransfer2
import numpy as np
import os
from style.settings import MEDIA_ROOT, BASE_DIR


# Create your views here.

def index(request):
    return render(request, 'style_transfer/home.html', {})


def upload_content(request):
    print("here")
    # Handle file upload
    if request.method == 'POST':
        form = Upload_content_style(request.POST or None, request.FILES or None)
        print(request.FILES)
        if form.is_valid():

            # extract style path
            style_id = form.cleaned_data['Style_num']
            style_object = Styles.objects.get(Style_num=style_id)
            style_path = style_object.Stylefile.path

            # extract content image id
            instance = form.save(commit=False)
            instance.save()
            content_object = Uploadpics.objects.get(id=instance.id)
            if content_object.Url_field is None:
                content_path = content_object.Contentfile.path
            else:
                content_path = "nothing"

            content_image = Tools.load_img(content_path, True)
            style_image = Tools.load_img(style_path, True)
            output_image = StyleTransfer2.style_transfer(content_image, style_image)

            random_name = "output{x}.jpg".format(x=np.random.randn())
            img_path2 = os.path.join(BASE_DIR, 'media', 'saved_images', random_name)

            output_image.save(img_path2)
            img_path2 = '/media/' + 'saved_images/' + random_name

            request.session["path_img"] = img_path2

        return redirect("style_transfer:saved")

    else:
        form = Upload_content_style()  # A empty, unbound form
        context = {
            "form": form,
            "list": Styles.objects.all(),
        }
        return render(request, 'style_transfer/upload.html', context)


def login_user(request):
    if request.user.is_authenticated:
        logout(request)

    if request.method == 'POST':
        form = login_form(request.POST)

        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(username=username, password=password)
            if user is None:
                form1 = login_form()
                context = {
                    "form": form1,
                }
                return render(request, 'style_transfer/login.html', context)
            # print("Brook was here")
            login(request, user)
            return redirect('style_transfer:upload_content')

    form = login_form()
    context = {
        "form": form,
    }
    return render(request, 'style_transfer/login.html', context)


def UserFormView(request):
    if request.method == "POST":
        form = UserForm(request.POST)
        print(request.POST["username"])
        if form.is_valid():
            form.save()
            return redirect("upload_content")
        print("invalid form")
        messages.error(request, "Error")
        form = UserForm()
        context = {
            "form": form,
        }
        return render(request, "style_transfer/signup.html", context)
    print("invalid post")
    form = UserForm()
    context = {
        "form": form,
    }
    return render(request, "style_transfer/signup.html", context)


def saved(request):
    context = {
        "path_img": request.session["path_img"]
    }
    return render(request, 'style_transfer/saved_img.html', context)
