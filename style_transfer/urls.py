from django.urls import path
from style_transfer import views

app_name = 'style_transfer'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_content, name='upload_content'),
    path('signup/', views.UserFormView, name='signup'),
    path('login/', views.login_user, name='login'),
    path('saved/', views.saved, name='saved'),
]
