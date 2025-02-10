# board/urls.py
from django.urls import path
from .views import chat_view

app_name = 'board'

urlpatterns = [
    path('chat/', chat_view, name='chat'),
]
