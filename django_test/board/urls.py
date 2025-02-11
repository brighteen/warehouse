from django.urls import path
from .views import chat_view, stream_answer

app_name = 'board'

urlpatterns = [
    path('chat/', chat_view, name='chat'),
    path('chat/<int:conversation_id>/', chat_view, name='chat_detail'),
    path('stream_answer/', stream_answer, name='stream_answer'),
]