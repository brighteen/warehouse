# board/models.py
from django.contrib.auth.models import User
from django.db import models

class ChatMessage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    is_user = models.BooleanField(default=True)  # True: 사용자 입력, False: 챗봇 응답
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'chat_message'
        ordering = ['timestamp']

    def __str__(self):
        sender = "나" if self.is_user else "챗봇"
        return f"{sender}: {self.message[:20]}"
