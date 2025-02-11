# board/models.py
from django.contrib.auth.models import User
from django.db import models

class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    # 첫 질문을 대표 제목으로 사용 (나중에 수정 가능)
    title = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class ChatMessage(models.Model):
    # 만약 기존 데이터가 있다면 임시로 null=True, blank=True를 설정할 수 있습니다.
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name="messages",
        default=1 
        # null=True,
        # blank=True,
        )
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    # True: 사용자가 입력, False: 챗봇 응답
    is_user = models.BooleanField(default=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        sender = "나" if self.is_user else "챗봇"
        return f"{sender}: {self.message[:20]}"
