from django.contrib.auth.models import User
from django.db import models

class Board(models.Model):
    title = models.CharField(max_length=20, null=True)
    content = models.TextField()
    #게시글 작성자는 User 테이블을 참조
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING, null=True)
    image = models.ImageField(upload_to='images/', blank=True, null=True)
    c_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'board'

class Comment(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    comment = models.TextField()
    w_date = models.DateTimeField(auto_now_add=True)
    post = models.ForeignKey(Board, null=True, blank=True, on_delete=models.CASCADE)

    def __str__(self):
        return self.comment
    
    class Meta:
        db_table = 'comment'