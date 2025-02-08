from django.urls import path
from .views import *

app_name = 'board'

urlpatterns = [
    path('', board_list, name = 'board_list'),
    path('new/', board_create, name = 'board_create'),
    path('detail/<int:id>/', board_detail, name = 'board_detail'),

    path('update/<int:id>', board_update , name='board_update'),
    path('delete/<int:id>', board_delete , name='board_delete'),

    path('comment_create/<int:board_id>', comment_create, name='comment_create'),
    path('comment_delete/<int:comment_id>', comment_delete , name='comment_delete'),
    
]