from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import BoardForm, CommentForm
from .models import Board, User, Comment

def board_list(request):
    boardForm = BoardForm
    board = Board.objects.all()
    context = {
    'boardForm': boardForm,
    # 'board': board,
    'board_list': board,
    
    }

    return render(request, 'board/board_list.html', context)

@login_required(login_url='/login/')  # 로그인되지 않은 경우 로그인 페이지로 이동
def board_create(request):
    if request.method == 'POST':
        title = request.POST['title']
        content = request.POST['content']
        user = request.user
        image = request.FILES.get('image')

        board = Board(
            title=title,
            content=content,
            user=user,
            image=image,
        )
        board.save()

        # auth.board_list(request, board)
        return redirect('/board')
    
    return render(request, 'board/board_create.html')

# def board_detail(request, id):
#     board = Board.objects.get(pk=id)

#     comments = CommentForm()
#     comment_view = Comment.objects.filter(post=Board.objects.get(pk=id))

#     return render(request, 'board/board_detail.html',{'board':board, 'comments': comments,
#                                                       'comment_view':comment_view})

def board_detail(request, id):
    board = get_object_or_404(Board, pk=id)  # 선택한 게시글 가져오기
    board_list = Board.objects.all()  # 📌 모든 게시글 목록 가져오기
    comments = Comment.objects.filter(board=board)  # 해당 게시글의 댓글 가져오기

    return render(request, 'board/board_detail.html', {
        'board': board,  
        'board_list': board_list,  # 📌 템플릿에서 사용될 전체 게시글 목록 추가
        'comments': comments
    })

# def board_update(request, id):
#     board = Board.objects.get(pk=id)

#     if request.method == "POST":
#         board.title = request.POST['title']
#         board.content = request.POST['content']
#         board.image = request.FILES['image']
#         board.user = request.user

#         board.save()
#         return redirect('board:board_detail',id)
    
#     return render(request, 'board/board_update.html')

def board_update(request, id):
    board = Board.objects.get(pk=id)

    if request.method == "POST":
        board.title = request.POST['title']
        board.content = request.POST['content']

        # 파일 업로드가 있을 때만 업데이트 (KeyError 방지)
        image = request.FILES.get('image')  
        if image:  # 파일이 있을 때만 업데이트
            board.image = image

        board.user = request.user
        board.save()
        return redirect('board:board_detail', id)
    
    return render(request, 'board/board_update.html', {'board': board})

def board_delete(request, id):
    board = Board.objects.get(pk=id)
    board.delete()

    return redirect('/board')

from django.shortcuts import get_object_or_404

def comment_create(request, board_id):
    comment_create = CommentForm(request.POST)
    user_id = request.session['user']
    user = User.objects.get(pk=user_id)

    if comment_create.is_valid():
        comments = comment_create.save(commit=False)
        comments.post = get_object_or_404(Board, pk=board_id)
        comments.author = user
        comments.save()

    return redirect('board:board_detail',board_id)

def comment_delete(requset, comment_id):
    comment = get_object_or_404(Comment, pk=comment_id)
    comment.delete()
    return redirect('board:board_detail', id = comment.post.id )