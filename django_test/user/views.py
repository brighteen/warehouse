from django.contrib import auth
from django.contrib.auth.models import User
from django.shortcuts import render, redirect

def signup(request):
    if request.method == 'POST':

        # password와 password2 입력된 값이 같다면
        if request.POST['password'] == request.POST['password2']:
            # user 객체 생성
            user = User.objects.create_user(
                username=request.POST['username'],
                password=request.POST['password'],
                email=request.POST['email'])
            # 로그인
            auth.login(request, user)
            return redirect('/')
        
    return render(request, 'user/signup.html')

def login(request):
    if request.method == 'POST':
        # login.html에서 넘어온 username과 password를 변수에 저장
        login_username = request.POST['username']
        login_password = request.POST['password']
        # 해당 username과 password와 일치하는 user 객체 찾기
        user = auth.authenticate(request, username=login_username, password=login_password)

        # 해당 user 객체가 존재한다면
        if user is not None:
            # 로그인
            auth.login(request, user)
            request.session['user'] = user.id
            return redirect('/')
        
        # 존재하지 않는다면 else:
        # else:
        return render(request, 'user/login.html')
            # return render(request, 'user/login.html', {'error': 'Invalid username or password'})
        
    else:
        return render(request, 'user/login.html')
    
def logout(request):
    auth.logout(request)
    if request.session.get('user'):
        del(request.session['user'])

    return redirect('/')

def index(request):
    return render(request, 'index.html')