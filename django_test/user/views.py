# user/views.py
from django.shortcuts import render, redirect
from django.contrib.auth import login as auth_login, authenticate, logout as auth_logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib import messages

def index(request):
    # 루트 페이지 요청 시 로그인 상태에 따라 분기
    if request.user.is_authenticated:
        return redirect('board:chat')
    else:
        return redirect('user:login')

def login(request):
    # 이미 로그인된 상태면 바로 대화창으로 이동
    if request.user.is_authenticated:
        return redirect('board:chat')

    if request.method == 'POST':
        form = AuthenticationForm(request=request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            return redirect('board:chat')
        else:
            messages.error(request, "로그인 정보가 올바르지 않습니다.")
    else:
        form = AuthenticationForm()
    return render(request, 'user/login.html', {'form': form})

def signup(request):
    if request.user.is_authenticated:
        return redirect('board:chat')
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "회원가입이 완료되었습니다. 로그인 해주세요.")
            return redirect('user:login')
    else:
        form = UserCreationForm()
    return render(request, 'user/signup.html', {'form': form})

def logout(request):
    auth_logout(request)
    return redirect('user:login')
