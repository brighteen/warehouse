# board/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import ChatForm
from .models import ChatMessage

# LLM 및 백터 스토어 관련 초기화 (필요한 경우 test.py에 있는 코드를 재사용)
from database_process import create_vector_store
from langchain_upstage import UpstageEmbeddings
from llm_process import generate_response

# 전역 vector_store (실제 서비스에서는 별도 캐싱/초기화 고려)
vector_store = create_vector_store(
    collection_name="example_collection",
    db_path="./chroma_langchain_db",
    passage_embeddings=UpstageEmbeddings(model="solar-embedding-1-large-passage")
)

@login_required(login_url='/login/')
def chat_view(request):
    if request.method == 'POST':
        form = ChatForm(request.POST)
        if form.is_valid():
            question = form.cleaned_data['question']
            # 사용자 질문 저장
            ChatMessage.objects.create(user=request.user, message=question, is_user=True)
            # LLM 처리 (질문에 대한 답변 생성)
            answer, evaluation = generate_response(vector_store, question)
            answer_text = str(answer)
            # 챗봇 응답 저장
            ChatMessage.objects.create(user=request.user, message=answer_text, is_user=False)
            return redirect('board:chat')
    else:
        form = ChatForm()
    # 현재 사용자에 대한 대화 기록 (시간순 정렬)
    messages = ChatMessage.objects.filter(user=request.user).order_by('timestamp')
    return render(request, 'board/chat.html', {'form': form, 'messages': messages})
