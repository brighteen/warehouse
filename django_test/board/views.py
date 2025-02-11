# board/views.py
import time
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse, JsonResponse
from .forms import ChatForm
from .models import ChatMessage, Conversation
from database_process import create_vector_store
from langchain_upstage import UpstageEmbeddings
from llm_process import generate_response

# 전역 vector_store 초기화 (실제 서비스에서는 캐싱/초기화 방식을 고려)
vector_store = create_vector_store(
    collection_name="example_collection",
    db_path="./chroma_langchain_db",
    passage_embeddings=UpstageEmbeddings(model="solar-embedding-1-large-passage")
)

@login_required(login_url='/login/')
def chat_view(request, conversation_id=None):
    """
    채팅 페이지 뷰: 메인 채팅창과 사이드바에 대화 기록을 표시합니다.
    """
    if conversation_id:
        conversation = get_object_or_404(Conversation, pk=conversation_id, user=request.user)
    else:
        conversation = None

    form = ChatForm()
    messages = conversation.messages.order_by('timestamp') if conversation else []
    conversation_list = Conversation.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'board/chat.html', {
        'form': form,
        'messages': messages,
        'conversation_list': conversation_list,
        'active_conversation': conversation
    })

@login_required(login_url='/login/')
def stream_answer(request):
    """
    스트리밍 방식으로 답변 토큰을 반환하는 뷰.
    POST 파라미터로 'question'과 (있다면) 'conversation_id'를 받습니다.
    새 Conversation이 없으면 생성하고, 질문 메시지를 저장한 후 답변 토큰들을 StreamingHttpResponse로 전송합니다.
    """
    if request.method == 'POST':
        conversation_id = request.POST.get('conversation_id')
        question = request.POST.get('question')
        if conversation_id:
            conversation = get_object_or_404(Conversation, pk=conversation_id, user=request.user)
            # 이미 질문이 저장되어 있지 않다면 저장
            ChatMessage.objects.create(
                conversation=conversation,
                user=request.user,
                message=question,
                is_user=True
            )
        else:
            conversation = Conversation.objects.create(user=request.user, title=question)
            ChatMessage.objects.create(
                conversation=conversation,
                user=request.user,
                message=question,
                is_user=True
            )

        # 스트리밍 응답 생성 함수
        def token_generator():
            # generate_response()를 호출하여 실제 LLM의 응답을 받아옵니다.
            answer_result, _ = generate_response(vector_store, question)
            try:
                # JSON의 "conclusion" 키 아래 "conclusion"가 리스트로 있다고 가정합니다.
                tokens = answer_result.get("conclusion", {}).get("conclusion", [])
                if not tokens:
                    tokens = [str(answer_result)]
            except Exception as e:
                tokens = [str(answer_result)]
            for token in tokens:
                time.sleep(0.1)  # 각 토큰 사이 딜레이 (스트리밍 효과 시뮬레이션)
                yield token
            # 스트리밍 완료 후, 전체 답변을 데이터베이스에 저장합니다.
            full_answer = "".join(tokens)
            ChatMessage.objects.create(
                conversation=conversation,
                user=request.user,
                message=full_answer,
                is_user=False
            )
        response = StreamingHttpResponse(token_generator(), content_type='text/plain')
        response['X-Conversation-ID'] = conversation.id
        return response
    else:
        return JsonResponse({"error": "Invalid request"}, status=400)
