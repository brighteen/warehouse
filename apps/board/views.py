import time
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse, JsonResponse
from .forms import ChatForm
from .models import ChatMessage, Conversation, UploadedFile
from pdf.database_process import create_vector_store
from pdf.llm_process import generate_response
from langchain_upstage import UpstageEmbeddings

# 변경: db_path 경로를 pdf/chroma_langchain_db로 수정
vector_store = create_vector_store(
    collection_name="example_collection",
    db_path="pdf/chroma_langchain_db",
    passage_embeddings=UpstageEmbeddings(model="solar-embedding-1-large-passage")  # UpstageEmbeddings는 llm_process 내에서 호출하거나 필요 시 추가
)

@login_required(login_url="/login/")
def chat_view(request, conversation_id=None):
    if conversation_id:
        conversation = get_object_or_404(Conversation, pk=conversation_id, user=request.user)
    else:
        conversation = None
    form = ChatForm()
    messages = conversation.messages.order_by("timestamp") if conversation else []
    conversation_list = Conversation.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "board/chat.html", {
        "form": form,
        "messages": messages,
        "conversation_list": conversation_list,
        "active_conversation": conversation,
    })

@login_required(login_url="/login/")
def stream_answer(request):
    if request.method == "POST":
        conversation_id = request.POST.get("conversation_id")
        question = request.POST.get("question")
        if conversation_id:
            conversation = get_object_or_404(Conversation, pk=conversation_id, user=request.user)
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

        def token_generator():
            answer_result, _ = generate_response(vector_store, question)
            try:
                tokens = answer_result.get("conclusion", {}).get("conclusion", [])
                if not tokens:
                    tokens = [str(answer_result)]
            except Exception as e:
                tokens = [str(answer_result)]
            for token in tokens:
                time.sleep(0.05)
                yield token
            yield "\n[[REASON_DELIMITER]]\n"
            reason = answer_result.get("reason", "")
            if reason:
                yield reason
            full_answer = "".join(tokens)
            if reason:
                full_answer += "\n" + reason
            ChatMessage.objects.create(
                conversation=conversation,
                user=request.user,
                message=full_answer,
                is_user=False
            )
        response = StreamingHttpResponse(token_generator(), content_type="text/plain")
        response["X-Conversation-ID"] = conversation.id
        return response
    else:
        return JsonResponse({"error": "Invalid request"}, status=400)

@login_required(login_url="/login/")
def upload_file(request):
    if not request.user.is_staff:
        return JsonResponse({'status': 'error', 'message': 'Permission denied'}, status=403)
    if request.method == "POST":
        if "file" in request.FILES:
            uploaded_file = request.FILES["file"]
            uf = UploadedFile.objects.create(user=request.user, file=uploaded_file)
            return JsonResponse({'status': 'success', 'file_url': uf.file.url, 'file_name': uf.file.name})
        else:
            return JsonResponse({'status': 'error', 'message': 'No file uploaded'}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)

from django.contrib.admin.views.decorators import staff_member_required

@staff_member_required
def conversation_list_admin(request):
    conversations = Conversation.objects.all().order_by("-created_at")
    return render(request, "admin/conversation_list.html", {"conversations": conversations})

@staff_member_required
def file_upload_list_admin(request):
    files = UploadedFile.objects.all().order_by("-uploaded_at")
    return render(request, "admin/file_upload_list.html", {"files": files})
