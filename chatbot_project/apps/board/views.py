import os
import time
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse, JsonResponse
from .forms import ChatForm
from .models import ChatMessage, Conversation, UploadedFile
from langchain_upstage import UpstageEmbeddings

from pdf.processed_documents import main  # PDF 텍스트 추출 함수 (수정된 load_document 포함)
from pdf.database_process import create_vector_store, add_documents
from pdf.llm_process import generate_response

from dotenv import load_dotenv
load_dotenv()

# --- Persistent Vector Store 초기화 (서버 시작 시 한 번 실행) ---

collection_name = "example_collection"
db_path = "pdf/chroma_langchain_db"
passage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage")

# vector_store 생성 (persist_directory와 collection_name을 이용하여 기존 데이터를 로드)
vector_store = create_vector_store(collection_name, db_path, passage_embeddings)

# 기존 데이터가 이미 존재하는지 확인합니다.
all_data = vector_store.get()
if all_data.get("ids") and len(all_data["ids"]) > 0:
    print("Persistent vector store data found, skipping re-embedding.")
else:
    print("Persistent vector store empty. 임베딩을 시작합니다.")
    # media/uploads 폴더 내의 모든 PDF 파일 처리하여 텍스트 추출
    uploads_folder = os.path.join(settings.MEDIA_ROOT, "uploads")
    all_texts = []
    if os.path.exists(uploads_folder):
        for file in os.listdir(uploads_folder):
            file_path = os.path.join(uploads_folder, file)
            if file.lower().endswith(".pdf"):
                # main() 함수는 file_path(문자열)를 위치 인자로 받아 텍스트(및 메타데이터) 리스트를 반환합니다.
                texts = main(file_path)
                all_texts.extend(texts)
    else:
        print(f"Uploads 폴더가 존재하지 않습니다: {uploads_folder}")
    
    print(f"전체 문서 개수: {len(all_texts)}")
    # 텍스트들을 vector_store에 추가 (임베딩 및 메타데이터 포함)
    vector_store = add_documents(vector_store, all_texts)
    print("문서 임베딩 완료.")

# print("현재 DB 데이터:", vector_store.get())

# --- view 함수들 ---
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
