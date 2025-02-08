✅변경된 부분

### base.html
1. 사이드바 추가 (nav 태그)
- 게시판 글 목록을 사이드바 형태로 표시
- board_list.title을 클릭하면 해당 글 상세 페이지로 이동
- 글쓰기 버튼을 사이드바에 추가
2. 메인 컨텐츠 (main 태그)
- 기존 content 블록을 col-md-9로 설정하여 오른쪽에 글쓰기 화면 표시

### board_create.html
1. form-control을 사용하여 글쓰기 폼을 깔끔하게 정리
2. placeholder를 추가하여 입력 가이드 제공
3. 버튼을 w-100으로 설정하여 전체 폭 차지

### board_list.html
1. table 대신 카드 형식으로 게시글 목록 정리
2. 제목을 클릭 가능하게 설정
3. 작성자, 작성 날짜를 작은 글씨(text-muted)로 표시하여 깔끔한 디자인

### board_detail.html
1. 게시글 정보 카드를 더 깔끔하게 정리
2. 댓글 목록을 카드 형태로 표시
3. 댓글 입력 폼을 간단하게 변경