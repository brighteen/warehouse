# 키워드
- 업무 자동화
- 생성형 AI
- GPTs(프롬프트 제작)

# 메모

## 챗봇
1. 싱글턴
2. 멀티턴
3. 지침 기반 챗봇화
- [참고Notion](https://comento.notion.site/1b93cb8bc55a815ab22dd13a0ebb5ebd)
- 위 내용들 기반으로 내 GPTs 만들기


### 챗봇 구성
1. 프롬프트
2. 프롬프트 + 지식
3. 프롬프트 + API(데이터를 가져와서 사용)

## 생성형 AI의 제약조건 : 
- 보안 문제
- **할루시네이션**

### 할루시네이션을 줄이는 프롬프트 최적화
- 파인튜닝을 통한 LLM 최적화
- RAG(검색증강생성) 기반 정교화
- 프롬프트 엔지니어링 최적화

## 주제선정
- 아이디에이션
- 리서치

> **gamma로 ppt 생성, napkin으로 도식화**

**PPT 제목, 헤드라인 메시지, 내용으로 정리**

> **각 주제별 워크플로우 생각을 잘 해야함.**

### 실습 프롬프트
```
너는 전 세계 최고의 맛집 블로거야. 대한민국 전주에 위치한 전주대학교를 출발점으로, 도보 20분 내에 위치한 2인 방문에 최적의 맛집 및 카페 추천 플랜을 수립해줘.  
조건은 다음과 같아:
1. 일식당은 초밥, 라멘, 돈까스 등 다양한 메뉴를 제공하는 곳을 선정하고,  
   - 영업시간, 영업 요일, 메뉴, 브레이크타임, 객관적인 후기를 반영하여 추천해줘.
   - 치히로 전북대점과 같이 전주대에서 도보 20분 이상 걸리는 곳은 제외할 것.
2. 카페는 분위기 좋은 곳으로, 강아지나 고양이 동반이 가능한 곳을 선정하고,  
   - 영업시간, 메뉴, 브레이크타임 등 세세한 정보를 포함시켜줘.
3. 각 업체의 클릭 가능한 링크(네이버 블로그, 플레이스, 인스타그램 등 공개 자료)를 포함시켜서 최신 정보로 검증된 내용을 제공해줘.
4. 최종 결과는 크게 제목, 헤드라인 메시지, 내용(추천 업체 목록 및 정보 요약)으로 정리해줘.

이 조건을 충족하는 최종 추천 플랜을 한 번에 도출해줘.
```
---

## 이미지 생성
- freepik

## 웹사이트 생성
- figma
- https://makereal.tldraw.com/

## 논문
- scispace

## 엑셀 활용
VBA로 자연어를 엑셀로 처리
> 1 2 3 (오류발생) 4 5
> 3에서 프롬프트 정리, 코드 저장 후 새 대화로 넘어가서 다시 시작

https://comento.notion.site/1b93cb8bc55a811f95c6f19ebdbc49f1