# CLAUDE.md — project-safehome-ai-api

This file provides guidance to Claude Code when working with this repository.

## Build & Run Commands

```bash
# 의존성 설치
pip install -r requirements.txt

# 개발 서버 시작 (http://localhost:5000)
python app.py

# 환경 변수 설정 (.env 파일 필요)
cp .env.example .env
# OPENAI_API_KEY=sk-... 입력 후 저장
```

## Tech Stack

Python 3.11+, Flask 3.1.0, OpenAI API (gpt-4o-mini), ChromaDB 1.x (PersistentClient), python-dotenv

## Architecture

단일 Flask 앱(`app.py`) + RAG 모듈(`rag/`). 앱 시작 시 ChromaDB 컬렉션을 초기화하고 전역 변수로 유지.

```
project-safehome-ai-api/
├── app.py               # Flask 앱 진입점 (라우트 + 시스템 프롬프트)
├── rag/
│   ├── __init__.py      # init_collection, retrieve 공개
│   ├── loader.py        # ChromaDB 초기화 + 데이터셋 시딩
│   └── retriever.py     # 위험 키워드 감지 + 벡터 유사도 검색
├── data/
│   ├── rag_legal_dataset.json       # 법령·판례 19개 청크
│   └── rag_news_cases_dataset.json  # 전세사기 뉴스·패턴
├── chroma_db/           # 벡터 저장소 (최초 실행 시 자동 생성)
├── .env                 # OPENAI_API_KEY (git 제외)
└── requirements.txt
```

## API Endpoints

| Method | Path | 설명 |
|--------|------|------|
| GET | `/` | 서버 상태 확인 |
| GET | `/health` | 헬스 체크 |
| POST | `/api/chat` | OpenAI Chat 범용 래퍼 |
| POST | `/api/deed/analyze` | 등기부등본 분석 **(핵심)** |

### POST /api/deed/analyze

**Request:**
```json
{
  "sections": {
    "표제부": ["line1", "line2"],
    "갑구": ["line1", ...],
    "을구": ["line1", ...]
  },
  "leaseType": "전세 또는 월세 (optional)"
}
```

**Response:**
```json
{
  "analysis": {
    "isValidDeed": true,
    "safetyLevel": "SAFE | CAUTION | DANGER",
    "propertyInfo": { ... },
    "ownershipInfo": { ... },
    "mortgageInfo": { ... },
    "otherRights": [ ... ],
    "legalRisks": [ ... ],
    "keyRiskPoints": [ ... ],
    "overallRiskSummary": "종합 위험도 요약 (3~5문장)",
    "leaseSpecificAnalysis": {
      "leaseType": "월세 | 전세 | 미지정",
      "summary": "임대차 유형별 종합 분석",
      "checkItems": [
        {
          "category": "보증금 안전성 | 계약 전 확인 | 법적 보호 | 등기 권고 | 기타",
          "title": "확인 사항 제목",
          "description": "상세 설명",
          "priority": "필수 | 권장 | 참고"
        }
      ]
    },
    "safetyChecklist": [ ... ],
    "recommendation": "...",
    "summary": "..."
  },
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
}
```

**유효하지 않은 입력 시:**
```json
{ "analysis": { "isValidDeed": false, "reason": "..." } }
```

## RAG Module

2단계 검색 파이프라인:
1. **키워드 감지** (`retriever.py`): 섹션 텍스트에서 16개 위험 신호 키워드 탐지
2. **벡터 검색** (`ChromaDB`): `text-embedding-3-small`으로 top-4 청크 반환

```python
# RAG 흐름
sections → _build_query() → collection.query() → 컨텍스트 문자열
```

**데이터셋 구성:**
- 주택임대차보호법 (대항력, 우선변제권, 최우선변제권)
- 전세사기특별법 2024 개정
- 민법·집행법·근저당 판례
- 갭투자·신탁등기 악용 사기 패턴 19개 청크

RAG 초기화 실패 시 graceful degradation — RAG 없이 LLM 단독 분석으로 동작.

## Key Patterns

### 새 엔드포인트 추가
```python
@app.route("/api/new-endpoint", methods=["POST"])
def new_endpoint():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400
    # 처리 로직
    return jsonify({...})
```

### OpenAI 호출 규칙
- 분석용 모델: `gpt-4o-mini`, `temperature=0.2` (일관성 우선)
- 범용 채팅: `gpt-4o-mini`, `temperature=1.0` (기본값)
- 구조화 출력: `response_format={"type": "json_object"}` 사용
- 응답 후 `json.loads(raw_content)`로 파싱

### 에러 처리
- 전역 `@app.errorhandler(Exception)` 핸들러가 500 응답 반환
- RAG/ChromaDB 오류는 try/except로 감싸고 로그만 출력 후 계속 진행

## RAG 데이터셋 확장

`data/` 에 JSON 파일 추가 후 `rag/loader.py`의 `DATA_FILES` 목록에 등록:
```python
DATA_FILES = [
    "data/rag_legal_dataset.json",
    "data/rag_news_cases_dataset.json",
    "data/new_dataset.json",   # 추가
]
```
청크 스키마: `{ "source": "법령명", "article": "조항", "content": "내용" }`

## Service Dependencies

- **호출 주체**: `project-safehome-api` (Spring Boot, `http://localhost:8080`)
- Spring Boot가 PDF 파싱 후 섹션 데이터를 `POST /api/deed/analyze`로 전송
- 분석 결과를 JSON으로 반환하면 Spring Boot가 SSE로 클라이언트에 전달

## 환경 변수

| 변수명 | 설명 | 필수 |
|--------|------|------|
| `OPENAI_API_KEY` | OpenAI API 키 | 필수 |
