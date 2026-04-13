# project-safehome-ai-api

전세·월세 계약을 앞둔 임차인을 위한 **등기부등본 AI 분석 서비스**의 Python AI 백엔드입니다.  
OpenAI GPT를 활용하여 등기부등본 텍스트를 분석하고, 임차인 보증금 보호 관점에서 위험 요소를 구조화된 JSON으로 반환합니다.

---

## 개요

```
project-safehome-ai-api/
├── app.py                      # Flask 애플리케이션 (메인)
├── rag/
│   ├── __init__.py             # init_collection, retrieve 노출
│   ├── loader.py               # ChromaDB 초기화 및 데이터셋 시딩
│   └── retriever.py            # 위험 키워드 감지 + 벡터 검색
├── data/
│   └── rag_legal_dataset.json  # RAG용 법령·판례·사기패턴 데이터셋
├── chroma_db/                  # ChromaDB 벡터 저장소 (첫 실행 시 자동 생성)
├── requirements.txt
├── .env.example
└── .env
```

---

## 기술 스택

| 항목 | 내용 |
|------|------|
| 언어 | Python 3.11+ |
| 프레임워크 | Flask 3.1.0 |
| AI | OpenAI API (gpt-4o-mini, text-embedding-3-small) |
| 벡터 DB | ChromaDB 1.x (PersistentClient, cosine 유사도) |
| 환경변수 | python-dotenv |

---

## 환경 설정

`.env.example`을 복사하여 `.env`를 생성하고 OpenAI API 키를 설정합니다.

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
```

---

## 실행

```bash
pip install -r requirements.txt
python app.py
```

서버가 `http://localhost:5000`에서 실행됩니다.

---

## API 엔드포인트

### `GET /`

서버 상태 확인

```json
{ "message": "Hello, World!" }
```

---

### `GET /health`

헬스체크

```json
{ "status": "ok" }
```

---

### `POST /api/chat`

OpenAI Chat Completions API 범용 호출 엔드포인트.

**Request Body**

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `messages` | `list` | 필수 | - | `[{role, content}]` 형식의 메시지 목록 |
| `model` | `string` | 선택 | `gpt-4o-mini` | 사용할 모델명 |
| `temperature` | `float` | 선택 | `1.0` | 생성 다양성 (0.0~2.0) |
| `max_tokens` | `int` | 선택 | - | 최대 생성 토큰 수 |

**Request 예시**

```json
{
  "messages": [
    { "role": "user", "content": "근저당권이 뭔가요?" }
  ],
  "model": "gpt-4o-mini",
  "temperature": 0.7
}
```

**Response**

```json
{
  "id": "chatcmpl-xxx",
  "model": "gpt-4o-mini",
  "message": {
    "role": "assistant",
    "content": "근저당권은 ..."
  },
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 150,
    "total_tokens": 170
  }
}
```

---

### `POST /api/deed/analyze`

등기부등본 섹션 데이터를 분석하여 임차인 보증금 보호 관점의 위험 평가를 반환합니다.

**Request Body**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `sections` | `object` | 필수 | `{"표제부": ["line1", ...], "갑구": [...], "을구": [...]}` |

**Request 예시**

```json
{
  "sections": {
    "표제부": ["[집합건물] 서울특별시 마포구 망원동 123-4", "전유부분 면적: 59.91㎡"],
    "갑구": ["순위1. 소유권보존 홍길동 2020.03.15"],
    "을구": ["순위1. 근저당권설정 채권최고액 금1억2천만원 국민은행 2020.04.01"]
  }
}
```

**Response**

```json
{
  "analysis": {
    "isValidDeed": true,
    "propertyInfo": { ... },
    "ownershipInfo": { ... },
    "mortgageInfo": { ... },
    "otherRights": [ ... ],
    "legalRisks": [ ... ],
    "safetyChecklist": [ ... ],
    "keyRiskPoints": [ ... ],
    "safetyLevel": "SAFE | CAUTION | DANGER",
    "recommendation": "...",
    "summary": "..."
  },
  "usage": {
    "prompt_tokens": 1200,
    "completion_tokens": 800,
    "total_tokens": 2000
  }
}
```

**safetyLevel 판단 기준**

| 레벨 | 조건 |
|------|------|
| `SAFE` | 단독소유, 법적 분쟁 없음, 담보 없거나 경미 |
| `CAUTION` | 근저당권 존재(과도하지 않음), 공유지분, 잦은 소유권 이전 |
| `DANGER` | 가압류·압류·가처분·경매개시결정, 신탁·가등기, 과도한 담보 |

**유효하지 않은 등기부등본인 경우 Response**

```json
{
  "analysis": {
    "isValidDeed": false,
    "reason": "등기부등본이 아닌 이유"
  }
}
```

---

## AI 분석 상세 — safetyChecklist 항목

`/api/deed/analyze`는 아래 11개 항목을 항상 판정합니다.

| 카테고리 | 항목 |
|----------|------|
| 소유권 | 소유권 명확성 (단독소유 여부, 공유지분 위험) |
| 소유권 | 소유권 이전 빈도 (3년 내 2회 이상 = 전세사기 위험 패턴) |
| 담보권 | 근저당 설정 규모 (선순위 담보 + 보증금 합산 위험) |
| 담보권 | 선순위 권리 존재 여부 |
| 법적위험 | 가압류·가처분 여부 |
| 법적위험 | 압류 여부 (세금 체납) |
| 법적위험 | 경매 진행 여부 |
| 특수권리 | 선순위 전세권·임차권등기 현황 |
| 특수권리 | 신탁등기 여부 |
| 특수권리 | 가등기 여부 |
| 특수권리 | 지상권·구분지상권 설정 여부 |

---

## RAG 데이터셋

`data/rag_legal_dataset.json`에 총 **19개 청크**가 포함되어 있습니다.

### 구성

| 분류 | 청크 수 | 내용 |
|------|---------|------|
| 주택임대차보호법 | 6 | 대항력(제3조), 우선변제권(제3조의2), 소액보증금(제8조), 계약갱신요구권(제6조의3), 차임증액제한(제7조), 임차권등기 |
| 전세사기피해자 특별법 | 2 | 피해자 인정 요건(제3조), 2024년 개정 사항 |
| 민법·민사집행법 | 2 | 전세권(민법 제303조), 경매개시결정(민사집행법 제83·91조) |
| 전세사기 패턴 | 5 | 갭투자형, 신탁등기 악용, 가등기 악용, 과다 근저당, 이중계약 |
| 실무 정보 | 2 | HUG 전세보증보험, 계약 전 체크리스트 |
| 민법 | 1 | 임차권등기(민법 제621조) |
| 민법 | 1 | 전세권(민법 제303조) |

### 청크 스키마

```json
{
  "id": "고유 ID",
  "source": "법령명 또는 출처",
  "article": "조항 번호",
  "title": "청크 제목",
  "content": "조문 또는 설명 전문",
  "keywords": ["검색 키워드"],
  "risk_context": "등기부 분석에서 이 청크를 활용하는 맥락",
  "tags": ["태그"]
}
```

### RAG 동작 흐름

```
앱 시작
  └─ init_collection()
       ├─ chroma_db/ 가 비어 있으면: 19개 청크를 text-embedding-3-small로 임베딩 저장
       └─ 이미 있으면: 기존 벡터 로드 (API 호출 없음)

POST /api/deed/analyze 요청
  ├─ 1. 갑구/을구 텍스트에서 위험 키워드 탐지 (16종)
  │       신탁 / 가등기 / 압류 / 가압류 / 경매 / 가처분
  │       근저당 / 전세권 / 임차권등기 / 소유권이전 등
  ├─ 2. ChromaDB cosine 유사도 검색 → 상위 4개 청크 반환
  ├─ 3. 프롬프트 조합
  │       [관련 법령 및 위험 패턴 참고 자료]
  │         검색된 청크 1~4
  │       다음 등기부등본 데이터를 분석해주세요:
  │         표제부 / 갑구 / 을구
  └─ 4. LLM 분석 → 구조화된 JSON 반환
```

RAG 초기화 또는 검색이 실패하더라도 기존 프롬프트만으로 분석이 정상 동작합니다 (graceful degradation).

### rag/ 모듈 구조

| 파일 | 역할 |
|------|------|
| `rag/loader.py` | `init_collection(api_key)` — ChromaDB PersistentClient 생성, 컬렉션이 비어 있으면 JSON 데이터셋을 임베딩하여 저장 |
| `rag/retriever.py` | `retrieve(collection, sections, n_results=4)` — 섹션 텍스트에서 위험 신호 감지 후 유사 청크 검색, 컨텍스트 문자열 반환 |
| `rag/__init__.py` | `init_collection`, `retrieve` 외부 노출 |

---

## 연계 서비스

이 API는 Spring Boot 기반 메인 서버(`project-safehome-api`)에서 호출됩니다.

| 서비스 | 역할 |
|--------|------|
| `project-safehome-api` | PDF 파싱, Job 관리, SSE 알림, 사용자 인증 |
| `project-safehome-ai-api` (본 서비스) | LLM 분석, RAG 검색 |

요청 흐름:
```
클라이언트
  → Spring Boot API (PDF 업로드 → 텍스트 추출 → 섹션 분리)
    → Flask AI API (POST /api/deed/analyze)
      → OpenAI GPT (+ RAG 컨텍스트)
        → 분석 결과 반환
```
