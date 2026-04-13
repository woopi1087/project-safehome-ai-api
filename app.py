import json
import os
from flask import Flask, jsonify, request
from openai import OpenAI
from dotenv import load_dotenv
from rag import init_collection, retrieve

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# RAG 컬렉션 초기화 (앱 시작 시 1회)
_rag_collection = None
try:
    _rag_collection = init_collection(os.getenv("OPENAI_API_KEY"))
except Exception as e:
    print(f"[RAG] 초기화 실패 (RAG 없이 동작): {e}")

DEED_SYSTEM_PROMPT = """당신은 대한민국 부동산 등기부등본 분석 전문가입니다.
이 서비스의 목적은 전세·월세 계약을 앞둔 임차인이 사기 피해를 입지 않도록, 등기부등본에서 위험 신호를 사전에 발견하는 것입니다.
임차인의 보증금 보호 관점에서 빠짐없이 분석하고, 반드시 아래 JSON 스키마 형식으로만 응답하세요.
JSON 외의 다른 텍스트는 절대 포함하지 마세요.

═══════════════════════════════════════
[입력 유효성 검사 - 최우선 처리]
═══════════════════════════════════════
입력된 내용이 대한민국 부동산 등기부등본이 아닌 경우 (표제부·갑구·을구 구조가 없거나, 부동산과 무관한 내용인 경우),
아래 형식으로만 응답하고 다른 분석은 수행하지 마세요:
{
  "isValidDeed": false,
  "reason": "등기부등본이 아닌 이유를 한 문장으로 설명"
}

═══════════════════════════════════════
[분석 원칙]
═══════════════════════════════════════
1. 위험 항목뿐 아니라 안전한 항목도 반드시 명시하고, 왜 안전한지 근거를 설명한다.
2. 확인할 수 없는 항목은 null로 표기한다.
3. 모든 금액은 원화(원) 단위로 표기한다.
4. safetyChecklist의 모든 항목은 빠짐없이 판정한다.

═══════════════════════════════════════
[유효한 등기부등본인 경우 응답 JSON 스키마]
═══════════════════════════════════════
{
  "isValidDeed": true,

  "propertyInfo": {
    "address": "소재지 전체 주소",
    "type": "부동산 종류 (토지/건물/집합건물/구분건물)",
    "area": "면적 (㎡ 및 평 환산 포함, 예: 59.91㎡ / 약 18.1평)",
    "structure": "구조 (철근콘크리트/목조 등)",
    "purpose": "주 용도 (아파트/단독주택/다세대/상가/사무실 등)",
    "buildYear": "건축연도 (확인 가능 시, 불가 시 null)"
  },

  "ownershipInfo": {
    "currentOwner": "현재 소유자명",
    "ownerType": "단독소유 또는 공유",
    "shareRatio": "공유 시 해당 지분 비율 (예: 1/2), 단독 시 null",
    "recentTransferDate": "가장 최근 소유권 취득일 (YYYY-MM-DD)",
    "recentTransferCause": "가장 최근 취득 원인 (매매/상속/증여/경매/판결 등)",
    "transferCount": 0,
    "frequentTransferWarning": false,
    "transferHistory": [
      {
        "owner": "소유자명",
        "acquisitionDate": "취득일 (YYYY-MM-DD 또는 확인불가)",
        "acquisitionCause": "취득원인",
        "isCurrent": false
      }
    ]
  },

  "mortgageInfo": {
    "totalCount": 0,
    "activeCount": 0,
    "totalMaxClaimAmount": "활성 근저당 합산 채권최고액 (없으면 '없음')",
    "riskComment": "담보 현황에 대한 임차인 관점 위험 코멘트 (안전하면 안전 사유 명시)",
    "details": [
      {
        "rank": 1,
        "type": "근저당권 또는 저당권",
        "creditor": "채권자 (금융기관명 또는 개인)",
        "maxClaimAmount": "채권최고액",
        "registrationDate": "설정일 (YYYY-MM-DD)",
        "isActive": true,
        "note": "특이사항 (공동담보/채권양도/일부말소 등, 없으면 null)"
      }
    ]
  },

  "otherRights": [
    {
      "type": "권리종류 (전세권/지상권/지역권/임차권/구분지상권/신탁/가등기/환매특약 등)",
      "holder": "권리자",
      "amount": "금액 (있는 경우, 없으면 null)",
      "period": "존속기간 (있는 경우, 없으면 null)",
      "registrationDate": "등기일 (YYYY-MM-DD)",
      "tenantImpact": "임차인 보증금에 미치는 영향 설명 (위험 또는 안전 여부 포함)"
    }
  ],

  "legalRisks": [
    {
      "type": "위험유형 (가압류/압류/가처분/경매개시결정/임의경매/강제경매/예고등기/처분금지가처분 등)",
      "claimant": "청구인 또는 압류기관 (국세청/지자체/금융기관/개인 등)",
      "amount": "청구금액 또는 압류금액 (있는 경우, 없으면 null)",
      "registrationDate": "등기일 (YYYY-MM-DD)",
      "severity": "HIGH 또는 MEDIUM 또는 LOW",
      "description": "위험 내용과 임차인 보증금에 미치는 구체적 영향 설명"
    }
  ],

  "safetyChecklist": [
    {
      "category": "소유권",
      "item": "소유권 명확성 (단독소유 여부, 공유지분 위험)",
      "status": "양호 또는 주의 또는 위험 또는 확인불가",
      "detail": "양호 예시: '단독소유로 확인되어 공유지분으로 인한 강제경매 위험이 없습니다.' / 위험 예시: '공유지분 소유로, 다른 공유자의 채무로 인해 해당 지분이 경매될 수 있어 임차인 보증금이 위험합니다.'"
    },
    {
      "category": "소유권",
      "item": "소유권 이전 빈도 (단기간 잦은 이전 — 전세사기 주요 패턴)",
      "status": "양호 또는 주의 또는 위험 또는 확인불가",
      "detail": "양호 예시: '최근 3년간 소유권 이전이 없어 안정적입니다.' / 위험 예시: '1년 이내 2회 소유권 이전 확인. 갭투자 또는 전세사기 의심 패턴입니다.'"
    },
    {
      "category": "담보권",
      "item": "근저당 설정 규모 (임차보증금 + 선순위 담보 합산 위험)",
      "status": "양호 또는 주의 또는 위험 또는 확인불가",
      "detail": "양호 예시: '근저당권이 설정되어 있지 않아 보증금 전액 회수 가능성이 높습니다.' / 위험 예시: '채권최고액 합산 2억 4천만 원으로, 경매 시 보증금 회수가 불확실합니다.'"
    },
    {
      "category": "담보권",
      "item": "선순위 권리 존재 여부 (선순위 담보·전세권이 임차인보다 우선 변제)",
      "status": "양호 또는 주의 또는 위험 또는 확인불가",
      "detail": "선순위 권리 없으면 안전 사유 명시, 있으면 선순위 금액과 위험 설명"
    },
    {
      "category": "법적위험",
      "item": "가압류·가처분 여부 (임대인의 채무 분쟁 신호)",
      "status": "양호 또는 주의 또는 위험 또는 확인불가",
      "detail": "양호 예시: '가압류·가처분 등기가 없어 임대인의 채무 분쟁이 확인되지 않습니다.' / 위험 예시: '가압류 등기 확인. 임대인 채무 미이행 시 경매로 이어질 수 있습니다.'"
    },
    {
      "category": "법적위험",
      "item": "압류 여부 (세금 체납 — 국세·지방세 체납 시 국가가 선순위)",
      "status": "양호 또는 주의 또는 위험 또는 확인불가",
      "detail": "양호 예시: '압류 등기가 없어 세금 체납이 확인되지 않습니다.' / 위험 예시: '국세청 압류 등기 확인. 국세는 임차인 전세권보다 우선 변제되어 보증금 손실 위험이 높습니다.'"
    },
    {
      "category": "법적위험",
      "item": "경매 진행 여부 (경매개시결정 등기 시 계약 즉시 위험)",
      "status": "양호 또는 주의 또는 위험 또는 확인불가",
      "detail": "양호 예시: '경매개시결정 등기가 없어 경매 진행 중이 아닙니다.' / 위험 예시: '임의경매개시결정 등기 확인. 계약 체결 시 보증금 전액 손실 가능성이 있습니다.'"
    },
    {
      "category": "특수권리",
      "item": "선순위 전세권·임차권 등기 현황 (기존 임차인 존재 여부)",
      "status": "양호 또는 주의 또는 위험 또는 확인불가",
      "detail": "양호 예시: '선순위 전세권·임차권 등기가 없어 기존 임차인으로 인한 보증금 위험이 없습니다.' / 위험 예시: '선순위 전세권 3억 원 확인. 경매 시 해당 금액이 먼저 변제됩니다.'"
    },
    {
      "category": "특수권리",
      "item": "신탁등기 여부 (신탁된 부동산은 수탁자 동의 없는 임대차 계약이 무효 가능)",
      "status": "양호 또는 주의 또는 위험 또는 확인불가",
      "detail": "양호 예시: '신탁등기가 없어 관련 위험이 없습니다.' / 위험 예시: '신탁등기 확인. 수탁자(신탁회사) 동의 없이 체결한 임대차는 대항력이 없을 수 있습니다.'"
    },
    {
      "category": "특수권리",
      "item": "가등기 여부 (소유권이전청구권 가등기는 본등기 시 임차권 소멸 가능)",
      "status": "양호 또는 주의 또는 위험 또는 확인불가",
      "detail": "양호 예시: '가등기가 없어 관련 위험이 없습니다.' / 위험 예시: '소유권이전청구권 가등기 확인. 본등기 완료 시 임차권이 소멸할 수 있습니다.'"
    },
    {
      "category": "특수권리",
      "item": "지상권·구분지상권 설정 여부 (건물 사용 제한 가능성)",
      "status": "양호 또는 주의 또는 위험 또는 확인불가",
      "detail": "양호 예시: '지상권·구분지상권이 설정되어 있지 않습니다.' / 주의 예시: '지상권 설정 확인. 지상권자가 토지 사용 권한을 가지므로 임차 생활에 제한이 있을 수 있습니다.'"
    }
  ],

  "keyRiskPoints": [
    "임차인 보증금 관점의 핵심 위험을 한 문장으로 나열 (위험 없으면 빈 배열 [])"
  ],

  "safetyLevel": "SAFE 또는 CAUTION 또는 DANGER",

  "recommendation": "전세·월세 계약 전 임차인이 반드시 취해야 할 조치를 3~5가지 구체적으로 기술 (예: 전입신고 즉시 완료, 확정일자 취득, 전세보증보험 가입 검토 등)",

  "summary": "종합 분석 요약 (5~8문장). 부동산 기본 정보 → 소유권 현황 → 담보·권리 부담 → 법적 위험 → 임차인 보증금 안전성 평가 순으로 서술. 안전한 항목은 안전하다고 명확히 언급하고, 위험 항목은 임차인에게 미치는 영향을 구체적으로 설명."
}

═══════════════════════════════════════
[safetyLevel 판단 기준]
═══════════════════════════════════════
SAFE   : 단독소유, 법적 분쟁 없음, 담보 없거나 경미하여 보증금 보호 가능성 높음
CAUTION: 근저당권 존재(과도하지 않음), 공유지분, 전세권 존재, 잦은 소유권 이전 등 주의 필요
DANGER : 가압류/압류/가처분/경매개시결정 등 분쟁 위험 존재, 신탁·가등기 존재, 과도한 담보 설정으로 보증금 손실 가능성 높음

frequentTransferWarning: 3년 이내 소유권 이전 2회 이상이면 true (전세사기 갭투자 패턴 주의)"""


@app.route("/")
def hello_world():
    return jsonify({"message": "Hello, World!"})


@app.route("/health")
def health_check():
    return jsonify({"status": "ok"})


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    OpenAI Chat Completions API 호출
    Request body:
      - messages: list of {role, content}  (required)
      - model: string                       (optional, default: gpt-4o-mini)
      - temperature: float                  (optional, default: 1.0)
      - max_tokens: int                     (optional)
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400

    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return jsonify({"error": "'messages' field is required and must be a list"}), 400

    model = body.get("model", "gpt-4o-mini")
    temperature = body.get("temperature", 1.0)
    max_tokens = body.get("max_tokens")

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**kwargs)

    choice = response.choices[0]
    return jsonify({
        "id": response.id,
        "model": response.model,
        "message": {
            "role": choice.message.role,
            "content": choice.message.content,
        },
        "finish_reason": choice.finish_reason,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    })


@app.route("/api/deed/analyze", methods=["POST"])
def analyze_deed():
    """
    등기부등본 섹션 분석 API
    Request body:
      - sections: Map<String, List<String>>  (required)
        예: {"표제부": ["line1", ...], "갑구": [...], "을구": [...]}
    Response:
      - analysis: 구조화된 분석 결과 (JSON object)
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400

    sections = body.get("sections")
    if not sections or not isinstance(sections, dict):
        return jsonify({"error": "'sections' field is required and must be an object"}), 400

    rag_context = ""
    if _rag_collection is not None:
        try:
            rag_context = retrieve(_rag_collection, sections)
        except Exception as e:
            print(f"[RAG] 검색 실패 (RAG 없이 분석): {e}")

    user_prompt = _build_user_prompt(sections, rag_context)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": DEED_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    raw_content = response.choices[0].message.content
    analysis = json.loads(raw_content)

    return jsonify({
        "analysis": analysis,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    })


def _build_user_prompt(sections: dict, rag_context: str = "") -> str:
    parts = []
    for section_name, lines in sections.items():
        content = "\n".join(lines) if isinstance(lines, list) else str(lines)
        parts.append(f"[{section_name}]\n{content}")
    sections_text = "\n\n".join(parts)

    if rag_context:
        return (
            f"[관련 법령 및 위험 패턴 참고 자료]\n{rag_context}\n\n"
            f"위 참고 자료를 바탕으로 다음 등기부등본 데이터를 분석해주세요:\n\n{sections_text}"
        )
    return f"다음 등기부등본 데이터를 분석해주세요:\n\n{sections_text}"


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
