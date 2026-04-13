import chromadb

# 등기부 텍스트에서 탐지할 위험 신호 키워드
_RISK_SIGNALS = [
    # 등기부 갑구 위험 신호
    "신탁", "가등기", "압류", "가압류", "경매",
    "가처분", "소유권이전", "예고등기", "처분금지",
    "임의경매", "강제경매", "환매",
    # 등기부 을구 위험 신호
    "근저당", "전세권", "임차권등기", "채권최고액",
    "지상권", "구분지상권",
    # 사기 패턴 연관 신호
    "법인", "공유", "지분",
]

_DEFAULT_QUERY = "임차인 보증금 보호 전세사기 위험 대항력 우선변제권"


def retrieve(collection: chromadb.Collection, sections: dict, n_results: int = 4) -> str:
    """
    등기부 섹션에서 위험 신호를 감지하여 관련 법령·패턴 청크를 검색하고
    LLM에 주입할 컨텍스트 문자열로 반환.
    """
    query = _build_query(sections)
    results = collection.query(query_texts=[query], n_results=n_results)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return ""

    parts = [
        f"[참고: {m['source']} {m['article']}]\n{doc}"
        for doc, m in zip(docs, metas)
    ]
    return "\n\n---\n\n".join(parts)


def _build_query(sections: dict) -> str:
    """등기부 섹션 텍스트에서 위험 키워드를 추출하여 검색 쿼리 생성."""
    all_text = " ".join(
        " ".join(v) if isinstance(v, list) else str(v)
        for v in sections.values()
    )
    found = [kw for kw in _RISK_SIGNALS if kw in all_text]
    return " ".join(found) if found else _DEFAULT_QUERY
