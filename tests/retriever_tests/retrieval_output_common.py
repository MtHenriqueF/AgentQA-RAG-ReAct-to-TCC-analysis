import json
import os
import sys
from datetime import datetime


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.retriever import RERANK_TOP_N


LOGS_DIR = os.path.join(PROJECT_ROOT, "tests", "retriever_tests", "logs")
PREVIEW_CHARS = 220
TEXT_EXTENSIONS = {".pdf", ".md"}
CODE_EXTENSIONS = {".py", ".c", ".h", ".js", ".html", ".css"}

TEST_CASES = [
    {
        "name": "doc",
        "collections": ["teoria"],
        "query": "Quais abordagens os artigos descrevem para otimização de recursos em cirurgias eletivas?",
        "expect_text": True,
        "expect_code": False,
    },
    {
        "name": "codigo",
        "collections": ["logica"],
        "query": "Onde estão as definições de especialidade dos recursos materiais?",
        "expect_text": False,
        "expect_code": True,
    },
    {
        "name": "doc_e_codigo",
        "collections": ["teoria", "logica"],
        "query": "Como os artigos sobre otimização de cirurgias eletivas se conectam com as definições de especialidade dos recursos materiais no código?",
        "expect_text": True,
        "expect_code": True,
    },
]


def classify_content_type(extension):
    normalized_extension = (extension or "").lower()

    if normalized_extension in TEXT_EXTENSIONS:
        return "text"
    if normalized_extension in CODE_EXTENSIONS:
        return "code"
    return "unknown"


def normalize_preview(text: str, limit: int = PREVIEW_CHARS):
    compact_text = " ".join(text.split())
    return compact_text[:limit]


def serialize_documents(documents, limit: int = RERANK_TOP_N):
    serialized = []

    for index, doc in enumerate(documents[:limit], start=1):
        metadata = dict(doc.metadata)
        serialized.append(
            {
                "rank": index,
                "source": metadata.get("source"),
                "collection_name": metadata.get("collection_name"),
                "extension": metadata.get("extension"),
                "content_type": classify_content_type(metadata.get("extension")),
                "preview": normalize_preview(doc.page_content),
            }
        )

    return serialized


def build_summary(results):
    text_results = [item for item in results if item["content_type"] == "text"]
    code_results = [item for item in results if item["content_type"] == "code"]

    def unique_sources(items):
        return list(dict.fromkeys(item["source"] for item in items if item["source"]))

    return {
        "has_text_results": bool(text_results),
        "has_code_results": bool(code_results),
        "has_mixed_results": bool(text_results and code_results),
        "text_sources": unique_sources(text_results),
        "code_sources": unique_sources(code_results),
    }


def case_passed(case, summary):
    return (
        summary["has_text_results"] == case["expect_text"]
        and summary["has_code_results"] == case["expect_code"]
    )


def format_output(results, summary):
    if not results:
        return "nenhum documento retornado"

    top_result = results[0]
    parts = [
        f"top_1_tipo={top_result['content_type']}",
        f"top_1_arquivo={top_result['source']}",
        f"top_1_preview={top_result['preview']}",
        f"texto={summary['text_sources']}",
        f"codigo={summary['code_sources']}",
    ]
    return " | ".join(parts)


def save_report(report, prefix: str):
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOGS_DIR, f"{prefix}_{timestamp}.json")

    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    return path
