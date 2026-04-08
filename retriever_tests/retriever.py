import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.retriever import retrieve_ranked_documents

from tests.retrieval_output_common import (
    TEST_CASES,
    build_summary,
    case_passed,
    format_output,
    save_report,
    serialize_documents,
)


def run_case(case):
    documents = retrieve_ranked_documents(
        collection_names=case["collections"],
        query=case["query"],
    )
    results = serialize_documents(documents)
    summary = build_summary(results)
    passed = case_passed(case, summary)
    output = format_output(results, summary)

    report = {
        "mode": "rerank",
        "case": case["name"],
        "collections": case["collections"],
        "query": case["query"],
        "passed": passed,
        "output": output,
        "results": results,
        "summary": summary,
    }
    report["log_path"] = save_report(report, prefix=f"{case['name']}_rerank")
    return report


def main():
    for case in TEST_CASES:
        report = run_case(case)
        status = "Passou" if report["passed"] else "Não passou"
        print(f"{status}, output:")
        print(report["output"])
        print(f"log: {report['log_path']}\n")


if __name__ == "__main__":
    main()
