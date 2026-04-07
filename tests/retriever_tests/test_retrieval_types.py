import pytest

from tests.retriever_tests.retrieval_output_common import TEST_CASES
from tests.retriever_tests.retriever import run_case


@pytest.mark.parametrize("case", TEST_CASES, ids=[case["name"] for case in TEST_CASES])
def test_retrieval_output(case):
    report = run_case(case)
    status = "Passou" if report["passed"] else "Não passou"

    print(f"{status}, output:")
    print(report["output"])
    print(f"log: {report['log_path']}\n")

    assert report["passed"], report["output"]
