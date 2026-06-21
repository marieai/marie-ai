from marie.executor.mock.mock_executor import IntegrationExecutorMock


def test_integration_executor_mock_does_not_mask_annotator_llm_endpoint() -> None:
    request_map = IntegrationExecutorMock.requests_by_class["IntegrationExecutorMock"]

    assert "/document/process" in request_map
    assert "/annotator/llm" not in request_map
