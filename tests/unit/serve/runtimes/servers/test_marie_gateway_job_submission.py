from marie.scheduler.llm_routing import (
    RoutingSubmissionError,
    reject_external_routing_selectors,
)


def test_gateway_submission_rejects_nested_caller_pool_selection() -> None:
    message = {
        'action': 'submit',
        'metadata': {
            'uri': 's3://bucket/document.tif',
            'features': [{'config': {'LLM_QUEUE_POOL_ID': 'document-small'}}],
        },
    }

    try:
        reject_external_routing_selectors(message)
    except RoutingSubmissionError as exc:
        assert exc.category == 'caller_pool_forbidden'
    else:
        raise AssertionError('gateway accepted a caller-supplied pool selector')
