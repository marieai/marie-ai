from docarray import DocList
from docarray.documents import TextDoc


def test_repository_executor_loads_all_transport_endpoints():
    from marie.executor.kb.repository_executor import RepositoryExecutor

    endpoints = RepositoryExecutor.requests_by_class['RepositoryExecutor']
    for name in (
        '/refresh_repository',
        '/code_search',
        '/open_code',
        '/code_navigation',
        '/delete_code_source',
    ):
        endpoint = endpoints[name]
        endpoint.validate()
        assert endpoint.request_schema == DocList[TextDoc]
