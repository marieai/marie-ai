import pytest

from hubble.executor.hubio import HubIO
from marie.orchestrate.pods.factory import PodFactory
from marie.parsers import set_pod_parser


@pytest.mark.parametrize(
    'uses', ['marieai+docker://jina-ai/DummyExecutor']
)
def test_container_pod(mocker, monkeypatch, uses):
    mock = mocker.Mock()

    def _mock_pull(self):
        return 'docker://jinahub/dummy_executor'

    monkeypatch.setattr(HubIO, 'pull', _mock_pull)

    args = set_pod_parser().parse_args(['--uses', uses])
    pod = PodFactory.build_pod(args)
    assert pod.args.uses == 'docker://mariehub/dummy_executor'
    assert pod.name == 'ContainerPod'
