from pathlib import Path

from marie.conf.helper import load_yaml
from marie.executor.pipeline.doc_determination_pipeline_executor import (
    get_pipeline_pages,
)


def test_doc_determination_service_config(tmp_path: Path) -> None:
    config_path = tmp_path / 'flow.yml'
    config_path.write_text(
        """jtype: Flow
executors:
  - uses:
      jtype: DocDeterminationPipelineExecutor
      with:
        pipelines:
          - pipeline:
              name: classification
              pages: [0, 2]
"""
    )
    executor = load_yaml(str(config_path), substitute=False)['executors'][0]['uses']
    assert executor['jtype'] == 'DocDeterminationPipelineExecutor'
    pipeline = executor['with']['pipelines'][0]['pipeline']
    assert get_pipeline_pages('input', 'test', {}, pipeline) == {0: [0, 2]}
