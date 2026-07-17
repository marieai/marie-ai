from __future__ import annotations

from pathlib import Path

import yaml

from marie.prompt import PromptTemplate

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
ANNOTATOR_DIR = ROOT / 'config' / 'extract' / 'TID-longextract-bench' / 'annotator'


def test_annotator_config_references_prompt_assets() -> None:
    config = yaml.safe_load((ANNOTATOR_DIR / 'config.yml').read_text())
    assert config['processing']['convert_to_structure'] is False
    annotators = config['annotators']
    expected = {
        'longextract-aggregation-policy': 'longextract-aggregation-policy.j2',
        'longextract-unit-extract': 'longextract-unit-extract.j2',
        'longextract-repair': 'longextract-repair.j2',
    }
    for key, prompt_name in expected.items():
        model_config = annotators[key]['model_config']
        assert model_config['prompt_path'] == f'./{prompt_name}'
        assert model_config['system_prompt_text'].strip()
        assert model_config['expect_output'] == 'json'
        assert model_config['temperature'] == 0.0
        assert (ANNOTATOR_DIR / prompt_name).exists()

    unit_config = annotators['longextract-unit-extract']['model_config']
    table_config = annotators['tables']['model_config']
    assert table_config['model_name'] == 'qwen_v3_30b_instruct'
    assert table_config['prompt_path'] == './tables.j2'
    assert table_config['refine_prompt_path'] == './tables-refine.j2'
    assert unit_config['refine_passes'] == 1
    assert unit_config['pass_models'] == ['qwen_v3_30b_instruct']
    assert unit_config['pass_temperatures'] == [0.0]
    assert table_config['refine_passes'] == 1
    assert annotators['longextract-unit-extract']['parser'] == 'noop'
    assert annotators['longextract-aggregated']['parser'] == 'longextract-aggregated'


def test_table_prompts_are_layout_local_and_model_agnostic() -> None:
    for name in ('tables.j2', 'tables-refine.j2'):
        prompt = (ANNOTATOR_DIR / name).read_text()
        assert 'LoRA' not in prompt
        assert 'EOB' not in prompt
        assert 'claim' not in prompt.casefold()
        assert 'PAGE_TEXT' in prompt
        assert 'OCR_TEXT' not in prompt


def _variables(name: str) -> set[str]:
    template = PromptTemplate.from_file_with_fallback(
        name, prompt_dir=str(ANNOTATOR_DIR)
    )
    return template.expected_variables


def test_unit_prompts_declare_runtime_variables() -> None:
    required = {
        'AGGREGATION_POLICY_JSON',
        'SCHEMA_UNITS_JSON',
        'PAGE_NUMBER',
        'PAGE_TEXT',
        'PAGE_TABLES_JSON',
        'DOCUMENT_METADATA_JSON',
        'OUTPUT_CONTRACT_JSON',
    }
    assert required.issubset(_variables('longextract-unit-extract.j2'))
    assert required | {'PREVIOUS_EXTRACTION'} <= _variables(
        'longextract-unit-extract-refine.j2'
    )


def test_unit_prompts_leave_aggregation_policy_to_parser() -> None:
    prompts = {
        name: (ANNOTATOR_DIR / name).read_text()
        for name in (
            'longextract-unit-extract.j2',
            'longextract-unit-extract-refine.j2',
        )
    }
    for prompt in prompts.values():
        assert 'AGGREGATION_POLICY_JSON' in prompt
        assert 'Document aggregation policy' in prompt
        assert 'carry_fields' not in prompt
        assert 'sequence_fields' not in prompt
    assert 'parser will attach it' in prompts['longextract-unit-extract.j2']
    assert 'cross-page aggregation' in prompts['longextract-unit-extract-refine.j2']


def test_unit_prompts_verify_exact_leaf_evidence() -> None:
    prompts = {
        name: (ANNOTATOR_DIR / name).read_text()
        for name in (
            'longextract-unit-extract.j2',
            'longextract-unit-extract-refine.j2',
        )
    }
    for prompt in prompts.values():
        assert 'apply only the\n  transformations required by that field' in prompt
        assert 'description does not explicitly remove or\n  reformat' in prompt
        assert 'in at least one data column' in prompt
        assert 'selected array schema' in prompt
        assert 'subject of a data row' in prompt
        assert 'page-table row inventory' in prompt
        assert 'Every inventoried source line with a value' in prompt
        assert 'visual row has no values in any data column' in prompt
        assert 'Assign headings only after all data rows have been enumerated' in prompt
        assert 'Do not append\n  an adjacent title, subtitle' in prompt
        assert 'source delimiter never removes' in prompt
        assert 'Preserve all source delimiters' in prompt
        assert '`( value )` to `(value)`, never to `value`' in prompt
        assert 'If its value is copied' in prompt
        assert 'has any data-column value' in prompt
        assert 'stop before an adjacent line whose role is' in prompt
        assert 'current-of-total marker in\n  the page text' in prompt
        assert 'when a supplied continuation flag contradicts it' in prompt or (
            'when the supplied flag contradicts it' in prompt
        )
    assert (
        'every required leaf has been verified'
        in prompts['longextract-unit-extract-refine.j2']
    )


def test_unit_prompt_values_are_inserted_once() -> None:
    expected_variables = {
        'longextract-unit-extract.j2': {
            'AGGREGATION_POLICY_JSON',
            'SCHEMA_UNITS_JSON',
            'PAGE_NUMBER',
            'PAGE_TEXT',
            'PAGE_TABLES_JSON',
            'DOCUMENT_METADATA_JSON',
            'OUTPUT_CONTRACT_JSON',
        },
        'longextract-unit-extract-refine.j2': {
            'AGGREGATION_POLICY_JSON',
            'SCHEMA_UNITS_JSON',
            'PAGE_NUMBER',
            'PAGE_TEXT',
            'PAGE_TABLES_JSON',
            'DOCUMENT_METADATA_JSON',
            'OUTPUT_CONTRACT_JSON',
            'PREVIOUS_EXTRACTION',
        },
    }
    for name, variables in expected_variables.items():
        prompt = (ANNOTATOR_DIR / name).read_text()
        for variable in variables:
            assert prompt.count(variable) == 1


def test_table_prompts_preserve_indexes_and_repeated_header_continuations() -> None:
    for name in ('tables.j2', 'tables-refine.j2'):
        prompt = (ANNOTATOR_DIR / name).read_text()
        assert 'structured index or list' in prompt
        assert 'Repeated column headers do not establish a new logical table' in prompt
        assert 'mandatory continuation audit' in prompt
        assert 'current number is 1, set is_continuation to false' in prompt
        assert 'current number is greater than 1, set is_continuation to true' in prompt
        assert 'physical document page number' in prompt
        assert 'section or subgroup label inside the table' in prompt
        assert 'separate current-of-total marker' in prompt
        assert 'marker decision is authoritative' in prompt
        assert 'detected data column' in prompt
        assert 'include it in rows as a data row' in prompt
        assert 'Only a line with no data-column values' in prompt
        assert 'direct transcription of the marker' in prompt
        assert 'is_continuation matches' in prompt
        assert 'no new table title or section establishes a new table' in prompt
        assert 'prepared-text' in prompt


def test_policy_prompt_declares_schema_contract() -> None:
    assert {
        'SCHEMA_UNITS_JSON',
        'POLICY_OUTPUT_CONTRACT_JSON',
    }.issubset(_variables('longextract-aggregation-policy.j2'))


def test_repair_prompt_declares_artifact_variables() -> None:
    assert {
        'SCHEMA_JSON',
        'STITCHED_RESULT_JSON',
        'VERIFICATION_FINDINGS_JSON',
        'REPAIR_SCOPE_JSON',
        'OUTPUT_CONTRACT_JSON',
    }.issubset(_variables('longextract-repair.j2'))


def test_active_config_matches_benchmark_assets() -> None:
    active_dir = (
        REPO_ROOT / 'config' / 'extract' / 'TID-longextract-bench' / 'annotator'
    )
    for source in ANNOTATOR_DIR.iterdir():
        assert (active_dir / source.name).read_bytes() == source.read_bytes()
    mapper = ROOT / 'config' / 'extract' / 'TID-longextract_bench' / 'mapper.yml'
    active_mapper = (
        REPO_ROOT / 'config' / 'extract' / 'TID-longextract_bench' / 'mapper.yml'
    )
    assert active_mapper.read_bytes() == mapper.read_bytes()
