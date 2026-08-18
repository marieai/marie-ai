from pathlib import Path

from marie.conf.helper import load_yaml


def test_doc_determination_service_configs() -> None:
    config_dir = Path('config/service/doc_determination')
    configs = sorted(config_dir.glob('marie-4.5.0-*.yml'))

    assert len(configs) == 4
    for config_path in configs:
        config = load_yaml(str(config_path), substitute=False)
        assert config['jtype'] == 'Flow'
        assert config['scheduler']['psql']['queue_names'] == ['doc_determination']
        assert config['executors']

    doc_class = load_yaml(
        str(config_dir / 'marie-4.5.0-doc-class.yml'), substitute=False
    )
    rule = doc_class['executors'][0]['uses']['with']['pipelines'][0]['pipeline'][
        'pages'
    ][0]
    assert rule['min_conf'] == 0.995

    split = load_yaml(
        str(config_dir / 'marie-4.5.0-split-boundary.yml'), substitute=False
    )
    exclusions = split['executors'][0]['uses']['with']['pipelines'][0]['pipeline'][
        'pages'
    ][0]
    assert exclusions['method'] == 'exclude'
    assert exclusions['classifications'] == [
        'OTHER',
        'SUBSTITUTION-DOC',
        'BLANK',
        'ENVELOPE-BACK',
        'CHECK-BACK',
        'PATPAY-BACK',
    ]
