import pytest

from marie.kb.datasource import datasource_items


def test_saved_output_references_keep_identity_and_do_not_copy_content():
    reference = {
        'kind': 'text',
        'uri': 's3://saved/wiki/a.md',
        'media_type': 'text/markdown',
        'sha256': 'a' * 64,
    }
    records = [{'id': 'wiki:42', 'uri': 's3://saved/wiki/a.md', 'input_ref': reference}]
    assert datasource_items(records) == records


@pytest.mark.parametrize(
    'records',
    [
        [{'id': 'chess:1', 'moves': ['e4', 'e5']}],
        [{'id': 'a', 'uri': 'https://provider/private'}],
        [{'id': 'a', 'uri': 's3://saved/a'}, {'id': 'a', 'uri': 's3://saved/b'}],
        [{'id': '', 'uri': 's3://saved/a'}],
    ],
)
def test_document_adapter_rejects_unsupported_or_ambiguous_output(records):
    with pytest.raises(ValueError):
        datasource_items(records)


def test_explicit_removal_does_not_imply_deleting_unseen_items():
    assert datasource_items([{'id': 'wiki:42', 'deleted': True}]) == [
        {'id': 'wiki:42', 'deleted': True}
    ]


def test_original_documents_keep_the_delivered_content_digest():
    records = [
        {'id': 'pdf:1', 'uri': 's3://saved/tenants/one/a.pdf', 'sha256': 'b' * 64}
    ]
    assert datasource_items(records) == records
