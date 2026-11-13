import pytest

from marie.kb.code.query import native_query, validate_branches


def atom(field, value, negated=False, regex=False):
    return {'field': field, 'value': value, 'negated': negated, 'regex': regex}


def test_compiled_query_cannot_escape_the_owned_handle():
    query = native_query('a' * 64, [atom('content', '") OR repo:.*')], False)
    assert query.startswith('repo:^' + 'a' * 64 + '$ case:no (')
    assert '\\"' in query
    assert native_query('a' * 64, [atom('file', r'\.py$', regex=True)], True).endswith(
        '(file:"\\\\.py$")'
    )


@pytest.mark.parametrize(
    'branches',
    [
        [],
        [[]],
        [[atom('unknown', 'x')]],
        [[{**atom('content', 'x'), 'negated': 'false'}]],
        [[atom('content', 'x' * 1001)]],
        [[atom('content', 'x')]] * 9,
    ],
)
def test_external_branch_shape_is_bounded_and_validated(branches):
    with pytest.raises(ValueError):
        validate_branches(branches)


def test_metadata_filters_never_reach_the_native_query_compiler():
    with pytest.raises(ValueError):
        native_query('a' * 64, [atom('repo', 'other')], True)
    with pytest.raises(ValueError):
        native_query('../other', [atom('content', 'x')], True)
