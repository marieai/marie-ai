def get_full_schema() -> dict:
    """Get full schema
    :return: the full schema for Jina core as a dict.
    """
    from marie import __version__
    from marie.importer import IMPORTED
    from marie.schemas.deployment import schema_deployment
    from marie.schemas.executor import schema_all_executors
    from marie.schemas.flow import schema_flow
    from marie.schemas.gateway import schema_gateway
    from marie.schemas.meta import schema_metas

    definitions = {}
    for s in [
        schema_gateway,
        schema_all_executors,
        schema_flow,
        schema_metas,
        schema_deployment,
        IMPORTED.schema_executors,
    ]:
        definitions.update(s)

    return {
        '$id': f'https://api.jina.ai/schemas/{__version__}.json',
        '$schema': 'http://json-schema.org/draft-07/schema#',
        'description': 'The YAML schema of Jina objects (Flow, Executor).',
        'type': 'object',
        'oneOf': [{'$ref': '#/definitions/Jina::Flow'}]
        + [{"$ref": f"#/definitions/{k}"} for k in IMPORTED.schema_executors.keys()],
        'definitions': definitions,
    }
