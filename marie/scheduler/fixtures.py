import os


def create_sql_from_file(schema: str, file_path: str) -> str:
    with open(file_path, encoding='utf-8') as sql_file:
        sql = sql_file.read()

    for placeholder in ('{schema}', '{SCHEMA}', '$(schema)', '__SCHEMA__'):
        sql = sql.replace(placeholder, schema)

    return f"-- SQL from file: {os.path.basename(file_path)}\n{sql}\n;"
