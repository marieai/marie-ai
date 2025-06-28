ALTER SYSTEM SET max_connections = 500;

CREATE EXTENSION IF NOT EXISTS vector;


-- Create extension if not exists
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

CREATE TABLE IF NOT EXISTS embeddings (
  id SERIAL PRIMARY KEY,
  embedding vector,
  text text,
  created_at timestamptz DEFAULT now()
);