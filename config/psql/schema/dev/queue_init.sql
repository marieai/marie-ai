CREATE TABLE queue (
  id UUID PRIMARY KEY,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,

  scheduled_for timestamptz NOT NULL,
  failed_attempts INT NOT NULL,
  status INT NOT NULL,
  message JSONB NOT NULL
);
CREATE INDEX index_queue_on_scheduled_for ON queue (scheduled_for);
CREATE INDEX index_queue_on_status ON queue (status);