-- File: 016_exponential_backoff.sql
-- Description: Function to calculate exponential backoff delay for retries
-- Dependencies: 001_schema.sql

-- Create or replace the exponential backoff function (idempotent)
CREATE OR REPLACE FUNCTION {schema}.exponential_backoff(retry_delay INT, retry_count INT)
RETURNS TIMESTAMP WITH TIME ZONE AS $$
BEGIN
    -- Calculate exponential backoff with jitter
    -- Formula: base_delay * (2^retry_count / 2) + random_jitter
    RETURN NOW() + (
        retry_delay * (2 ^ LEAST(16, retry_count + 1) / 2) +
        retry_delay * (2 ^ LEAST(16, retry_count + 1) / 2) * random()
    ) * INTERVAL '1 second';
END;
$$ LANGUAGE plpgsql;
