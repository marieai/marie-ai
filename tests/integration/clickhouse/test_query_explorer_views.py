from __future__ import annotations

from conftest import QueryFixture


def test_materialized_columns_pricing_and_current_annotations(clickhouse: QueryFixture) -> None:
    seed = clickhouse.query("""
INSERT INTO otel.model_prices (id,name,match_pattern,input_price,output_price,input_breakdown,output_breakdown,version)
VALUES ('00000000-0000-4000-8000-000000000001','fixture','^fixture-model$',2,3,{'cache_read':1},{},1);
INSERT INTO otel.otel_traces (Timestamp,TraceId,SpanId,ParentSpanId,SpanName,Duration,SpanAttributes)
VALUES ('2026-10-14 12:00:00.123456789','priced','root','','fixture',1000000,
{'openinference.span.kind':'LLM','llm.model_name':'fixture-model','llm.token_count.prompt':'1000',
'llm.token_count.completion':'100','llm.token_count.prompt_details.cache_read':'200'}),
('2026-10-14 12:00:01','recorded','r','','recorded',2000000,{'llm.cost.total':'1.5'}),
('2026-10-14 12:00:02','unknown','u','','unknown',0,{});
INSERT INTO otel.span_annotations (id,trace_id,span_id,project_id,name,updated_at)
VALUES ('00000000-0000-4000-8000-000000000002','priced','','org','rating','2026-10-14 12:00:00'),
('00000000-0000-4000-8000-000000000003','priced','root','org','rating','2026-10-14 12:00:01');
SYSTEM RELOAD DICTIONARY otel.model_price_dict;
""")
    assert seed.returncode == 0, seed.stderr
    result = clickhouse.query('SELECT TraceId,oi_span_kind,round(cost_total,4),latency_ms FROM query.spans ORDER BY TraceId', reader=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['priced\tLLM\t0.0021\t1', 'recorded\t\t1.5\t2', 'unknown\t\t\\N\t0']
    annotations = clickhouse.query('SELECT TraceId,SpanId,marie_project_id FROM query.annotations ORDER BY SpanId', reader=True)
    assert annotations.returncode == 0, annotations.stderr
    assert annotations.stdout.splitlines() == ['priced\t\torg', 'priced\troot\torg']
    total = clickhouse.query('SELECT round(sum(cost_total),4) FROM query.spans WHERE TraceId IN (SELECT TraceId FROM query.annotations)', reader=True)
    assert total.returncode == 0, total.stderr
    assert total.stdout.strip() == '0.0021'


def test_current_prices_filter_deleted_versions(clickhouse: QueryFixture) -> None:
    result = clickhouse.query("""
INSERT INTO otel.model_prices (id,name,match_pattern,input_price,output_price,input_breakdown,output_breakdown,version,is_deleted)
VALUES ('00000000-0000-4000-8000-000000000004','deleted','deleted',1,1,{},{},1,0),
('00000000-0000-4000-8000-000000000004','deleted','deleted',1,1,{},{},2,1);
""")
    assert result.returncode == 0, result.stderr
    current = clickhouse.query("SELECT count() FROM query.model_prices WHERE name='deleted'", reader=True)
    assert current.returncode == 0, current.stderr
    assert current.stdout.strip() == '0'
