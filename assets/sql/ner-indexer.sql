


SELECT ref_id,
       pages->>'category' as "category",
       pages->'value'->'answer'->'text'->>'text' as "text",
       pages->'value'->'answer'->'text'->>'confidence' as "confidence",
       pages->'value'->'answer'->'key' as "key",
       pages->'value'->'answer'->'text' as "value",
       pages->'value'->'answer'->>'key', pages, content->'kv', *
FROM check_ner_executor,
jsonb_array_elements(content->'kv')  pages
--WHERE pages->'value'->'answer'->>'key' IN('PATIENT_NAME_ANSWER', 'PAN_ANSWER')

--WHERE  pages->'value'->'answer' @> ANY (ARRAY ['[{"key":"ANSWER"}]', '[{"key":"PAN_ANSWER"}]']::jsonb[]);


WITH PivotData AS (
    SELECT ref_id,
           pages->>'category' as "category",
           pages->'value'->'answer'->'text'->>'text' as "text",
           pages->'value'->'answer'->'text'->>'confidence' as "confidence"
    FROM check_ner_executor,
    jsonb_array_elements(content->'kv')  pages
    WHERE pages->'value'->'answer'->>'key' IN('PATIENT_NAME_ANSWER', 'PAN_ANSWER', 'MEMBER_NAME_ANSWER', 'MEMBER_NUMBER_ANSWER')
)
--SELECT * FROM PivotData
SELECT
    ref_id,
    max (CASE WHEN (category='PAN') then text end)           as PAN,
    max (CASE WHEN (category='PATIENT_NAME') then text end)  as PATIENT_NAME,
    max (CASE WHEN (category='MEMBER_NAME') then text end)   as MEMBER_NAME,
    max (CASE WHEN (category='MEMBER_NUMBER') then text end) as MEMBER_NUMBER
FROM PivotData
group by ref_id


--
-- CREATE INDEX reports_data_gin_idx ON reports
-- USING gin ((data->'objects') jsonb_path_ops);