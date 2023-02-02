

--DELETE FROM shared_docs WHERE 1=1

SELECT * FROM shared_docs

SELECT doc_id, ref_id,
      CAST(  tags->>'index' AS INTEGER)  page_index,
      blob
FROM shared_docs -- jsonb_array_elements(tags->'kv')  tags
WHERE ref_type = 'overlay' AND tags ->> 'type' = 'blended'
ORDER BY 1 DESC