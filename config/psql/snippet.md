```sql
   CASE
      WHEN NOW() > hard_sla THEN 'missed_hard'
      WHEN NOW() > soft_sla THEN 'missed_soft'
      WHEN soft_sla IS NOT NULL THEN 'on_time'
      ELSE 'no_sla'
    END AS sla_tier
```