# Marie Scheduler - Monitoring Query Guide

## Quick Start

All queries are located in: **`config/psql/schema/monitoring/example_queries.sql`**

### Prerequisites
1. ✅ Run `config/psql/schema/monitoring/slots_columns.sql` first
2. ✅ Ensure job table has slot/day generated columns
3. ✅ All timestamps are in America/Chicago timezone (configurable)

---

## Query Categories

### 📊 1. HEATMAPS (Queries 1.1 - 1.4)
**Purpose:** Visualize job patterns throughout the day/week

| Query | Description | Output |
|-------|-------------|--------|
| **1.1** | Daily Heatmap by 15-min Slot | 96 rows (every 15 min) |
| **1.2** | Weekly Heatmap (Day × Time) | 7 days × 96 slots matrix |
| **1.3** | Hourly Aggregation | 24 rows (simplified) |
| **1.4** | SLA Deadline Heatmap | When SLAs are due |

**Sample Output (Query 1.1):**
```
 slot | time_label | total_jobs | completed_jobs | failed_jobs | completion_rate_pct
------+------------+------------+----------------+-------------+--------------------
    0 | 00:00      |        145 |            142 |           3 |              97.9
    1 | 00:15      |        132 |            128 |           4 |              97.0
   48 | 12:00      |       1203 |           1189 |          14 |              98.8
   72 | 18:00      |        876 |            861 |          15 |              98.3
```

**Use Cases:**
- 📈 Identify peak creation times
- 🕐 Plan maintenance windows
- 
- 📊 Capacity planning
- 🎯 Load balancing

---

### 🔧 2. CAPACITY PLANNING (Queries 2.1 - 2.3)
**Purpose:** Resource allocation and capacity requirements

| Query | Description | Key Metrics |
|-------|-------------|-------------|
| **2.1** | Peak Load Analysis | Active + pending jobs by slot |
| **2.2** | Resource Recommendations | Recommended workers per slot |
| **2.3** | Utilization Heatmap | % utilization throughout day |

**Sample Output (Query 2.2):**
```
 time_slot | jobs_per_slot | avg_duration_sec | recommended_workers | peak_capacity_workers | health_status
-----------+---------------+------------------+--------------------+----------------------+---------------
 09:00     |           245 |            320.5 |                 87 |                  112 | HEALTHY
 14:00     |           512 |            285.2 |                162 |                  198 | HEALTHY
 22:00     |            89 |            412.8 |                 41 |                   56 | MODERATE_FAILURE
```

**Use Cases:**
- 👷 Determine worker pool size
- 📊 Identify under/over-provisioned periods
- 💰 Cost optimization
- ⚡ Scale-up/scale-down automation

---

### 🎯 3. SLA TRACKING (Queries 3.1 - 3.3)
**Purpose:** Monitor SLA compliance and identify issues

| Query | Description | Key Metrics |
|-------|-------------|-------------|
| **3.1** | SLA Compliance Rate | Met vs missed by time slot |
| **3.2** | Real-Time SLA Pressure | Current overdue/warning jobs |
| **3.3** | SLA Miss Reasons | Why SLAs failed by slot |

**Sample Output (Query 3.1):**
```
 soft_sla_slot | jobs_with_soft_sla | met_soft_sla | missed_soft_sla | soft_sla_compliance_pct
---------------+--------------------+--------------+-----------------+------------------------
 09:00         |                145 |          142 |               3 |                   97.9
 12:00         |                287 |          265 |              22 |                   92.3
 18:00         |                198 |          184 |              14 |                   92.9
```

**Sample Output (Query 3.2 - Real-Time Dashboard):**
```
 time_slot | total_jobs | hard_missed | soft_missed | hard_warning | soft_warning | on_track | avg_minutes_to_deadline
-----------+------------+-------------+-------------+--------------+--------------+----------+------------------------
 14:00     |         23 |           2 |           5 |            3 |            8 |        5 |                    -12.5
 14:15     |         18 |           0 |           2 |            4 |            6 |        6 |                      8.2
```

**Use Cases:**
- 🚨 Alert on SLA violations
- 📊 Measure service quality
- 🔍 Root cause analysis
- 📈 Trend analysis

---

### 📈 4. LOAD DISTRIBUTION (Queries 4.1 - 4.4)
**Purpose:** Understand traffic patterns and peak periods

| Query | Description | Analysis Type |
|-------|-------------|---------------|
| **4.1** | Top 10 Peak Load Slots | Ranked busiest periods |
| **4.2** | Business vs Off-Hours | Workload by time category |
| **4.3** | 24-Hour Load Profile | Complete daily breakdown |
| **4.4** | Weekend vs Weekday | Pattern comparison |

**Sample Output (Query 4.1):**
```
 time_slot | total_jobs | active_days | avg_jobs_per_day | avg_priority | failed_jobs | failure_rate_pct | load_rank
-----------+------------+-------------+------------------+--------------+-------------+------------------+-----------
 13:45     |       2847 |          30 |            94.90 |        125.3 |          45 |             1.58 |         1
 09:15     |       2634 |          30 |            87.80 |        118.7 |          38 |             1.44 |         2
 14:30     |       2512 |          30 |            83.73 |        122.1 |          52 |             2.07 |         3
```

**Sample Output (Query 4.2 - Business Hours Analysis):**
```
 time_category                 | total_jobs | pct_of_total | completed | failed | success_rate_pct
-------------------------------+------------+--------------+-----------+--------+-----------------
 Business Hours (9AM-5PM)      |      18542 |        68.45 |     18234 |    308 |            98.34
 Evening (6PM-10PM)            |       5123 |        18.91 |      5002 |    121 |            97.64
 Early Morning (6AM-9AM)       |       2341 |         8.64 |      2298 |     43 |            98.16
 Night (10PM-6AM)              |       1089 |         4.02 |      1063 |     26 |            97.61
```

**Use Cases:**
- 📊 Traffic pattern analysis
- 🕐 Maintenance scheduling
- 👥 Staffing decisions
- 📅 Capacity forecasting

---

### 🎨 5. ADVANCED ANALYTICS (Queries 5.1 - 5.2)
**Purpose:** Combined insights and trend analysis

| Query | Description | Use Case |
|-------|-------------|----------|
| **5.1** | Comprehensive Dashboard | Single query for real-time monitoring |
| **5.2** | Week-over-Week Trends | Compare current to previous week |

**Sample Output (Query 5.1 - Dashboard View):**
```
 time_slot | total_jobs | completed | active | pending | failed | avg_duration_sec | soft_overdue | hard_overdue | success_rate_pct
-----------+------------+-----------+--------+---------+--------+------------------+--------------+--------------+-----------------
 14:00     |         87 |        65 |     12 |       8 |      2 |            285.3 |            3 |            1 |            74.7
 14:15     |         92 |        71 |     15 |       5 |      1 |            312.8 |            2 |            0 |            77.2
```

**Sample Output (Query 5.2 - Trend Analysis):**
```
 time_slot | current_week_jobs | previous_week_jobs | jobs_change | jobs_change_pct | trend
-----------+-------------------+--------------------+-------------+-----------------+-------------
 09:00     |               512 |                487 |          25 |            5.13 | ➡️  STABLE
 14:00     |               689 |                542 |         147 |           27.12 | 📈 INCREASING
 22:00     |               234 |                412 |        -178 |          -43.20 | 📉 DECREASING
```

---

### 📤 6. EXPORT FORMATS (Queries 6.1 - 6.2)
**Purpose:** Integration with visualization tools

| Query | Format | Use Case |
|-------|--------|----------|
| **6.1** | JSON | Chart.js, D3.js, Grafana |
| **6.2** | CSV | Excel, Google Sheets, Tableau |

**Sample Output (Query 6.1 - JSON):**
```json
[
  {
    "slot": 0,
    "hour": 0,
    "minute": 0,
    "label": "00:00",
    "jobs": 145,
    "completed": 142,
    "failed": 3,
    "success_rate": 97.93
  },
  {
    "slot": 48,
    "hour": 12,
    "minute": 0,
    "label": "12:00",
    "jobs": 1203,
    "completed": 1189,
    "failed": 14,
    "success_rate": 98.84
  }
]
```

---

## Common Use Cases

### 🚀 Quick Start: Top 5 Essential Queries

1. **Current Day Dashboard** → Query 5.1
2. **Peak Load Times** → Query 4.1
3. **SLA Compliance** → Query 3.1
4. **Resource Requirements** → Query 2.2
5. **Hourly Heatmap** → Query 1.3

---

## How to Run Queries

### Method 1: psql Command Line
```bash
# Connect to database
psql -h localhost -U your_user -d your_database

# Run specific query
\i config/psql/schema/monitoring/example_queries.sql

# Or copy-paste individual queries
```

### Method 2: Python Integration
```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="your_database",
    user="your_user",
    password="your_password",
)

# Read query file
with open('config/psql/schema/monitoring/example_queries.sql', 'r') as f:
    query = f.read()

# Extract and run specific query (e.g., Query 1.1)
# Parse the file to get individual queries...
```

### Method 3: DBeaver / pgAdmin
1. Open SQL editor
2. Load `example_queries.sql`
3. Select query section
4. Execute (F5 or Ctrl+Enter)

---

## Customization Tips

### 📅 Adjust Time Ranges
```sql
-- Change from 7 days to 30 days
WHERE day_local_created >= CURRENT_DATE - INTERVAL '30 days'

-- Today only
WHERE day_local_created = CURRENT_DATE

-- Specific date range
WHERE day_local_created BETWEEN '2025-01-01' AND '2025-01-31'
```

### 🕐 Change Time Grouping
```sql
-- 15-minute slots (default)
GROUP BY slot_idx15_created

-- Hourly buckets
GROUP BY (slot_idx15_created / 4)

-- 30-minute buckets
GROUP BY (slot_idx15_created / 2)
```

### 🌍 Filter by Job Name
```sql
-- Add to WHERE clause
AND name = 'document_processing'

-- Pattern matching
AND name LIKE 'doc_%'
```

### 📊 Add Custom Metrics
```sql
-- Add percentiles
PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM duration)) AS p50_duration,
PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM duration)) AS p95_duration,
PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM duration)) AS p99_duration
```

---

## Visualization Examples

### 📊 Using Query Results in Grafana

**Query 1.3 (Hourly Aggregation) → Grafana Time Series:**
```sql
-- Add timestamp for Grafana
SELECT
    CURRENT_DATE + ((slot_idx15_created / 4) * interval '1 hour') AS time,
    COUNT(*) AS jobs
FROM marie_scheduler.job
WHERE day_local_created >= CURRENT_DATE - INTERVAL '7 days'
  AND slot_idx15_created IS NOT NULL
GROUP BY slot_idx15_created
ORDER BY slot_idx15_created;
```

### 📈 Chart.js Integration

**Query 6.1 (JSON Export) → Chart.js:**
```javascript
fetch('/api/heatmap-data')
  .then(response => response.json())
  .then(data => {
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.map(d => d.label),
        datasets: [{
          label: 'Jobs per Time Slot',
          data: data.map(d => d.jobs),
          backgroundColor: data.map(d =>
            d.success_rate > 95 ? 'green' : 'orange'
          )
        }]
      }
    });
  });
```

---

## Performance Notes

### ✅ Fast Queries (< 100ms)
- All queries using slot_idx15_* and day_local_* columns
- Index on (day_local_effective, slot_idx15_effective) used automatically
- STORED generated columns = no runtime computation

### ⚠️ Slower Operations
- Full table scans without date filters
- Large date ranges (> 90 days)
- Queries without slot/day columns

### 🚀 Optimization Tips
1. **Always include date filters**
2. **Use indexed columns (day_local_*, slot_idx15_*)**
3. **Limit large aggregations with date ranges**
4. **Materialize frequently-used results**

---

## Troubleshooting

### ❌ Error: Column "slot_idx15_created" does not exist
**Solution:** Run `slots_columns.sql` first to create generated columns

### ❌ Error: Index not found
**Solution:** Recreate index:
```sql
CREATE INDEX IF NOT EXISTS job_day_slot_idx
ON marie_scheduler.job (day_local_effective, slot_idx15_effective);
```

### ❌ Empty Results
**Check:**
1. Date range includes actual data
2. Timezone is correct for your data
3. Job table has data in specified range

---

## Next Steps

1. ✅ Run sample queries to verify setup
2. 📊 Integrate with your monitoring dashboard
3. 🎯 Customize date ranges and filters
4. 📈 Set up scheduled reports
5. 🚨 Create alerts based on thresholds

---

## Additional Resources

- **Schema Documentation:** `SLOTS_COLUMNS_ANALYSIS.md`
- **Timezone Fixes:** `TIMEZONE_FIXES_SUMMARY.md`
- **INSERT/UPDATE Verification:** `INSERT_UPDATE_VERIFICATION.md`

---

## Support

For issues or questions:
1. Check timezone configuration in `slots_columns.sql`
2. Verify generated columns exist: `\d marie_scheduler.job`
3. Review query execution plans: `EXPLAIN ANALYZE <query>`
