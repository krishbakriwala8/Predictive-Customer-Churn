-- Example DuckDB/SQL snippet reproducing RFM-style features (replace <path_to_table> as needed)
-- This assumes you've exported filtered Germany transactions into a CSV or table.

-- 1) Basic aggregation per customer
SELECT
  CustomerID,
  MIN(InvoiceDate) AS first_purchase,
  MAX(InvoiceDate) AS last_purchase,
  COUNT(DISTINCT Invoice) AS num_invoices,
  SUM(Quantity) AS total_quantity,
  SUM(Quantity * UnitPrice) AS monetary,
  COUNT(DISTINCT StockCode) AS distinct_items
FROM germany_transactions
GROUP BY CustomerID;

-- 2) Recency relative to observation_end (pass observation_end as a parameter)
SELECT
  CustomerID,
  DATE_DIFF('day', MAX(InvoiceDate), DATE '2020-12-31') AS recency_days -- example observation_end date
FROM germany_transactions
GROUP BY CustomerID;
