-- ============================================================
-- Customer Churn Analysis - SQL Data Preparation
-- Author: Arundhathi Reddy
-- GitHub: https://github.com/Arundhathi5
-- ============================================================


-- ── 1. RAW DATA OVERVIEW ─────────────────────────────────────

SELECT
    COUNT(*)                                          AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)   AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                 AS churn_rate_pct
FROM customers;


-- ── 2. CLEAN & CAST COLUMNS ──────────────────────────────────

CREATE OR REPLACE VIEW customers_clean AS
SELECT
    customer_id,
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    CAST(monthly_charges AS DECIMAL(10,2))   AS monthly_charges,
    CAST(
        NULLIF(TRIM(total_charges), '')       AS DECIMAL(10,2)
    )                                         AS total_charges,
    CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END AS churn_flag
FROM customers
WHERE TRIM(total_charges) <> ''          -- remove rows with blank TotalCharges
  AND tenure > 0;


-- ── 3. CHURN BY CONTRACT TYPE ────────────────────────────────

SELECT
    contract,
    COUNT(*)                                          AS total,
    SUM(churn_flag)                                   AS churned,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 2)      AS churn_rate_pct
FROM customers_clean
GROUP BY contract
ORDER BY churn_rate_pct DESC;


-- ── 4. CHURN BY TENURE BUCKET ────────────────────────────────

SELECT
    CASE
        WHEN tenure BETWEEN 0  AND 12 THEN '0-12 months'
        WHEN tenure BETWEEN 13 AND 24 THEN '13-24 months'
        WHEN tenure BETWEEN 25 AND 48 THEN '25-48 months'
        ELSE '49+ months'
    END                                               AS tenure_bucket,
    COUNT(*)                                          AS total,
    SUM(churn_flag)                                   AS churned,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 2)      AS churn_rate_pct
FROM customers_clean
GROUP BY 1
ORDER BY churn_rate_pct DESC;


-- ── 5. CHURN BY MONTHLY CHARGE SEGMENT ───────────────────────

SELECT
    CASE
        WHEN monthly_charges < 35  THEN 'Low (<$35)'
        WHEN monthly_charges < 65  THEN 'Medium ($35–$65)'
        WHEN monthly_charges < 90  THEN 'High ($65–$90)'
        ELSE 'Very High (>$90)'
    END                                               AS charge_segment,
    COUNT(*)                                          AS total,
    SUM(churn_flag)                                   AS churned,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 2)      AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)                    AS avg_monthly_charge
FROM customers_clean
GROUP BY 1
ORDER BY churn_rate_pct DESC;


-- ── 6. AVERAGE TENURE & CHARGES BY CHURN STATUS ──────────────

SELECT
    CASE WHEN churn_flag = 1 THEN 'Churned' ELSE 'Retained' END AS churn_status,
    ROUND(AVG(tenure), 1)          AS avg_tenure_months,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charge,
    ROUND(AVG(total_charges), 2)   AS avg_total_charge,
    COUNT(*)                       AS customer_count
FROM customers_clean
GROUP BY churn_flag;


-- ── 7. INTERNET SERVICE + SECURITY vs CHURN ──────────────────

SELECT
    internet_service,
    online_security,
    COUNT(*)                                          AS total,
    SUM(churn_flag)                                   AS churned,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 2)      AS churn_rate_pct
FROM customers_clean
GROUP BY internet_service, online_security
ORDER BY churn_rate_pct DESC;


-- ── 8. PAYMENT METHOD vs CHURN ───────────────────────────────

SELECT
    payment_method,
    COUNT(*)                                          AS total,
    SUM(churn_flag)                                   AS churned,
    ROUND(100.0 * SUM(churn_flag) / COUNT(*), 2)      AS churn_rate_pct
FROM customers_clean
GROUP BY payment_method
ORDER BY churn_rate_pct DESC;


-- ── 9. FINAL ANALYTICAL DATASET FOR PYTHON / TABLEAU ─────────
-- Export this query result to CSV for use in ML pipeline

SELECT
    customer_id,
    gender,
    senior_citizen,
    tenure,
    contract,
    internet_service,
    online_security,
    monthly_charges,
    total_charges,
    ROUND(total_charges / NULLIF(tenure, 0), 2)   AS avg_charge_per_month,
    paperless_billing,
    payment_method,
    churn_flag
FROM customers_clean
ORDER BY customer_id;
