WITH request AS MATERIALIZED (
    SELECT
        %(matrix)s::text AS matrix,
        %(source_date)s::date AS source_date,
        %(hour)s::integer AS hour,
        ARRAY[
            DATE '2020-12-21',
            DATE '2020-12-22',
            DATE '2020-12-24',
            DATE '2020-12-25'
        ] AS clear_dates
),
significance AS MATERIALIZED (
    SELECT significance.*
    FROM flow_significance AS significance
    CROSS JOIN request
    WHERE significance.matrix = request.matrix
      AND significance.scope = CASE
          WHEN request.source_date IS NULL THEN 'clear-days-stable'
          ELSE to_char(request.source_date, 'YYYY-MM-DD')
      END
),
observed AS MATERIALIZED (
    SELECT
        flow.from_region,
        flow.to_region,
        sum(flow.trips)::double precision AS weight
    FROM flow_od AS flow
    CROSS JOIN request
    WHERE request.matrix = 'flow_od'
      AND flow.hour = request.hour
      AND (
          flow.source_date = request.source_date
          OR request.source_date IS NULL
             AND flow.source_date = ANY(request.clear_dates)
      )
    GROUP BY flow.from_region, flow.to_region

    UNION ALL

    SELECT
        flow.from_region,
        flow.to_region,
        sum(flow.tracks)::double precision AS weight
    FROM flow_channel AS flow
    CROSS JOIN request
    WHERE request.matrix = 'flow_channel'
      AND flow.hour = request.hour
      AND (
          flow.source_date = request.source_date
          OR request.source_date IS NULL
             AND flow.source_date = ANY(request.clear_dates)
      )
    GROUP BY flow.from_region, flow.to_region
),
pairs AS (
    SELECT from_region, to_region
    FROM significance

    UNION

    SELECT observed.from_region, observed.to_region
    FROM observed
    CROSS JOIN request
    WHERE request.source_date IS NOT NULL
)
SELECT
    request.matrix,
    CASE
        WHEN request.source_date IS NULL THEN 'clear-days-stable'
        ELSE to_char(request.source_date, 'YYYY-MM-DD')
    END AS scope,
    request.hour,
    pairs.from_region,
    pairs.to_region,
    coalesce(observed.weight, 0) / CASE
        WHEN request.source_date IS NULL THEN 4.0
        ELSE 1.0
    END AS weight,
    significance.matrix IS NOT NULL AS is_tested,
    significance.observed,
    significance.is_significant,
    significance.gated,
    significance.is_self_loop
FROM pairs
CROSS JOIN request
LEFT JOIN observed USING (from_region, to_region)
LEFT JOIN significance USING (from_region, to_region)
ORDER BY pairs.from_region, pairs.to_region;
