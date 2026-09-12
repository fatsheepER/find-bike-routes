WITH request AS MATERIALIZED (
    SELECT
        %(source_date)s::date AS source_date,
        span(
            %(start)s::timestamptz,
            %(end)s::timestamptz,
            true,
            false
        ) AS time_window
),
selection_geometry AS MATERIALIZED (
    SELECT region.geometry_analysis AS geometry
    FROM region
    WHERE %(selection_type)s = 'region'
      AND region.region_id = %(region_id)s

    UNION ALL

    SELECT ST_Transform(
        ST_MakeEnvelope(
            %(west)s,
            %(south)s,
            %(east)s,
            %(north)s,
            4326
        ),
        32650
    )
    WHERE %(selection_type)s = 'bounds'
),
windowed AS MATERIALIZED (
    SELECT
        track.track_id,
        atTime(track.trajectory, request.time_window) AS windowed_trajectory,
        selection_geometry.geometry
    FROM track
    CROSS JOIN request
    CROSS JOIN selection_geometry
    WHERE track.source_date = request.source_date
      AND track.trajectory && stbox(selection_geometry.geometry, request.time_window)
),
matched AS MATERIALIZED (
    SELECT track_id, windowed_trajectory
    FROM windowed
    WHERE windowed_trajectory IS NOT NULL
      AND eIntersects(windowed_trajectory, geometry)
),
samples AS (
    SELECT track_id, windowed_trajectory
    FROM matched
    ORDER BY track_id
    LIMIT GREATEST(LEAST(%(sample_limit)s::integer, 200), 0)
)
SELECT
    counts.total_count,
    samples.track_id,
    ST_AsGeoJSON(
        ST_Transform(trajectory(samples.windowed_trajectory), 4326)
    ) AS geometry
FROM (
    SELECT count(DISTINCT track_id) AS total_count
    FROM matched
) AS counts
LEFT JOIN samples ON true
ORDER BY samples.track_id;
