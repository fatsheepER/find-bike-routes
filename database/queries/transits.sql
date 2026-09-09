WITH request AS MATERIALIZED (
    SELECT
        ST_Transform(
            ST_MakeEnvelope(
                %(west)s,
                %(south)s,
                %(east)s,
                %(north)s,
                4326
            ),
            32650
        ) AS bounds,
        tstzspan(
            %(start_local)s::timestamp AT TIME ZONE 'Asia/Shanghai',
            %(end_local)s::timestamp AT TIME ZONE 'Asia/Shanghai',
            '[)'
        ) AS time_window
),
windowed AS MATERIALIZED (
    SELECT
        track.track_id,
        atTime(track.trajectory, request.time_window) AS windowed_trajectory,
        request.bounds
    FROM track
    CROSS JOIN request
),
matched AS MATERIALIZED (
    SELECT track_id, windowed_trajectory
    FROM windowed
    WHERE windowed_trajectory IS NOT NULL
      AND eIntersects(windowed_trajectory, bounds)
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
    trajectory(samples.windowed_trajectory) AS geometry_32650
FROM (
    SELECT count(DISTINCT track_id) AS total_count
    FROM matched
) AS counts
LEFT JOIN samples ON true
ORDER BY samples.track_id;
