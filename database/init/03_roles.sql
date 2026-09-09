DO $roles$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'bike_routes_import') THEN
        CREATE ROLE bike_routes_import LOGIN;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'bike_routes_api') THEN
        CREATE ROLE bike_routes_api LOGIN;
    END IF;
END
$roles$;

REVOKE ALL ON TABLE
    dataset_release, district, region, grid_cell, track, region_metric,
    flow_od, flow_channel, flow_significance
FROM PUBLIC;

GRANT USAGE ON SCHEMA public TO bike_routes_import, bike_routes_api;

GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON TABLE
    dataset_release, district, region, grid_cell, track, region_metric,
    flow_od, flow_channel, flow_significance
TO bike_routes_import;

GRANT SELECT ON TABLE
    dataset_release, district, region, grid_cell, track, region_metric,
    flow_od, flow_channel, flow_significance
TO bike_routes_api;
