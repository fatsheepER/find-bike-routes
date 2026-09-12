CREATE TABLE dataset_release (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    schema_version text NOT NULL CHECK (btrim(schema_version) <> ''),
    release_digest text NOT NULL CHECK (release_digest ~ '^(sha256:)?[0-9a-f]{64}$'),
    upstream_digests jsonb NOT NULL CHECK (jsonb_typeof(upstream_digests) = 'object'),
    region_cells_digest text NOT NULL CHECK (region_cells_digest ~ '^(sha256:)?[0-9a-f]{64}$'),
    imported_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE district (
    district_id integer PRIMARY KEY CHECK (district_id > 0),
    label text NOT NULL CHECK (btrim(label) <> ''),
    regions integer NOT NULL CHECK (regions > 0),
    cells integer NOT NULL CHECK (cells > 0),
    area_km2 double precision NOT NULL CHECK (area_km2 > 0),
    geometry geometry(Geometry, 32650) NOT NULL,
    label_candidate text NOT NULL,
    label_second text NOT NULL,
    CHECK (GeometryType(geometry) IN ('POLYGON', 'MULTIPOLYGON'))
);

CREATE TABLE region (
    region_id integer PRIMARY KEY CHECK (region_id > 0),
    district_id integer NOT NULL REFERENCES district (district_id),
    region_code text NOT NULL UNIQUE CHECK (btrim(region_code) <> ''),
    cells integer NOT NULL CHECK (cells > 0),
    area_km2 double precision NOT NULL CHECK (area_km2 > 0),
    geometry_analysis geometry(Polygon, 32650) NOT NULL,
    geometry_display geometry(Geometry, 32650) NOT NULL,
    label_candidate text NOT NULL,
    label_second text NOT NULL,
    area_residential_m2 double precision NOT NULL CHECK (area_residential_m2 >= 0),
    area_employment_m2 double precision NOT NULL CHECK (area_employment_m2 >= 0),
    area_education_m2 double precision NOT NULL CHECK (area_education_m2 >= 0),
    area_transport_m2 double precision NOT NULL CHECK (area_transport_m2 >= 0),
    classified_area_m2 double precision NOT NULL CHECK (classified_area_m2 >= 0),
    share_residential double precision CHECK (share_residential BETWEEN 0 AND 1),
    share_employment double precision CHECK (share_employment BETWEEN 0 AND 1),
    share_education double precision CHECK (share_education BETWEEN 0 AND 1),
    share_transport double precision CHECK (share_transport BETWEEN 0 AND 1),
    classified_share double precision NOT NULL CHECK (classified_share BETWEEN 0 AND 1),
    poi_residential integer NOT NULL CHECK (poi_residential >= 0),
    poi_employment integer NOT NULL CHECK (poi_employment >= 0),
    poi_education integer NOT NULL CHECK (poi_education >= 0),
    poi_transport integer NOT NULL CHECK (poi_transport >= 0),
    poi_total integer NOT NULL CHECK (poi_total >= 0),
    bus_stops integer NOT NULL CHECK (bus_stops >= 0),
    bus_stops_per_km2 double precision NOT NULL CHECK (bus_stops_per_km2 >= 0),
    CHECK (GeometryType(geometry_display) IN ('POLYGON', 'MULTIPOLYGON'))
);

CREATE TABLE grid_cell (
    cell_x integer NOT NULL,
    cell_y integer NOT NULL,
    region_id integer NOT NULL REFERENCES region (region_id),
    geometry geometry(Polygon, 32650) NOT NULL,
    PRIMARY KEY (cell_x, cell_y),
    CHECK (ST_Area(geometry) = 22500)
);

CREATE TABLE track (
    track_id text PRIMARY KEY CHECK (btrim(track_id) <> ''),
    bicycle_id text NOT NULL CHECK (btrim(bicycle_id) <> ''),
    source_date date NOT NULL CHECK (source_date IN (
        DATE '2020-12-21', DATE '2020-12-22', DATE '2020-12-23',
        DATE '2020-12-24', DATE '2020-12-25'
    )),
    start_time timestamptz NOT NULL,
    end_time timestamptz NOT NULL,
    duration_s bigint NOT NULL CHECK (duration_s >= 0),
    points bigint NOT NULL CHECK (points > 0),
    range_m double precision NOT NULL CHECK (range_m >= 0),
    slow_point_share double precision NOT NULL CHECK (slow_point_share BETWEEN 0 AND 1),
    mean_speed_mps double precision NOT NULL CHECK (mean_speed_mps >= 0),
    match_rate double precision NOT NULL CHECK (match_rate BETWEEN 0 AND 1),
    matched_points bigint NOT NULL CHECK (matched_points > 0 AND matched_points <= points),
    matched_length_m double precision NOT NULL CHECK (matched_length_m >= 0),
    observed_length_m double precision NOT NULL CHECK (observed_length_m >= 0),
    inferred_length_m double precision NOT NULL CHECK (inferred_length_m >= 0),
    inferred_share double precision NOT NULL CHECK (inferred_share BETWEEN 0 AND 1),
    path_breaks bigint NOT NULL CHECK (path_breaks >= 0),
    pieces bigint NOT NULL CHECK (pieces > 0),
    contraflow_points bigint NOT NULL CHECK (contraflow_points >= 0),
    trajectory tgeompoint(SequenceSet, Point, 32650) NOT NULL,
    CHECK (end_time >= start_time)
);

CREATE INDEX track_trajectory_gist ON track USING gist (trajectory);
CREATE INDEX track_source_date_btree ON track USING btree (source_date);

CREATE TABLE region_metric (
    source_date date NOT NULL CHECK (source_date IN (
        DATE '2020-12-21', DATE '2020-12-22', DATE '2020-12-23',
        DATE '2020-12-24', DATE '2020-12-25'
    )),
    hour integer NOT NULL CHECK (hour IN (6, 7, 8, 9)),
    region_id integer NOT NULL REFERENCES region (region_id),
    unlocks bigint NOT NULL CHECK (unlocks >= 0),
    locks bigint NOT NULL CHECK (locks >= 0),
    net_inflow bigint NOT NULL,
    net_inflow_per_km2 double precision NOT NULL,
    order_events_per_km2 double precision NOT NULL CHECK (order_events_per_km2 >= 0),
    tracks_visiting bigint NOT NULL CHECK (tracks_visiting >= 0),
    tracks_transit bigint NOT NULL CHECK (tracks_transit >= 0),
    pi_r double precision CHECK (pi_r BETWEEN 0 AND 1),
    chords bigint NOT NULL CHECK (chords >= 0),
    sum_cos double precision NOT NULL,
    sum_sin double precision NOT NULL,
    sum_cos2 double precision NOT NULL,
    sum_sin2 double precision NOT NULL,
    r double precision CHECK (r BETWEEN 0 AND 1),
    r_axial double precision CHECK (r_axial BETWEEN 0 AND 1),
    mean_bearing_deg double precision CHECK (mean_bearing_deg >= 0 AND mean_bearing_deg < 360),
    axis_bearing_deg double precision CHECK (axis_bearing_deg >= 0 AND axis_bearing_deg < 180),
    sector_00 bigint NOT NULL CHECK (sector_00 >= 0),
    sector_01 bigint NOT NULL CHECK (sector_01 >= 0),
    sector_02 bigint NOT NULL CHECK (sector_02 >= 0),
    sector_03 bigint NOT NULL CHECK (sector_03 >= 0),
    sector_04 bigint NOT NULL CHECK (sector_04 >= 0),
    sector_05 bigint NOT NULL CHECK (sector_05 >= 0),
    sector_06 bigint NOT NULL CHECK (sector_06 >= 0),
    sector_07 bigint NOT NULL CHECK (sector_07 >= 0),
    sector_08 bigint NOT NULL CHECK (sector_08 >= 0),
    sector_09 bigint NOT NULL CHECK (sector_09 >= 0),
    sector_10 bigint NOT NULL CHECK (sector_10 >= 0),
    sector_11 bigint NOT NULL CHECK (sector_11 >= 0),
    sector_12 bigint NOT NULL CHECK (sector_12 >= 0),
    sector_13 bigint NOT NULL CHECK (sector_13 >= 0),
    sector_14 bigint NOT NULL CHECK (sector_14 >= 0),
    sector_15 bigint NOT NULL CHECK (sector_15 >= 0),
    PRIMARY KEY (source_date, hour, region_id),
    CHECK (net_inflow = locks - unlocks),
    CHECK (tracks_transit <= tracks_visiting),
    CHECK (chords <= tracks_transit),
    CHECK (chords = sector_00 + sector_01 + sector_02 + sector_03 +
        sector_04 + sector_05 + sector_06 + sector_07 + sector_08 + sector_09 +
        sector_10 + sector_11 + sector_12 + sector_13 + sector_14 + sector_15)
);

CREATE TABLE flow_od (
    source_date date NOT NULL CHECK (source_date IN (
        DATE '2020-12-21', DATE '2020-12-22', DATE '2020-12-23',
        DATE '2020-12-24', DATE '2020-12-25'
    )),
    hour integer NOT NULL CHECK (hour IN (6, 7, 8, 9)),
    from_region integer NOT NULL REFERENCES region (region_id),
    to_region integer NOT NULL REFERENCES region (region_id),
    distance_band text NOT NULL CHECK (distance_band IN ('< 1 km', '1–3 km', '≥ 3 km')),
    trips bigint NOT NULL CHECK (trips > 0),
    PRIMARY KEY (source_date, hour, from_region, to_region, distance_band)
);

CREATE TABLE flow_channel (
    source_date date NOT NULL CHECK (source_date IN (
        DATE '2020-12-21', DATE '2020-12-22', DATE '2020-12-23',
        DATE '2020-12-24', DATE '2020-12-25'
    )),
    hour integer NOT NULL CHECK (hour IN (6, 7, 8, 9)),
    from_region integer NOT NULL REFERENCES region (region_id),
    to_region integer NOT NULL REFERENCES region (region_id),
    tracks bigint NOT NULL CHECK (tracks > 0),
    PRIMARY KEY (source_date, hour, from_region, to_region),
    CHECK (from_region <> to_region)
);

CREATE TABLE flow_significance (
    matrix text NOT NULL CHECK (matrix IN ('flow_od', 'flow_channel')),
    scope text NOT NULL CHECK (scope IN (
        '2020-12-21', '2020-12-22', '2020-12-23',
        '2020-12-24', '2020-12-25', 'clear-days-stable'
    )),
    from_region integer NOT NULL REFERENCES region (region_id),
    to_region integer NOT NULL REFERENCES region (region_id),
    observed bigint NOT NULL CHECK (observed >= 0),
    null_mean double precision CHECK (null_mean >= 0),
    null_sd double precision CHECK (null_sd >= 0),
    z double precision,
    p_normal double precision CHECK (p_normal BETWEEN 0 AND 1),
    p_empirical double precision CHECK (p_empirical BETWEEN 0 AND 1),
    q double precision CHECK (q BETWEEN 0 AND 1),
    is_significant boolean NOT NULL,
    gated boolean NOT NULL,
    is_self_loop boolean NOT NULL,
    null_model text NOT NULL CHECK (btrim(null_model) <> ''),
    reps integer NOT NULL CHECK (reps > 0),
    z_min double precision,
    z_median double precision,
    days_significant integer CHECK (days_significant BETWEEN 0 AND 4),
    PRIMARY KEY (matrix, scope, from_region, to_region),
    CHECK (is_self_loop = (from_region = to_region))
);
