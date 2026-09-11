export const CLEAR_DAYS = ["2020-12-21", "2020-12-22", "2020-12-24", "2020-12-25"] as const

export const ALL_DAYS = ["2020-12-21", "2020-12-22", "2020-12-23", "2020-12-24", "2020-12-25"] as const
export type DateScope = "clear-days" | (typeof ALL_DAYS)[number] | `${(typeof ALL_DAYS)[number]}..${(typeof ALL_DAYS)[number]}`
export type Aggregation = "average" | "sum"

export type RegionMetricValues = {
  unlocks: number | null
  locks: number | null
  net_inflow: number | null
  net_inflow_per_km2: number | null
  order_events_per_km2: number | null
  tracks_visiting: number | null
  tracks_transit: number | null
  pi_r: number | null
  chords: number | null
  sum_cos: number | null
  sum_sin: number | null
  sum_cos2: number | null
  sum_sin2: number | null
  r: number | null
  r_axial: number | null
  mean_bearing_deg: number | null
  axis_bearing_deg: number | null
}

export type RegionMetric = RegionMetricValues & {
  date: string
  hour: number
  sectors: Array<number | null>
}

export type RegionFeature = {
  type: "Feature"
  id: number
  geometry: GeoJSON.Geometry
  properties: {
    region_id: number
    region_code: string
    district_id: number
    cells: number
    area_km2: number
    functional_composition: {
      residential: number
      employment: number
      education: number
      transport: number
    }
    classified_share: number
    bus_stops_per_km2: number
    map_anchor?: GeoJSON.Geometry
    metrics: RegionMetric[]
  }
}

export type RegionSelection = {
  dateScope: DateScope
  startHour: number
  endHour: number
  aggregation: Aggregation
}

export type AggregatedRegion = Omit<RegionFeature["properties"], "metrics"> & RegionMetricValues & {
  available: boolean
  sectors: number[] | null
}

export function datesForScope(scope: DateScope): readonly string[] {
  if (scope === "clear-days") return CLEAR_DAYS
  if (!scope.includes("..")) return [scope]
  const [start, end] = scope.split("..")
  return ALL_DAYS.filter((date) => date >= start && date <= end)
}

export function aggregateRegion(feature: RegionFeature, selection: RegionSelection): AggregatedRegion {
  const dates = datesForScope(selection.dateScope)
  const selectedDates = new Set<string>(dates)
  const rows = feature.properties.metrics.filter(
    (row) => selectedDates.has(row.date) && row.hour >= selection.startHour && row.hour < selection.endHour,
  )
  const expectedRows = dates.length * (selection.endHour - selection.startHour)
  const rowKeys = new Set(rows.map((row) => `${row.date}/${row.hour}`))
  const additiveFields = [
    "unlocks",
    "locks",
    "net_inflow",
    "tracks_visiting",
    "tracks_transit",
    "chords",
    "sum_cos",
    "sum_sin",
    "sum_cos2",
    "sum_sin2",
  ] as const
  const available =
    rows.length === expectedRows &&
    rowKeys.size === expectedRows &&
    rows.every((row) => additiveFields.every((field) => typeof row[field] === "number")) &&
    rows.every((row) => row.sectors.length === 16 && row.sectors.every((value) => typeof value === "number")) &&
    feature.properties.area_km2 > 0
  const divisor = selection.aggregation === "average" ? dates.length : 1
  const rawSum = (field: (typeof additiveFields)[number]) =>
    available ? rows.reduce((total, row) => total + (row[field] as number), 0) : null
  const scaledSum = (field: (typeof additiveFields)[number]) => {
    const value = rawSum(field)
    return value === null ? null : value / divisor
  }
  const unlocks = scaledSum("unlocks")
  const locks = scaledSum("locks")
  const netInflow = scaledSum("net_inflow")
  const tracksVisiting = rawSum("tracks_visiting")
  const tracksTransit = rawSum("tracks_transit")
  const chords = rawSum("chords")
  const sumCos = rawSum("sum_cos")
  const sumSin = rawSum("sum_sin")
  const sumCos2 = rawSum("sum_cos2")
  const sumSin2 = rawSum("sum_sin2")
  const directedLength = sumCos === null || sumSin === null ? null : Math.hypot(sumCos, sumSin)
  const axialLength = sumCos2 === null || sumSin2 === null ? null : Math.hypot(sumCos2, sumSin2)
  const normalize = (degrees: number, period: number) => ((degrees % period) + period) % period
  const zeroEpsilon = 1e-12

  return {
    ...feature.properties,
    available,
    unlocks,
    locks,
    net_inflow: netInflow,
    net_inflow_per_km2: netInflow === null ? null : netInflow / feature.properties.area_km2,
    order_events_per_km2:
      unlocks === null || locks === null ? null : (unlocks + locks) / feature.properties.area_km2,
    tracks_visiting: tracksVisiting === null ? null : tracksVisiting / divisor,
    tracks_transit: tracksTransit === null ? null : tracksTransit / divisor,
    pi_r: tracksVisiting ? (tracksTransit as number) / tracksVisiting : null,
    chords: chords === null ? null : chords / divisor,
    sum_cos: sumCos === null ? null : sumCos / divisor,
    sum_sin: sumSin === null ? null : sumSin / divisor,
    sum_cos2: sumCos2 === null ? null : sumCos2 / divisor,
    sum_sin2: sumSin2 === null ? null : sumSin2 / divisor,
    r: chords ? (directedLength as number) / chords : null,
    r_axial: chords ? (axialLength as number) / chords : null,
    mean_bearing_deg:
      directedLength !== null && directedLength > zeroEpsilon
        ? normalize(Math.atan2(sumSin as number, sumCos as number) * 180 / Math.PI, 360)
        : null,
    axis_bearing_deg:
      axialLength !== null && axialLength > zeroEpsilon
        ? normalize(Math.atan2(sumSin2 as number, sumCos2 as number) * 90 / Math.PI, 180)
        : null,
    sectors: available
      ? Array.from(
          { length: 16 },
          (_, index) => rows.reduce((total, row) => total + (row.sectors[index] as number), 0) / divisor,
        )
      : null,
  }
}
