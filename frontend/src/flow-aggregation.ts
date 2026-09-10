import { CLEAR_DAYS, type Aggregation, type DateScope } from "./region-aggregation"

export type FlowMatrix = "od" | "channel"
export type FlowSignificance = "significant" | "all"

export type FlowRow = {
  matrix: FlowMatrix
  scope: string
  hour: number
  from_region: number
  to_region: number
  weight: number
  is_tested: boolean
  observed: number | null
  is_significant: boolean | null
  gated: boolean
  is_self_loop: boolean
}

export type FlowSlice = {
  scope: string
  hour: number
  rows: FlowRow[]
}

export type FlowAggregationSelection = {
  dateScope: DateScope
  aggregation: Aggregation
  significance: FlowSignificance
}

export type AggregatedFlow = Omit<FlowRow, "scope" | "hour" | "observed"> & {
  scope: string
}

export type FlowRequest = {
  scope: string
  hour: number
  url: string
}

export type DistrictFlow = {
  from_district: number
  to_district: number
  weight: number
  is_internal: boolean
}

export function flowRequests(
  matrix: FlowMatrix,
  significance: FlowSignificance,
  dateScope: DateScope,
  startHour: number,
  endHour: number,
): FlowRequest[] {
  const scopes = dateScope === "clear-days"
    ? significance === "significant" ? ["clear-days-stable"] : [...CLEAR_DAYS]
    : [dateScope]
  return scopes.flatMap((scope) =>
    Array.from({ length: endHour - startHour }, (_, index) => {
      const hour = startHour + index
      const date = scope === "clear-days-stable" ? "" : `&date=${scope}`
      return { scope, hour, url: `/api/flows?matrix=${matrix}&hour=${hour}${date}` }
    }),
  )
}

export function aggregateFlows(
  slices: FlowSlice[],
  selection: FlowAggregationSelection,
): AggregatedFlow[] {
  const byPair = new Map<string, FlowRow[]>()
  for (const slice of slices) {
    for (const flow of slice.rows) {
      const key = `${flow.from_region}/${flow.to_region}`
      byPair.set(key, [...(byPair.get(key) ?? []), flow])
    }
  }

  const usesStableClearDayAggregate = slices.some((slice) => slice.scope === "clear-days-stable")
  const requiredScopes = selection.dateScope === "clear-days"
    ? usesStableClearDayAggregate ? ["clear-days-stable"] : [...CLEAR_DAYS]
    : [selection.dateScope]
  const scale = selection.dateScope !== "clear-days"
    ? 1
    : usesStableClearDayAggregate
      ? selection.aggregation === "sum" ? CLEAR_DAYS.length : 1
      : selection.aggregation === "average" ? 1 / CLEAR_DAYS.length : 1

  const flows = [...byPair.values()].map((rows) => {
    const first = rows[0]
    const isTested = requiredScopes.every((scope) =>
      rows.some((flow) => flow.scope === scope && flow.is_tested),
    )
    const isSignificant = requiredScopes.every((scope) =>
      rows.some((flow) => flow.scope === scope && flow.is_significant === true),
    )
    return {
      matrix: first.matrix,
      scope: selection.dateScope,
      from_region: first.from_region,
      to_region: first.to_region,
      weight: rows.reduce((total, flow) => total + flow.weight, 0) * scale,
      is_tested: isTested,
      is_significant: selection.dateScope !== "clear-days" && !isTested ? null : isSignificant,
      gated: rows.some((flow) => flow.gated),
      is_self_loop: first.from_region === first.to_region || rows.some((flow) => flow.is_self_loop),
    }
  })
  return selection.significance === "significant"
    ? flows.filter((flow) => flow.is_significant)
    : flows
}

export function topRegionFlows(flows: AggregatedFlow[], limit = 50): AggregatedFlow[] {
  return flows
    .filter((flow) => !flow.is_self_loop && flow.from_region !== flow.to_region)
    .sort((left, right) =>
      right.weight - left.weight ||
      left.from_region - right.from_region ||
      left.to_region - right.to_region,
    )
    .slice(0, limit)
}

export function aggregateDistrictFlows(
  flows: AggregatedFlow[],
  districtByRegion: ReadonlyMap<number, number>,
): DistrictFlow[] {
  const totals = new Map<string, DistrictFlow>()
  for (const flow of flows) {
    if (flow.is_self_loop || flow.from_region === flow.to_region) continue
    const fromDistrict = districtByRegion.get(flow.from_region)
    const toDistrict = districtByRegion.get(flow.to_region)
    if (fromDistrict === undefined || toDistrict === undefined) continue
    const key = `${fromDistrict}/${toDistrict}`
    const current = totals.get(key)
    totals.set(key, {
      from_district: fromDistrict,
      to_district: toDistrict,
      weight: (current?.weight ?? 0) + flow.weight,
      is_internal: fromDistrict === toDistrict,
    })
  }
  return [...totals.values()].sort((left, right) =>
    left.from_district - right.from_district || left.to_district - right.to_district,
  )
}
