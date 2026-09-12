import { describe, expect, it } from "vitest"
import {
  aggregateDistrictFlows,
  aggregateFlows,
  flowRequests,
  topRegionFlows,
  type FlowRow,
  type FlowSlice,
} from "./flow-aggregation"

function row(hour: number, weight: number, overrides: Partial<FlowRow> = {}): FlowRow {
  return {
    matrix: "od",
    scope: "2020-12-23",
    hour,
    from_region: 1,
    to_region: 2,
    weight,
    is_tested: true,
    observed: weight,
    is_significant: true,
    gated: false,
    is_self_loop: false,
    ...overrides,
  }
}

function slice(scope: FlowSlice["scope"], hour: number, rows: FlowRow[]): FlowSlice {
  return { scope, hour, rows }
}

describe("flow aggregation", () => {
  it("sums selected hourly slices without scaling a single day", () => {
    const slices = [
      slice("2020-12-23", 7, [row(7, 3)]),
      slice("2020-12-23", 8, [row(8, 5)]),
    ]

    expect(aggregateFlows(slices, {
      dateScope: "2020-12-23",
      aggregation: "average",
      significance: "significant",
    })[0].weight).toBe(8)
    expect(aggregateFlows(slices, {
      dateScope: "2020-12-23",
      aggregation: "sum",
      significance: "significant",
    })[0].weight).toBe(8)
  })

  it("uses stable clear-day slices and converts the server daily mean only for sum", () => {
    const slices = [
      slice("clear-days-stable", 6, [row(6, 3, { scope: "clear-days-stable" })]),
      slice("clear-days-stable", 7, [row(7, 5, { scope: "clear-days-stable" })]),
    ]

    expect(aggregateFlows(slices, {
      dateScope: "clear-days",
      aggregation: "average",
      significance: "significant",
    })[0].weight).toBe(8)
    expect(aggregateFlows(slices, {
      dateScope: "clear-days",
      aggregation: "sum",
      significance: "significant",
    })[0].weight).toBe(32)
    expect(flowRequests("od", "significant", "clear-days", 6, 8)).toEqual([
      { scope: "clear-days-stable", hour: 6, url: "/api/flows?matrix=od&hour=6" },
      { scope: "clear-days-stable", hour: 7, url: "/api/flows?matrix=od&hour=7" },
    ])
  })

  it("unions four daily clear-day slices, treats missing weights as zero, and composes significance", () => {
    const scopes = ["2020-12-21", "2020-12-22", "2020-12-24", "2020-12-25"]
    const slices = scopes.map((scope, index) => slice(scope, 6, [
      row(6, index + 1, { scope }),
      ...(index === 3 ? [] : [row(6, 2, {
        scope,
        from_region: 2,
        to_region: 3,
        is_significant: index !== 1,
      })]),
    ]))

    const flows = aggregateFlows(slices, {
      dateScope: "clear-days",
      aggregation: "average",
      significance: "all",
    })

    expect(flows.find((flow) => flow.from_region === 1)?.weight).toBe(2.5)
    expect(flows.find((flow) => flow.from_region === 1)?.is_significant).toBe(true)
    expect(flows.find((flow) => flow.from_region === 2)?.weight).toBe(1.5)
    expect(flows.find((flow) => flow.from_region === 2)?.is_tested).toBe(false)
    expect(flows.find((flow) => flow.from_region === 2)?.is_significant).toBe(false)
    expect(flowRequests("channel", "all", "clear-days", 6, 7)).toHaveLength(4)
    expect(flowRequests("channel", "all", "clear-days", 6, 7)[3]).toEqual({
      scope: "2020-12-25",
      hour: 6,
      url: "/api/flows?matrix=channel&hour=6&date=2020-12-25",
    })
  })

  it("strictly filters a single day while all keeps untested, gated, and non-significant pairs", () => {
    const rows = [
      row(6, 7),
      row(6, 6, { from_region: 2, to_region: 3, is_tested: false, is_significant: null }),
      row(6, 5, { from_region: 3, to_region: 4, is_significant: false, gated: true }),
    ]
    const slices = [slice("2020-12-23", 6, rows)]

    expect(aggregateFlows(slices, {
      dateScope: "2020-12-23",
      aggregation: "sum",
      significance: "significant",
    })).toHaveLength(1)
    const all = aggregateFlows(slices, {
      dateScope: "2020-12-23",
      aggregation: "sum",
      significance: "all",
    })
    expect(all).toHaveLength(3)
    expect(all.find((flow) => flow.from_region === 2)?.is_significant).toBeNull()
  })

  it("excludes self-loops and takes a stable Top 50 after sorting", () => {
    const flows = aggregateFlows([slice("2020-12-23", 6, [
      row(6, 999, { from_region: 1, to_region: 1, is_self_loop: true }),
      ...Array.from({ length: 52 }, (_, index) => row(6, 10, {
        from_region: index % 3,
        to_region: 60 - index,
      })),
    ])], {
      dateScope: "2020-12-23",
      aggregation: "sum",
      significance: "all",
    })

    const top = topRegionFlows(flows)
    expect(top).toHaveLength(50)
    expect(top.some((flow) => flow.from_region === flow.to_region)).toBe(false)
    expect(top.slice(0, 3).map((flow) => [flow.from_region, flow.to_region])).toEqual([
      [0, 9],
      [0, 12],
      [0, 15],
    ])
  })

  it("aggregates every filtered non-self-loop flow by district and keeps district self-links", () => {
    const flows = aggregateFlows([slice("2020-12-23", 6, [
      row(6, 4, { from_region: 1, to_region: 2 }),
      row(6, 3, { from_region: 2, to_region: 3 }),
      row(6, 20, { from_region: 3, to_region: 3, is_self_loop: true }),
    ])], {
      dateScope: "2020-12-23",
      aggregation: "sum",
      significance: "all",
    })
    const districts = new Map([[1, 7], [2, 7], [3, 8]])

    expect(aggregateDistrictFlows(flows, districts)).toEqual([
      { from_district: 7, to_district: 7, weight: 4, is_internal: true },
      { from_district: 7, to_district: 8, weight: 3, is_internal: false },
    ])
  })
})
