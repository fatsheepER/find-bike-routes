import { describe, expect, it } from "vitest"
import { aggregateRegion, type RegionFeature } from "./region-aggregation"

function metric(date: string, hour: number, value: number, overrides: Record<string, unknown> = {}) {
  return {
    date,
    hour,
    unlocks: value,
    locks: value * 2,
    net_inflow: value,
    net_inflow_per_km2: 999,
    order_events_per_km2: 999,
    tracks_visiting: value * 4,
    tracks_transit: value * 2,
    pi_r: 999,
    chords: value,
    sum_cos: value,
    sum_sin: 0,
    sum_cos2: 0,
    sum_sin2: value,
    r: 999,
    r_axial: 999,
    mean_bearing_deg: 999,
    axis_bearing_deg: 999,
    sectors: Array.from({ length: 16 }, (_, index) => value + index),
    ...overrides,
  }
}

function region(metrics: ReturnType<typeof metric>[]): RegionFeature {
  return {
    type: "Feature",
    id: 7,
    geometry: { type: "Polygon", coordinates: [] },
    properties: {
      region_id: 7,
      region_code: "思明-7",
      district_id: 2,
      cells: 3,
      area_km2: 2,
      functional_composition: {
        residential: 0.1,
        employment: 0.2,
        education: 0.3,
        transport: 0.4,
      },
      classified_share: 0.75,
      bus_stops_per_km2: 6,
      metrics,
    },
  }
}

describe("region aggregation", () => {
  it("sums the selected hours, then averages only across the selected dates", () => {
    const feature = region([
      metric("2020-12-21", 6, 1),
      metric("2020-12-21", 7, 2),
      metric("2020-12-22", 6, 3),
      metric("2020-12-22", 7, 4),
      metric("2020-12-24", 6, 5),
      metric("2020-12-24", 7, 6),
      metric("2020-12-25", 6, 7),
      metric("2020-12-25", 7, 8),
    ])

    const average = aggregateRegion(feature, {
      dateScope: "clear-days",
      startHour: 6,
      endHour: 8,
      aggregation: "average",
    })
    const sum = aggregateRegion(feature, {
      dateScope: "clear-days",
      startHour: 6,
      endHour: 8,
      aggregation: "sum",
    })

    expect(average.unlocks).toBe(9)
    expect(average.net_inflow_per_km2).toBe(4.5)
    expect(average.order_events_per_km2).toBe(13.5)
    expect(sum.unlocks).toBe(36)
    expect(sum.net_inflow_per_km2).toBe(18)
    expect(sum.order_events_per_km2).toBe(54)
    expect(average.functional_composition).toEqual(feature.properties.functional_composition)
    expect(average.classified_share).toBe(0.75)
    expect(average.bus_stops_per_km2).toBe(6)
  })

  it("keeps single-day average equal to sum across a complete or partial time range", () => {
    const feature = region([
      metric("2020-12-23", 6, 1),
      metric("2020-12-23", 7, 2),
      metric("2020-12-23", 8, 4),
      metric("2020-12-23", 9, 8),
    ])
    const partial = { dateScope: "2020-12-23", startHour: 7, endHour: 9 } as const
    const full = { dateScope: "2020-12-23", startHour: 6, endHour: 10 } as const

    expect(aggregateRegion(feature, { ...partial, aggregation: "average" }).unlocks).toBe(6)
    expect(aggregateRegion(feature, { ...partial, aggregation: "sum" }).unlocks).toBe(6)
    expect(aggregateRegion(feature, { ...full, aggregation: "average" }).unlocks).toBe(15)
    expect(aggregateRegion(feature, { ...full, aggregation: "sum" }).unlocks).toBe(15)
  })

  it("recomputes ratios, directions, and all 16 rose sectors from additive components", () => {
    const feature = region([
      metric("2020-12-21", 6, 1, {
        tracks_visiting: 4,
        tracks_transit: 1,
        chords: 2,
        sum_cos: 1,
        sum_sin: 0,
        sum_cos2: 0,
        sum_sin2: 1,
        sectors: [1, ...Array(15).fill(0)],
      }),
      metric("2020-12-21", 7, 1, {
        tracks_visiting: 2,
        tracks_transit: 2,
        chords: 2,
        sum_cos: 0,
        sum_sin: 1,
        sum_cos2: -1,
        sum_sin2: 0,
        sectors: [2, ...Array(14).fill(0), 3],
      }),
    ])

    const result = aggregateRegion(feature, {
      dateScope: "2020-12-21",
      startHour: 6,
      endHour: 8,
      aggregation: "sum",
    })

    expect(result.tracks_visiting).toBe(6)
    expect(result.tracks_transit).toBe(3)
    expect(result.pi_r).toBe(0.5)
    expect(result.chords).toBe(4)
    expect(result.r).toBeCloseTo(Math.SQRT2 / 4)
    expect(result.r_axial).toBeCloseTo(Math.SQRT2 / 4)
    expect(result.mean_bearing_deg).toBe(45)
    expect(result.axis_bearing_deg).toBe(67.5)
    expect(result.sectors).toEqual([3, ...Array(14).fill(0), 3])
  })

  it("distinguishes zero denominators and zero vectors from valid numeric zero", () => {
    const noTraffic = aggregateRegion(
      region([metric("2020-12-21", 6, 0)]),
      { dateScope: "2020-12-21", startHour: 6, endHour: 7, aggregation: "sum" },
    )
    const balancedDirections = aggregateRegion(
      region([
        metric("2020-12-21", 6, 1, {
          tracks_visiting: 2,
          tracks_transit: 0,
          chords: 2,
          sum_cos: 1,
          sum_sin: 1,
          sum_cos2: 1,
          sum_sin2: 1,
        }),
        metric("2020-12-21", 7, -1, {
          unlocks: 0,
          locks: 0,
          tracks_visiting: 0,
          tracks_transit: 0,
          chords: 0,
          sum_cos: -1,
          sum_sin: -1,
          sum_cos2: -1,
          sum_sin2: -1,
          sectors: Array(16).fill(0),
        }),
      ]),
      { dateScope: "2020-12-21", startHour: 6, endHour: 8, aggregation: "sum" },
    )

    expect(noTraffic.net_inflow).toBe(0)
    expect(noTraffic.pi_r).toBeNull()
    expect(noTraffic.r).toBeNull()
    expect(noTraffic.r_axial).toBeNull()
    expect(noTraffic.mean_bearing_deg).toBeNull()
    expect(noTraffic.axis_bearing_deg).toBeNull()
    expect(balancedDirections.pi_r).toBe(0)
    expect(balancedDirections.r).toBe(0)
    expect(balancedDirections.r_axial).toBe(0)
    expect(balancedDirections.mean_bearing_deg).toBeNull()
    expect(balancedDirections.axis_bearing_deg).toBeNull()
  })

  it("returns unavailable values when a selected row or additive component is missing", () => {
    const missingRow = aggregateRegion(
      region([metric("2020-12-21", 6, 0)]),
      { dateScope: "2020-12-21", startHour: 6, endHour: 8, aggregation: "sum" },
    )
    const missingValue = aggregateRegion(
      region([metric("2020-12-21", 6, 0, { net_inflow: null })]),
      { dateScope: "2020-12-21", startHour: 6, endHour: 7, aggregation: "sum" },
    )
    const duplicateInsteadOfNextHour = aggregateRegion(
      region([metric("2020-12-21", 6, 0), metric("2020-12-21", 6, 0)]),
      { dateScope: "2020-12-21", startHour: 6, endHour: 8, aggregation: "sum" },
    )

    expect(missingRow.available).toBe(false)
    expect(missingRow.net_inflow).toBeNull()
    expect(missingValue.available).toBe(false)
    expect(missingValue.net_inflow_per_km2).toBeNull()
    expect(duplicateInsteadOfNextHour.available).toBe(false)
  })
})
