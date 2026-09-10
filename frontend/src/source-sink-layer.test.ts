import { flushPromises, mount } from "@vue/test-utils"
import { afterEach, describe, expect, it, vi } from "vitest"
import App from "./App.vue"
import {
  type AggregatedRegion,
  type RegionFeature,
  type RegionMetric,
  type RegionSelection,
} from "./region-aggregation"
import {
  colorScaleLimit,
  formatMetric,
  sourceSinkColor,
  sourceSinkLegend,
  sourceSinkState,
  sourceSinkTooltip,
} from "./source-sink-layer"

const leaflet = vi.hoisted(() => {
  const featureLayer = { bindTooltip: vi.fn() }
  const geometryLayer = { addTo: vi.fn() }
  const mapInstance = {
    fitBounds: vi.fn(),
    invalidateSize: vi.fn(),
    on: vi.fn(),
    remove: vi.fn(),
    removeLayer: vi.fn(),
    setMaxBounds: vi.fn(),
  }
  return {
    featureLayer,
    geoJSON: vi.fn((data: { features?: RegionFeature[] }, options?: { onEachFeature?: (feature: RegionFeature, layer: typeof featureLayer) => void }) => {
      data.features?.forEach((feature) => options?.onEachFeature?.(feature, featureLayer))
      return geometryLayer
    }),
    latLngBounds: vi.fn((southWest: number[], northEast: number[]) => [southWest, northEast]),
    map: vi.fn(() => mapInstance),
    mapInstance,
    tileLayer: vi.fn(() => ({ addTo: vi.fn(), on: vi.fn() })),
  }
})

vi.mock("leaflet", () => ({ default: leaflet }))

const selection: RegionSelection = {
  dateScope: "clear-days",
  startHour: 6,
  endHour: 10,
  aggregation: "average",
}

function aggregatedRegion(net_inflow_per_km2: number | null): AggregatedRegion {
  return {
    available: net_inflow_per_km2 !== null,
    region_id: 1,
    region_code: "思明-1",
    district_id: 3,
    cells: 4,
    area_km2: 2,
    functional_composition: {
      residential: 0.1,
      employment: 0.2,
      education: 0.3,
      transport: 0.4,
    },
    classified_share: 0.75,
    bus_stops_per_km2: 6,
    unlocks: net_inflow_per_km2 === null ? null : 10,
    locks: net_inflow_per_km2 === null ? null : 14,
    net_inflow: net_inflow_per_km2 === null ? null : 4,
    net_inflow_per_km2,
    order_events_per_km2: net_inflow_per_km2 === null ? null : 12,
    tracks_visiting: net_inflow_per_km2 === null ? null : 20,
    tracks_transit: net_inflow_per_km2 === null ? null : 5,
    pi_r: net_inflow_per_km2 === null ? null : 0.25,
    chords: net_inflow_per_km2 === null ? null : 4,
    sum_cos: net_inflow_per_km2 === null ? null : 2,
    sum_sin: net_inflow_per_km2 === null ? null : 0,
    sum_cos2: net_inflow_per_km2 === null ? null : 0,
    sum_sin2: net_inflow_per_km2 === null ? null : 2,
    r: net_inflow_per_km2 === null ? null : 0.5,
    r_axial: net_inflow_per_km2 === null ? null : 0.5,
    mean_bearing_deg: net_inflow_per_km2 === null ? null : 0,
    axis_bearing_deg: net_inflow_per_km2 === null ? null : 45,
    sectors: net_inflow_per_km2 === null ? null : [1, ...Array(15).fill(0)],
  }
}

describe("source-sink layer", () => {
  it("uses the largest available absolute intensity as a symmetric dynamic scale", () => {
    const regions = [aggregatedRegion(-8), aggregatedRegion(3), aggregatedRegion(0), aggregatedRegion(null)]

    expect(colorScaleLimit(regions)).toBe(8)
    expect(sourceSinkColor(-8, 8)).not.toBe(sourceSinkColor(-3, 8))
    expect(sourceSinkLegend(8, selection)).toContain("-8 至 +8 单/平方公里")
  })

  it("keeps null out of the numeric domain and distinguishes it from valid zero", () => {
    expect(colorScaleLimit([aggregatedRegion(null)])).toBeNull()
    expect(colorScaleLimit([aggregatedRegion(null), aggregatedRegion(0)])).toBe(0)
    expect(sourceSinkLegend(null, selection)).toContain("无可计算数值")
    expect(formatMetric(null)).toBe("不可计算")
    expect(formatMetric(0)).toBe("0")
    expect(sourceSinkState(null)).toBe("不可计算")
    expect(sourceSinkState(0)).toBe("平衡 0")
    expect(sourceSinkState(-2)).toContain("源 −")
    expect(sourceSinkState(2)).toContain("汇 +")
  })

  it("puts the date, cumulative time, aggregation meaning, direction rose, and static profile next to tooltip values", () => {
    const tooltip = sourceSinkTooltip(aggregatedRegion(2), selection, "思明片区")

    expect(sourceSinkLegend(2, selection)).toContain("晴天集")
    expect(sourceSinkLegend(2, selection)).toContain("4 日平均，06:00–10:00 所选时段累计")
    expect(tooltip).toContain("汇 +2")
    expect(tooltip).toContain("思明片区")
    expect(tooltip).toContain("解锁 10")
    expect(tooltip).toContain("PI_r 0.25")
    expect(tooltip).toContain("方向玫瑰（16 扇区）")
    expect(tooltip.match(/<line /g)).toHaveLength(16)
    expect(tooltip).toContain("静态画像（不随筛选变化）")
    expect(tooltip).toContain("4 日平均，06:00–10:00 所选时段累计")
    expect(sourceSinkTooltip(aggregatedRegion(null), selection, "思明片区")).not.toContain("不可计算°")
  })

  it("renders the default layer and keeps global filters when switching away and back", async () => {
    const dates = ["2020-12-21", "2020-12-22", "2020-12-23", "2020-12-24", "2020-12-25"]
    const metrics = dates.flatMap((date) =>
      [6, 7, 8, 9].map((hour) => ({
        date,
        hour,
        unlocks: 1,
        locks: 3,
        net_inflow: 2,
        net_inflow_per_km2: 999,
        order_events_per_km2: 999,
        tracks_visiting: 2,
        tracks_transit: 1,
        pi_r: 999,
        chords: 1,
        sum_cos: 1,
        sum_sin: 0,
        sum_cos2: 1,
        sum_sin2: 0,
        r: 999,
        r_axial: 999,
        mean_bearing_deg: 999,
        axis_bearing_deg: 999,
        sectors: [1, ...Array(15).fill(0)],
      } satisfies RegionMetric)),
    )
    const feature = {
      type: "Feature",
      id: 1,
      geometry: { type: "Polygon", coordinates: [] },
      properties: {
        ...aggregatedRegion(0),
        metrics,
      },
    } as unknown as RegionFeature
    vi.stubEnv("VITE_CARTO_API_KEY", "test-key")
    vi.stubGlobal("ResizeObserver", class { observe() {}; disconnect() {} })
    vi.stubGlobal("fetch", vi.fn(async (input) => ({
      ok: true,
      json: async () => input === "/api/health"
        ? { status: "ok", components: {} }
        : {
            release_digest: "a".repeat(64),
            island_boundary: { type: "Feature", geometry: { type: "Polygon", coordinates: [] } },
            island_bounds: { west: 118, south: 24, east: 118.2, north: 24.2 },
            map_bounds: { west: 117.9, south: 23.9, east: 118.3, north: 24.3 },
            districts: {
              type: "FeatureCollection",
              features: [{ type: "Feature", geometry: feature.geometry, properties: { district_id: 3, label: "思明片区" } }],
            },
            regions: { type: "FeatureCollection", features: [feature] },
          },
    } as Response)))

    const wrapper = mount(App)
    await flushPromises()

    expect(wrapper.get('[aria-label="净流入强度图例"]').text()).toContain("4 日平均，06:00–10:00 所选时段累计")
    expect(leaflet.featureLayer.bindTooltip).toHaveBeenCalledWith(
      expect.stringContaining("思明片区"),
      expect.objectContaining({ sticky: true }),
    )
    expect(leaflet.featureLayer.bindTooltip).toHaveBeenCalledWith(
      expect.stringContaining("汇 +4"),
      expect.anything(),
    )

    await wrapper.get('[aria-label="日期"]').setValue("2020-12-23")
    await wrapper.get('[aria-label="开始时间"]').setValue("7")
    await wrapper.get('[aria-label="结束时间"]').setValue("9")
    await wrapper.get('[aria-label="聚合口径"]').setValue("sum")
    await wrapper.get('[aria-label="内容图层"]').setValue("flows")
    expect(wrapper.find('[aria-label="净流入强度图例"]').exists()).toBe(false)

    await wrapper.get('[aria-label="内容图层"]').setValue("source-sink")
    expect(wrapper.get('[aria-label="净流入强度图例"]').text()).toContain("2020-12-23")
    expect(wrapper.get('[aria-label="净流入强度图例"]').text()).toContain("单日，07:00–09:00 所选时段累计")
    expect((wrapper.get('[aria-label="聚合口径"]').element as HTMLSelectElement).value).toBe("sum")

    wrapper.unmount()
  })
})

afterEach(() => {
  vi.clearAllMocks()
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})
