import { selectDate, currentDate, focusHour } from "./app-test-support"
import { flushPromises, mount, type VueWrapper } from "@vue/test-utils"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import App from "./App.vue"

const leaflet = vi.hoisted(() => {
  const regionClicks = new Map<number, () => void>()
  const mapHandlers = new Map<string, (event?: { latlng?: { lat: number; lng: number } }) => void>()
  const mapInstance = {
    createPane: vi.fn(() => document.createElement("div")),
    dragging: { disable: vi.fn(), enable: vi.fn() },
    fitBounds: vi.fn(),
    getCenter: vi.fn(() => ({ lat: 24.15, lng: 118.15 })),
    getZoom: vi.fn(() => 13),
    invalidateSize: vi.fn(),
    on: vi.fn((event: string, handler: (event?: { latlng?: { lat: number; lng: number } }) => void) => {
      mapHandlers.set(event, handler)
    }),
    remove: vi.fn(),
    removeLayer: vi.fn(),
    setMaxBounds: vi.fn(),
    setView: vi.fn(),
  }
  const mapObjects: Array<{ addTo: ReturnType<typeof vi.fn>; on: ReturnType<typeof vi.fn> }> = []
  const mapObject = () => {
    const object = { addTo: vi.fn(() => object), on: vi.fn(() => object) }
    mapObjects.push(object)
    return object
  }

  return {
    DomEvent: { stopPropagation: vi.fn() },
    circleMarker: vi.fn(mapObject),
    divIcon: vi.fn((options) => options),
    geoJSON: vi.fn((data: { features?: Array<{ properties?: Record<string, unknown> }> }, options?: {
      onEachFeature?: (feature: { properties?: Record<string, unknown> }, layer: object) => void
    }) => {
      data.features?.forEach((feature) => options?.onEachFeature?.(feature, {
        bindTooltip: vi.fn(),
        on: vi.fn((event: string, handler: () => void) => {
          if (event === "click" && typeof feature.properties?.region_id === "number") {
            regionClicks.set(feature.properties.region_id, handler)
          }
        }),
      }))
      return mapObject()
    }),
    latLngBounds: vi.fn((southWest, northEast) => [southWest, northEast]),
    layerGroup: vi.fn(() => {
      const group = { addTo: vi.fn(() => group), clearLayers: vi.fn() }
      return group
    }),
    map: vi.fn(() => mapInstance),
    mapHandlers,
    mapInstance,
    mapObjects,
    marker: vi.fn(mapObject),
    polyline: vi.fn(mapObject),
    rectangle: vi.fn(mapObject),
    regionClicks,
    tileLayer: vi.fn(() => ({ addTo: vi.fn(), on: vi.fn() })),
  }
})

const echarts = vi.hoisted(() => ({
  chart: { dispose: vi.fn(), resize: vi.fn(), setOption: vi.fn() },
  init: vi.fn(() => echarts.chart),
}))

vi.mock("leaflet", () => ({ default: leaflet }))
vi.mock("echarts", () => ({ init: echarts.init }))

function region(regionId: number) {
  return {
    type: "Feature",
    id: regionId,
    geometry: { type: "Point", coordinates: [118 + regionId / 1000, 24 + regionId / 1000] },
    properties: {
      region_id: regionId,
      region_code: `R-${regionId}`,
      district_id: 1,
      cells: 1,
      area_km2: 1,
      functional_composition: { residential: 0, employment: 0, education: 0, transport: 0 },
      classified_share: 0,
      bus_stops_per_km2: 0,
      map_anchor: { type: "Point", coordinates: [118 + regionId / 1000, 24 + regionId / 1000] },
      metrics: [],
    },
  }
}

const regionResponse = {
  release_digest: "a".repeat(64),
  island_boundary: region(1),
  island_bounds: { west: 118, south: 24, east: 118.2, north: 24.2 },
  map_bounds: { west: 117.9, south: 23.9, east: 118.3, north: 24.3 },
  districts: {
    type: "FeatureCollection",
    features: [{ type: "Feature", geometry: region(1).geometry, properties: { district_id: 1, label: "测试片区" } }],
  },
  regions: { type: "FeatureCollection", features: [region(7), region(8)] },
}

const flows = [
  { from_region: 7, to_region: 8, weight: 9 },
  { from_region: 8, to_region: 7, weight: 8 },
].map((flow) => ({
  matrix: "od",
  scope: "clear-days-stable",
  hour: 6,
  is_tested: true,
  observed: flow.weight * 4,
  is_significant: true,
  gated: false,
  is_self_loop: false,
  ...flow,
}))

const sequenceResponse = {
  scope: "clear-days",
  min_contiguous_support: 0.001,
  min_contiguous_support_count: 1,
  valid_tracks: 10,
  limit: 20,
  patterns: [
    { region_ids: [7, 8], region_codes: ["R-7", "R-8"], district_ids: [1, 1], length: 2, support: 2, contiguous_support: 2 },
    { region_ids: [8, 7], region_codes: ["R-8", "R-7"], district_ids: [1, 1], length: 2, support: 1, contiguous_support: 1 },
  ],
}

let wrapper: VueWrapper | undefined

beforeEach(() => {
  vi.clearAllMocks()
  leaflet.regionClicks.clear()
  leaflet.mapHandlers.clear()
  leaflet.mapObjects.length = 0
  class ResizeObserverStub {
    observe = vi.fn()
    disconnect = vi.fn()
  }
  vi.stubGlobal("ResizeObserver", ResizeObserverStub)
  vi.stubGlobal("fetch", vi.fn(async (input) => {
    const url = String(input)
    if (url === "/api/regions") return { ok: true, status: 200, json: async () => regionResponse } as Response
    if (url === "/api/health") return { ok: true, status: 200, json: async () => ({ status: "ok", components: {} }) } as Response
    if (url.startsWith("/api/flows")) {
      const scope = new URL(url, "http://test").searchParams.get("date") ?? "clear-days-stable"
      return { ok: true, status: 200, json: async () => ({ flows: flows.map((flow) => ({ ...flow, scope })) }) } as Response
    }
    if (url.startsWith("/api/sequences")) return { ok: true, status: 200, json: async () => sequenceResponse } as Response
    return {
      ok: true,
      status: 200,
      json: async () => ({ total_count: 1, samples: { type: "FeatureCollection", features: [] } }),
    } as Response
  }))
})

afterEach(() => {
  wrapper?.unmount()
  wrapper = undefined
  vi.unstubAllGlobals()
})

describe("focus restoration", () => {
  it.each([
    ["source-sink", "7", "9"],
    ["flows", "7", "9"],
    ["sequences", "6", "10"],
  ])("enters from the %s background with the correct local time and locks it", async (background, focusStart, focusEnd) => {
    wrapper = mount(App)
    await flushPromises()
    await wrapper.get('[aria-label="开始时间"]').setValue("7")
    await wrapper.get('[aria-label="结束时间"]').setValue("9")
    await wrapper.get(`[data-layer="${background}"]`).trigger("click")
    await flushPromises()

    leaflet.regionClicks.get(7)?.()
    await flushPromises()

    expect(wrapper.find('[aria-label="区域聚焦详情"]').exists()).toBe(true)
    expect((wrapper.get('[role="tab"][aria-selected="true"]').element as HTMLButtonElement).disabled).toBe(true)
    expect((wrapper.get('[aria-label="框选范围"]').element as HTMLButtonElement).disabled).toBe(true)
    expect(focusHour(wrapper, "start")).toBe(focusStart)
    expect(focusHour(wrapper, "end")).toBe(focusEnd)
  })

  it("gives flow arcs, sequence lines, and step nodes priority over region polygons", async () => {
    wrapper = mount(App)
    await flushPromises()
    await wrapper.get('[data-layer="flows"]').trigger("click")
    await flushPromises()
    const flowArc = leaflet.polyline.mock.results[0].value
    flowArc.on.mock.calls[0][1]({ originalEvent: new Event("click") })
    await wrapper.vm.$nextTick()

    expect(leaflet.DomEvent.stopPropagation).toHaveBeenCalled()
    expect(wrapper.find('[aria-label="区域聚焦详情"]').exists()).toBe(false)

    leaflet.polyline.mockClear()
    leaflet.circleMarker.mockClear()
    await wrapper.get('[data-layer="sequences"]').trigger("click")
    await flushPromises()
    leaflet.polyline.mock.results[1].value.on.mock.calls[0][1]({ originalEvent: new Event("click") })
    leaflet.circleMarker.mock.results[0].value.on.mock.calls[0][1]({ originalEvent: new Event("click") })
    await wrapper.vm.$nextTick()

    expect(wrapper.find('[aria-label="区域聚焦详情"]').exists()).toBe(false)
    expect(leaflet.DomEvent.stopPropagation).toHaveBeenCalledTimes(3)
  })

  it("restores the selected flow and map view on exit, then globally resets normal state", async () => {
    wrapper = mount(App)
    await flushPromises()
    await selectDate(wrapper, "2020-12-23")
    await wrapper.get('[aria-label="开始时间"]').setValue("7")
    await wrapper.get('[aria-label="聚合口径"]').setValue("sum")
    await wrapper.get('[data-layer="flows"]').trigger("click")
    await flushPromises()
    await wrapper.get('[aria-label="Top 50 区域流列表"]').findAll("button")[1].trigger("click")

    leaflet.regionClicks.get(7)?.()
    await flushPromises()
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }))
    await wrapper.vm.$nextTick()

    expect(wrapper.get('[role="tab"][aria-selected="true"]').attributes("data-layer")).toBe("flows")
    expect(currentDate(wrapper)).toBe("2020-12-23")
    expect((wrapper.get('[aria-label="开始时间"]').element as HTMLInputElement).value).toBe("7")
    expect((wrapper.get('[aria-label="聚合口径"]').element as HTMLSelectElement).value).toBe("sum")
    expect(wrapper.get('[aria-current="true"]').text()).toContain("R-8 → R-7")
    expect(leaflet.mapInstance.setView).toHaveBeenCalledWith(
      { lat: 24.15, lng: 118.15 },
      13,
      { animate: false },
    )

    await wrapper.get('[aria-label="框选范围"]').trigger('click')
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
    await wrapper.vm.$nextTick()
    expect(wrapper.get('[aria-label="框选范围"]').text()).toBe('框选')
    expect(currentDate(wrapper)).toBe('2020-12-23')
  })
})
