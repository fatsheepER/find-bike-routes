import { flushPromises, mount, type VueWrapper } from "@vue/test-utils"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import App from "./App.vue"

const leaflet = vi.hoisted(() => {
  const mapInstance = {
    fitBounds: vi.fn(),
    invalidateSize: vi.fn(),
    on: vi.fn(),
    remove: vi.fn(),
    removeLayer: vi.fn(),
    setMaxBounds: vi.fn(),
  }
  const geometryLayer = { addTo: vi.fn() }
  const groups: Array<{ addTo: ReturnType<typeof vi.fn>; clearLayers: ReturnType<typeof vi.fn> }> = []
  const mapObjects: Array<{ addTo: ReturnType<typeof vi.fn>; on: ReturnType<typeof vi.fn> }> = []
  const mapObject = () => {
    const object = {
      addTo: vi.fn(() => object),
      on: vi.fn(() => object),
    }
    mapObjects.push(object)
    return object
  }
  return {
    DomEvent: { stopPropagation: vi.fn() },
    divIcon: vi.fn((options) => options),
    geoJSON: vi.fn(() => geometryLayer),
    groups,
    latLngBounds: vi.fn((southWest: number[], northEast: number[]) => [southWest, northEast]),
    layerGroup: vi.fn(() => {
      const group = { addTo: vi.fn(), clearLayers: vi.fn() }
      groups.push(group)
      return group
    }),
    map: vi.fn(() => mapInstance),
    mapInstance,
    mapObjects,
    marker: vi.fn(mapObject),
    polyline: vi.fn(mapObject),
    tileLayer: vi.fn(() => ({ addTo: vi.fn(), on: vi.fn() })),
  }
})

const echarts = vi.hoisted(() => {
  const chart = { dispose: vi.fn(), resize: vi.fn(), setOption: vi.fn() }
  return { chart, init: vi.fn(() => chart) }
})

vi.mock("leaflet", () => ({ default: leaflet }))
vi.mock("echarts", () => ({ init: echarts.init }))

const point = (coordinates: [number, number]) => ({ type: "Point", coordinates })
const regions = Array.from({ length: 53 }, (_, index) => {
  const regionId = index + 1
  return {
    type: "Feature",
    id: regionId,
    properties: {
      region_id: regionId,
      region_code: `R-${regionId}`,
      district_id: regionId <= 2 ? 1 : 2,
      map_anchor: point([118 + regionId / 1000, 24 + regionId / 1000]),
      metrics: [],
    },
    geometry: point([118 + regionId / 1000, 24 + regionId / 1000]),
  }
})
const regionResponse = {
  release_digest: "a".repeat(64),
  island_boundary: { type: "Feature", properties: {}, geometry: point([118, 24]) },
  island_bounds: { west: 118, south: 24, east: 118.3, north: 24.3 },
  map_bounds: { west: 117.9, south: 23.9, east: 118.4, north: 24.4 },
  districts: {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { district_id: 1, label: "湖里" }, geometry: point([118.1, 24.1]) },
      { type: "Feature", properties: { district_id: 2, label: "思明" }, geometry: point([118.2, 24.2]) },
    ],
  },
  regions: { type: "FeatureCollection", features: regions },
}

function flow(from: number, to: number, weight: number) {
  return {
    matrix: "od",
    scope: "clear-days-stable",
    hour: 6,
    from_region: from,
    to_region: to,
    weight,
    is_tested: true,
    observed: weight * 4,
    is_significant: true,
    gated: false,
    is_self_loop: false,
  }
}

let wrapper: VueWrapper | undefined

beforeEach(() => {
  vi.clearAllMocks()
  leaflet.groups.length = 0
  leaflet.mapObjects.length = 0
  class ResizeObserverStub {
    observe = vi.fn()
    disconnect = vi.fn()
  }
  vi.stubGlobal("ResizeObserver", ResizeObserverStub)
  vi.stubGlobal("fetch", vi.fn())
  vi.mocked(fetch).mockImplementation(async (input) => {
    const url = String(input)
    if (url === "/api/regions") return { ok: true, json: async () => regionResponse } as Response
    if (url === "/api/health") return { ok: true, json: async () => ({ status: "ok", components: {} }) } as Response
    if (url.startsWith("/api/flows")) {
      const hour = Number(new URL(url, "http://test").searchParams.get("hour"))
      return {
        ok: true,
        json: async () => ({
          release_digest: "a".repeat(64),
          flows: hour === 6
            ? Array.from({ length: 52 }, (_, index) => flow(index + 1, index + 2, 52 - index))
            : [],
        }),
      } as Response
    }
    throw new Error(`unexpected request ${url}`)
  })
})

afterEach(() => {
  wrapper?.unmount()
  wrapper = undefined
  vi.unstubAllGlobals()
})

describe("region flow layer", () => {
  it("shows flow-only defaults, requests every hour, limits map and list to 50, and charts all flows", async () => {
    wrapper = mount(App)
    await flushPromises()
    expect(wrapper.find('[aria-label="流矩阵"]').exists()).toBe(false)

    await wrapper.get('[aria-label="内容图层"]').setValue("flows")
    await flushPromises()

    expect((wrapper.get('[aria-label="流矩阵"]').element as HTMLSelectElement).value).toBe("od")
    expect((wrapper.get('[aria-label="显著性"]').element as HTMLSelectElement).value).toBe("significant")
    for (const hour of [6, 7, 8, 9]) {
      expect(fetch).toHaveBeenCalledWith(`/api/flows?matrix=od&hour=${hour}`, expect.anything())
    }
    const flowList = wrapper.get('[aria-label="Top 50 区域流列表"]')
    expect(flowList.findAll("li")).toHaveLength(50)
    expect(flowList.find("button").text()).toContain("4 日平均，06:00–10:00 所选时段累计")
    expect(leaflet.polyline).toHaveBeenCalled()
    expect(echarts.init).toHaveBeenCalledTimes(1)
    const option = echarts.chart.setOption.mock.calls.at(-1)?.[0]
    expect(option.series[0].type).toBe("chord")
    expect(option.series[0].links.reduce((total: number, link: { value: number }) => total + link.value, 0)).toBe(1378)
    expect(option.series[0].links).toContainEqual(expect.objectContaining({
      source: "1",
      target: "1",
      name: "片区内部区域间流动",
    }))
  })

  it("synchronizes list and arc selection, stops map propagation, and shows flow details", async () => {
    wrapper = mount(App)
    await flushPromises()
    await wrapper.get('[aria-label="内容图层"]').setValue("flows")
    await flushPromises()

    const firstArcAfterSelection = leaflet.polyline.mock.results.length
    await wrapper.get('[aria-label="Top 50 区域流列表"]').findAll("button")[1].trigger("click")
    expect(wrapper.get('[aria-current="true"]').text()).toContain("R-2 → R-3")
    const details = wrapper.get('[aria-label="所选流对详情"]')
    expect(details.text()).toContain("出行流")
    expect(details.text()).toContain("湖里 → 思明")
    expect(details.text()).toContain("4 日平均，06:00–10:00 所选时段累计")
    expect(details.text()).toContain("已检验是")
    expect(details.text()).toContain("显著是")
    expect(details.text()).toContain("被门槛挡住否")

    const arcClick = leaflet.polyline.mock.results[firstArcAfterSelection]?.value.on.mock.calls[0][1]
    const originalEvent = new Event("click")
    arcClick({ originalEvent })
    await wrapper.vm.$nextTick()
    expect(leaflet.DomEvent.stopPropagation).toHaveBeenCalledWith(originalEvent)
    expect(wrapper.get('[aria-current="true"]').text()).toContain("R-1 → R-2")
  })

  it("requests daily union slices for all clear-day flows and isolates retryable failures", async () => {
    wrapper = mount(App)
    await flushPromises()
    await wrapper.get('[aria-label="内容图层"]').setValue("flows")
    await flushPromises()
    vi.mocked(fetch).mockImplementation(async (input) => {
      const url = String(input)
      if (url === "/api/regions") return { ok: true, json: async () => regionResponse } as Response
      if (url === "/api/health") return { ok: true, json: async () => ({ status: "ok", components: {} }) } as Response
      return { ok: false, json: async () => ({ detail: "private database password" }) } as Response
    })

    await wrapper.get('[aria-label="显著性"]').setValue("all")
    await flushPromises()

    expect(fetch).toHaveBeenCalledWith(
      "/api/flows?matrix=od&hour=6&date=2020-12-21",
      expect.anything(),
    )
    expect(fetch).toHaveBeenCalledWith(
      "/api/flows?matrix=od&hour=9&date=2020-12-25",
      expect.anything(),
    )
    expect(wrapper.text()).toContain("区域流暂不可用，区域地图仍可查看")
    expect(wrapper.text()).not.toContain("private database password")
    expect(leaflet.map).toHaveBeenCalledTimes(1)
    expect(wrapper.get('[aria-label="区域间流动"] [role="alert"] button').text()).toBe("重试")
  })
})
