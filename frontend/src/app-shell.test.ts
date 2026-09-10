import { flushPromises, mount, type VueWrapper } from "@vue/test-utils"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import App from "./App.vue"

const leaflet = vi.hoisted(() => {
  const handlers: Record<string, (event?: { originalEvent?: Event }) => void> = {}
  const mapInstance = {
    fitBounds: vi.fn(),
    invalidateSize: vi.fn(),
    on: vi.fn((event: string, handler: (event?: { originalEvent?: Event }) => void) => {
      handlers[event] = handler
    }),
    remove: vi.fn(),
    removeLayer: vi.fn(),
    setMaxBounds: vi.fn(),
  }
  const tile = { addTo: vi.fn(), on: vi.fn() }
  const geometryLayer = { addTo: vi.fn() }

  return {
    geometryLayer,
    geoJSON: vi.fn(() => geometryLayer),
    handlers,
    latLngBounds: vi.fn((southWest: number[], northEast: number[]) => [southWest, northEast]),
    map: vi.fn(() => mapInstance),
    mapInstance,
    tile,
    tileLayer: vi.fn(() => tile),
  }
})

vi.mock("leaflet", () => ({ default: leaflet }))

const regionResponse = {
  release_digest: "a".repeat(64),
  island_boundary: {
    type: "Feature",
    properties: { name: "测试岛" },
    geometry: {
      type: "Polygon",
      coordinates: [[[118.0, 24.0], [118.3, 24.0], [118.3, 24.3], [118.0, 24.3], [118.0, 24.0]]],
    },
  },
  island_bounds: { west: 117.9, south: 23.9, east: 118.3, north: 24.3 },
  map_bounds: { west: 117.86, south: 23.86, east: 118.34, north: 24.34 },
  districts: { type: "FeatureCollection", features: [] },
  regions: { type: "FeatureCollection", features: [] },
}

let resizeCallback: ResizeObserverCallback
let wrapper: VueWrapper | undefined

beforeEach(() => {
  vi.clearAllMocks()
  vi.stubEnv("VITE_CARTO_API_KEY", "test-carto-key")
  Object.keys(leaflet.handlers).forEach((event) => delete leaflet.handlers[event])
  document.body.innerHTML = '<div id="app"></div>'
  class ResizeObserverStub {
    constructor(callback: ResizeObserverCallback) {
      resizeCallback = callback
    }

    observe = vi.fn()
    disconnect = vi.fn()
    unobserve = vi.fn()
  }
  vi.stubGlobal("ResizeObserver", ResizeObserverStub)
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve(regionResponse) }),
  )
})

afterEach(() => {
  wrapper?.unmount()
  wrapper = undefined
  vi.resetModules()
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe("single-map app shell", () => {
  it("starts from the development entry with the specified toolbar defaults", async () => {
    await import("./main")
    await flushPromises()

    expect(document.querySelector("h1")?.textContent).toContain("厦门本岛早高峰共享单车流动")
    expect((document.querySelector('[aria-label="内容图层"]') as HTMLSelectElement).value).toBe("source-sink")
    expect((document.querySelector('[aria-label="日期"]') as HTMLSelectElement).value).toBe("clear-days")
    expect((document.querySelector('[aria-label="开始时间"]') as HTMLInputElement).value).toBe("6")
    expect((document.querySelector('[aria-label="结束时间"]') as HTMLInputElement).value).toBe("10")
    expect((document.querySelector('[aria-label="聚合口径"]') as HTMLSelectElement).value).toBe("average")
  })

  it("keeps one map, fits exact island bounds, and protects a user-adjusted view", async () => {
    wrapper = mount(App, { attachTo: document.body })
    const mapElement = wrapper.get("#map").element
    let width = 1200
    let height = 800
    Object.defineProperties(mapElement, {
      clientWidth: { get: () => width },
      clientHeight: { get: () => height },
    })
    await flushPromises()

    expect(leaflet.map).toHaveBeenCalledTimes(1)
    expect(leaflet.mapInstance.setMaxBounds).toHaveBeenCalledWith([
      [23.86, 117.86],
      [24.34, 118.34],
    ])
    expect(leaflet.mapInstance.fitBounds).toHaveBeenLastCalledWith(
      [
        [23.9, 117.9],
        [24.3, 118.3],
      ],
      { animate: false, padding: [32, 32] },
    )

    height = 400
    resizeCallback([], {} as ResizeObserver)
    expect(leaflet.mapInstance.fitBounds).toHaveBeenCalledTimes(2)
    expect(leaflet.mapInstance.fitBounds).toHaveBeenLastCalledWith(expect.anything(), {
      animate: false,
      padding: [24, 24],
    })

    width = 3000
    height = 2000
    resizeCallback([], {} as ResizeObserver)
    expect(leaflet.mapInstance.fitBounds).toHaveBeenCalledTimes(3)
    expect(leaflet.mapInstance.fitBounds).toHaveBeenLastCalledWith(expect.anything(), {
      animate: false,
      padding: [64, 64],
    })

    leaflet.handlers.movestart()
    resizeCallback([], {} as ResizeObserver)
    expect(leaflet.mapInstance.fitBounds).toHaveBeenCalledTimes(3)

    await wrapper.get("button").trigger("click")
    expect(leaflet.map).toHaveBeenCalledTimes(1)
    expect(leaflet.mapInstance.fitBounds).toHaveBeenCalledTimes(4)

    resizeCallback([], {} as ResizeObserver)
    expect(leaflet.mapInstance.fitBounds).toHaveBeenCalledTimes(5)
  })

  it("preserves filters across layers and resets the full toolbar", async () => {
    wrapper = mount(App)
    await flushPromises()
    const date = wrapper.get('[aria-label="日期"]')
    const start = wrapper.get('[aria-label="开始时间"]')
    const end = wrapper.get('[aria-label="结束时间"]')
    const aggregation = wrapper.get('[aria-label="聚合口径"]')
    const layer = wrapper.get('[aria-label="内容图层"]')

    await date.setValue("2020-12-23")
    await start.setValue("7")
    await end.setValue("9")
    await aggregation.setValue("sum")
    await layer.setValue("sequences")

    expect((start.element as HTMLInputElement).disabled).toBe(true)
    expect((end.element as HTMLInputElement).disabled).toBe(true)
    expect((aggregation.element as HTMLSelectElement).disabled).toBe(true)
    expect((start.element as HTMLInputElement).value).toBe("7")
    expect((end.element as HTMLInputElement).value).toBe("9")
    expect((aggregation.element as HTMLSelectElement).value).toBe("sum")

    await layer.setValue("flows")
    expect((start.element as HTMLInputElement).disabled).toBe(false)
    expect((aggregation.element as HTMLSelectElement).value).toBe("sum")

    await wrapper.get("button").trigger("click")
    expect((layer.element as HTMLSelectElement).value).toBe("source-sink")
    expect((date.element as HTMLSelectElement).value).toBe("clear-days")
    expect((start.element as HTMLInputElement).value).toBe("6")
    expect((end.element as HTMLInputElement).value).toBe("10")
    expect((aggregation.element as HTMLSelectElement).value).toBe("average")
  })

  it("keeps local geometry and attribution when the external tiles fail", async () => {
    wrapper = mount(App)
    await flushPromises()

    expect(leaflet.geoJSON).toHaveBeenCalledTimes(3)
    expect(leaflet.tileLayer).toHaveBeenCalledWith(
      expect.stringContaining("light_nolabels/{z}/{x}/{y}{r}.png?key=test-carto-key"),
      expect.objectContaining({ attribution: expect.stringContaining("OpenStreetMap") }),
    )
    expect(wrapper.text()).toContain("OpenStreetMap")
    expect(wrapper.text()).toContain("CARTO")

    const tileError = leaflet.tile.on.mock.calls.find(([event]) => event === "tileerror")?.[1]
    tileError()
    await wrapper.vm.$nextTick()

    expect(leaflet.mapInstance.removeLayer).toHaveBeenCalledWith(leaflet.tile)
    expect(wrapper.text()).toContain("外部底图不可用，本地业务地图仍可查看")
    expect(leaflet.geoJSON).toHaveBeenCalledTimes(3)
  })

  it("shows component health without blocking an available region map", async () => {
    vi.mocked(fetch).mockImplementation(async (input) => {
      if (input === "/api/health") {
        return {
          ok: false,
          json: async () => ({
            status: "unavailable",
            components: {
              database: { status: "ok" },
              extensions: { status: "ok" },
              dataset_release: { status: "ok" },
              island_boundary: { status: "ok" },
              sequences: { status: "unavailable" },
              digest_match: { status: "unavailable" },
            },
          }),
        } as Response
      }
      return { ok: true, json: async () => regionResponse } as Response
    })

    wrapper = mount(App)
    await flushPromises()

    expect(leaflet.map).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain("数据库：正常")
    expect(wrapper.text()).toContain("频繁区域序列：不可用")
    expect(wrapper.text()).toContain("组件状态：部分不可用")
  })

  it("shows a safe region error, retries, and accepts an empty release", async () => {
    const fetchMock = vi.mocked(fetch)
    fetchMock.mockImplementation(async (input) => {
      if (input === "/api/health") {
        return { ok: false, json: async () => ({ status: "unavailable" }) } as Response
      }
      return { ok: false, json: async () => ({ detail: "private database password" }) } as Response
    })

    wrapper = mount(App)
    expect(wrapper.text()).toContain("正在加载区域地图")
    await flushPromises()

    expect(wrapper.text()).toContain("区域地图暂不可用")
    expect(wrapper.text()).not.toContain("private database password")
    expect(wrapper.text()).toContain("组件状态：部分不可用")
    expect(leaflet.map).not.toHaveBeenCalled()

    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => regionResponse,
    } as Response)
    await wrapper.get(".map-status button").trigger("click")
    await flushPromises()

    expect(wrapper.text()).toContain("区域数据为空，共 0 个区域")
    expect(wrapper.text()).toContain("活动发布摘要")
    expect(wrapper.text()).toContain("a".repeat(64))
    expect(leaflet.map).toHaveBeenCalledTimes(1)
  })
})
