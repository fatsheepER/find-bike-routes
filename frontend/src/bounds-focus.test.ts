import { flushPromises, mount, type VueWrapper } from "@vue/test-utils"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import App from "./App.vue"

const leaflet = vi.hoisted(() => {
  const mapHandlers = new Map<string, (event: { latlng: { lat: number; lng: number } }) => void>()
  const mapInstance = {
    createPane: vi.fn(() => document.createElement("div")),
    dragging: { disable: vi.fn(), enable: vi.fn() },
    fitBounds: vi.fn(),
    getCenter: vi.fn(() => ({ lat: 24.1, lng: 118.1 })),
    getZoom: vi.fn(() => 12),
    invalidateSize: vi.fn(),
    on: vi.fn((event: string, handler: (event: { latlng: { lat: number; lng: number } }) => void) => {
      mapHandlers.set(event, handler)
    }),
    remove: vi.fn(),
    removeLayer: vi.fn(),
    setMaxBounds: vi.fn(),
    setView: vi.fn(),
  }
  const layer = { addTo: vi.fn(() => layer), on: vi.fn(() => layer) }
  const group = { addTo: vi.fn(() => group), clearLayers: vi.fn() }

  return {
    circleMarker: vi.fn(() => layer),
    divIcon: vi.fn((options) => options),
    geoJSON: vi.fn(() => layer),
    layerGroup: vi.fn(() => group),
    latLngBounds: vi.fn((southWest: number[], northEast: number[]) => [southWest, northEast]),
    map: vi.fn(() => mapInstance),
    mapHandlers,
    mapInstance,
    marker: vi.fn(() => layer),
    polyline: vi.fn(() => layer),
    rectangle: vi.fn(() => layer),
    tileLayer: vi.fn(() => ({ addTo: vi.fn(), on: vi.fn() })),
  }
})

vi.mock("leaflet", () => ({ default: leaflet }))

const regionResponse = {
  release_digest: "a".repeat(64),
  island_boundary: { type: "Feature", properties: {}, geometry: { type: "Point", coordinates: [118, 24] } },
  island_bounds: { west: 118, south: 24, east: 118.2, north: 24.2 },
  map_bounds: { west: 117.9, south: 23.9, east: 118.3, north: 24.3 },
  districts: { type: "FeatureCollection", features: [] },
  regions: { type: "FeatureCollection", features: [] },
}

let wrapper: VueWrapper | undefined

function trackCalls() {
  return vi.mocked(fetch).mock.calls.filter(([input]) => input === "/api/tracks/query")
}

async function drawBounds(start: [number, number], end: [number, number]) {
  await wrapper!.get('[aria-label="框选范围"]').trigger("click")
  leaflet.mapHandlers.get("mousedown")?.({ latlng: { lat: start[1], lng: start[0] } })
  leaflet.mapHandlers.get("mousemove")?.({ latlng: { lat: end[1], lng: end[0] } })
  leaflet.mapHandlers.get("mouseup")?.({ latlng: { lat: end[1], lng: end[0] } })
  await flushPromises()
}

beforeEach(() => {
  vi.clearAllMocks()
  leaflet.mapHandlers.clear()
  class ResizeObserverStub {
    observe = vi.fn()
    disconnect = vi.fn()
  }
  vi.stubGlobal("ResizeObserver", ResizeObserverStub)
  vi.stubGlobal("fetch", vi.fn(async (input, init) => {
    if (String(input) === "/api/regions") return { ok: true, status: 200, json: async () => regionResponse } as Response
    if (String(input) === "/api/health") {
      return { ok: true, status: 200, json: async () => ({ status: "ok", components: {} }) } as Response
    }
    const body = JSON.parse(String(init?.body)) as { start: string }
    return {
      ok: true,
      status: 200,
      json: async () => ({
        total_count: Number(body.start.slice(8, 10)),
        samples: { type: "FeatureCollection", features: [] },
      }),
    } as Response
  }))
})

afterEach(() => {
  wrapper?.unmount()
  wrapper = undefined
  vi.unstubAllGlobals()
})

describe("bounds focus", () => {
  it("queries an ordered in-range rectangle immediately and shows its bounds", async () => {
    wrapper = mount(App)
    await flushPromises()

    await drawBounds([118, 24], [118.1, 24.1])

    expect(trackCalls()).toHaveLength(4)
    expect(JSON.parse(String(trackCalls()[0][1]?.body))).toEqual(expect.objectContaining({
      selection: { type: "bounds", west: 118, south: 24, east: 118.1, north: 24.1 },
      sample_limit: 20,
    }))
    const panel = wrapper.get('[aria-label="矩形聚焦详情"]')
    expect(panel.text()).toContain("西 118.00000，南 24.00000，东 118.10000，北 24.10000")
    expect(panel.text()).toContain("局部时间06:00–10:00")
    expect(panel.text()).toContain("跨日平均")
    expect(panel.text()).toContain("唯一有效轨迹总数92")
    expect(panel.text()).toContain("实际样例数0")
    expect(panel.text()).toContain("样例最多 20 条")
    expect(panel.find('[aria-label="区域画像"]').exists()).toBe(false)
    expect(leaflet.rectangle).toHaveBeenCalled()
  })

  it.each([
    [[118.1, 24.1], [118, 24]],
    [[118, 24], [118, 24.1]],
    [[117.8, 24], [118.1, 24.1]],
    [[118, 24], [118.4, 24.1]],
  ] as Array<[[number, number], [number, number]]>)(
    "does not query a reversed, degenerate, or out-of-range rectangle",
    async (start, end) => {
      wrapper = mount(App)
      await flushPromises()

      await drawBounds(start, end)

      expect(trackCalls()).toHaveLength(0)
      expect(wrapper.find('[aria-label="矩形聚焦详情"]').exists()).toBe(false)
      expect(wrapper.get('[aria-label="框选范围"]').text()).toBe("框选范围")
      expect(leaflet.mapInstance.dragging.enable).toHaveBeenCalled()
    },
  )

  it("cancels drawing by button or Escape without changing analysis controls", async () => {
    wrapper = mount(App)
    await flushPromises()
    await wrapper.get('[aria-label="日期"]').setValue("2020-12-23")
    await wrapper.get('[aria-label="开始时间"]').setValue("7")

    await wrapper.get('[aria-label="框选范围"]').trigger("click")
    expect(wrapper.get('[aria-label="框选范围"]').text()).toBe("取消框选")
    expect((wrapper.get('[aria-label="内容图层"]').element as HTMLSelectElement).disabled).toBe(true)
    expect((wrapper.get('[aria-label="日期"]').element as HTMLSelectElement).disabled).toBe(true)
    await wrapper.get('[aria-label="框选范围"]').trigger("click")
    await wrapper.get('[aria-label="框选范围"]').trigger("click")
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }))
    await wrapper.vm.$nextTick()

    expect(trackCalls()).toHaveLength(0)
    expect(wrapper.get('[aria-label="框选范围"]').text()).toBe("框选范围")
    expect((wrapper.get('[aria-label="日期"]').element as HTMLSelectElement).value).toBe("2020-12-23")
    expect((wrapper.get('[aria-label="开始时间"]').element as HTMLInputElement).value).toBe("7")
  })

  it("uses one request for a single day and keeps a legal empty result", async () => {
    vi.mocked(fetch).mockImplementation(async (input, init) => {
      if (String(input) === "/api/regions") return { ok: true, status: 200, json: async () => regionResponse } as Response
      if (String(input) === "/api/health") {
        return { ok: true, status: 200, json: async () => ({ status: "ok", components: {} }) } as Response
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({ total_count: 0, samples: { type: "FeatureCollection", features: [] } }),
      } as Response
    })
    wrapper = mount(App)
    await flushPromises()
    await wrapper.get('[aria-label="日期"]').setValue("2020-12-23")
    await wrapper.get('[aria-label="开始时间"]').setValue("7")
    await wrapper.get('[aria-label="结束时间"]').setValue("9")

    await drawBounds([118, 24], [118.1, 24.1])

    expect(trackCalls()).toHaveLength(1)
    expect(JSON.parse(String(trackCalls()[0][1]?.body))).toEqual(expect.objectContaining({
      start: "2020-12-23T07:00:00+08:00",
      end: "2020-12-23T09:00:00+08:00",
    }))
    expect(wrapper.get('[aria-label="矩形聚焦详情"]').text()).toContain("共有 0 条唯一有效轨迹，样例为空")
  })
})
