import { selectDate } from "./app-test-support"
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
  const tile = { addTo: vi.fn(), on: vi.fn() }
  const geometryLayer = { addTo: vi.fn() }
  const sequenceGroup = { addTo: vi.fn(), clearLayers: vi.fn() }
  const sequenceObjects: Array<{
    addTo: ReturnType<typeof vi.fn>
    on: ReturnType<typeof vi.fn>
  }> = []
  const sequenceObject = () => {
    const object = {
      addTo: vi.fn(() => object),
      on: vi.fn(() => object),
    }
    sequenceObjects.push(object)
    return object
  }

  return {
    DomEvent: { stopPropagation: vi.fn() },
    circleMarker: vi.fn(sequenceObject),
    geoJSON: vi.fn(() => geometryLayer),
    latLngBounds: vi.fn((southWest: number[], northEast: number[]) => [southWest, northEast]),
    layerGroup: vi.fn(() => sequenceGroup),
    map: vi.fn(() => mapInstance),
    mapInstance,
    polyline: vi.fn(sequenceObject),
    sequenceGroup,
    sequenceObjects,
    tile,
    tileLayer: vi.fn(() => tile),
  }
})

vi.mock("leaflet", () => ({ default: leaflet }))

const point = (coordinates: [number, number]) => ({ type: "Point", coordinates })
const regionResponse = {
  release_digest: "a".repeat(64),
  island_boundary: {
    type: "Feature",
    properties: {},
    geometry: {
      type: "Polygon",
      coordinates: [[[118, 24], [118.3, 24], [118.3, 24.3], [118, 24.3], [118, 24]]],
    },
  },
  island_bounds: { west: 118, south: 24, east: 118.3, north: 24.3 },
  map_bounds: { west: 117.96, south: 23.96, east: 118.34, north: 24.34 },
  districts: {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { district_id: 1, label: "湖里" }, geometry: point([118.1, 24.1]) },
      { type: "Feature", properties: { district_id: 2, label: "思明" }, geometry: point([118.2, 24.2]) },
    ],
  },
  regions: {
    type: "FeatureCollection",
    features: [1, 2, 3].map((regionId) => ({
      type: "Feature",
      properties: {
        region_id: regionId,
        region_code: `R-${regionId}`,
        district_id: regionId === 3 ? 2 : 1,
        map_anchor: point([118 + regionId / 100, 24 + regionId / 100]),
        metrics: [],
      },
      geometry: point([118 + regionId / 100, 24 + regionId / 100]),
    })),
  },
}
const sequenceResponse = {
  release_digest: "a".repeat(64),
  region_cells_digest: "b".repeat(64),
  scope: "clear-days",
  min_contiguous_support: 0.001,
  min_contiguous_support_count: 5,
  valid_tracks: 1_000,
  limit: 20,
  patterns: [
    {
      region_ids: [1, 2, 3],
      region_codes: ["R-1", "R-2", "R-3"],
      district_ids: [1, 1, 2],
      length: 3,
      support: 8,
      contiguous_support: 6,
    },
    {
      region_ids: [2, 3],
      region_codes: ["R-2", "R-3"],
      district_ids: [1, 2],
      length: 2,
      support: 7,
      contiguous_support: 5,
    },
  ],
}
const scopes = [
  "clear-days",
  "2020-12-21",
  "2020-12-22",
  "2020-12-23",
  "2020-12-24",
  "2020-12-25",
]
const supportLevels = ["0.0002", "0.0005", "0.001", "0.002", "0.005", "0.01"]

let wrapper: VueWrapper | undefined

beforeEach(() => {
  vi.clearAllMocks()
  leaflet.sequenceObjects.length = 0
  class ResizeObserverStub {
    observe = vi.fn()
    disconnect = vi.fn()
    unobserve = vi.fn()
  }
  vi.stubGlobal("ResizeObserver", ResizeObserverStub)
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input) => ({
      ok: true,
      json: async () =>
        String(input).startsWith("/api/sequences")
          ? sequenceResponse
          : String(input) === "/api/regions"
            ? regionResponse
            : { status: "ok", components: {} },
    })),
  )
})

afterEach(() => {
  wrapper?.unmount()
  wrapper = undefined
  vi.unstubAllGlobals()
})

describe("commute sequence layer", () => {
  it("requests the default audited rung, selects the first result, and draws every chain", async () => {
    wrapper = mount(App)
    await flushPromises()
    await wrapper.get('[data-layer="sequences"]').trigger("click")
    await flushPromises()

    expect(fetch).toHaveBeenCalledWith(
      "/api/sequences?scope=clear-days&min_contiguous_support=0.001&limit=20",
      expect.anything(),
    )
    expect((wrapper.get('[aria-label="连续支持度门槛"]').element as HTMLSelectElement).value).toBe("0.001")
    expect((wrapper.get('[aria-label="Top-N"]').element as HTMLInputElement).value).toBe("20")
    expect(wrapper.get('[aria-current="true"]').text()).toContain("R-1 → R-2 → R-3")
    expect(wrapper.get(".sequence-details").text()).toContain("湖里 → 湖里 → 思明")
    expect(wrapper.get(".sequence-details").text()).toContain("支持度（区域序列条数）8")
    expect(wrapper.get(".sequence-details").text()).toContain("连续支持度6")
    expect(wrapper.get(".sequence-diagnostics").text()).toContain("绝对连续支持度门槛5")
    expect(wrapper.get(".sequence-diagnostics").text()).toContain("有效轨迹分母1000")
    expect(leaflet.polyline).toHaveBeenCalledTimes(2)
    expect(leaflet.circleMarker).toHaveBeenCalledTimes(5)
  })

  it.each(scopes)("passes the %s date scope unchanged", async (scope) => {
    wrapper = mount(App)
    await flushPromises()
    await wrapper.get('[data-layer="sequences"]').trigger("click")
    await selectDate(wrapper, scope)
    await flushPromises()

    expect(fetch).toHaveBeenLastCalledWith(
      `/api/sequences?scope=${scope}&min_contiguous_support=0.001&limit=20`,
      expect.anything(),
    )
  })

  it.each(supportLevels)("offers and passes the %s audited support rung", async (level) => {
    wrapper = mount(App)
    await flushPromises()
    await wrapper.get('[data-layer="sequences"]').trigger("click")
    await wrapper.get('[aria-label="连续支持度门槛"]').setValue(level)
    await flushPromises()

    expect(wrapper.findAll('[aria-label="连续支持度门槛"] option').map((option) => option.text())).toEqual(
      supportLevels,
    )
    expect(fetch).toHaveBeenLastCalledWith(
      `/api/sequences?scope=clear-days&min_contiguous_support=${level}&limit=20`,
      expect.anything(),
    )
  })

  it("clamps Top-N to integer limits before requesting", async () => {
    wrapper = mount(App)
    await flushPromises()
    await wrapper.get('[data-layer="sequences"]').trigger("click")
    const limit = wrapper.get('[aria-label="Top-N"]')

    await limit.setValue("0")
    await flushPromises()
    expect((limit.element as HTMLInputElement).value).toBe("1")
    expect(fetch).toHaveBeenLastCalledWith(expect.stringContaining("limit=1"), expect.anything())

    await limit.setValue("101")
    await flushPromises()
    expect((limit.element as HTMLInputElement).value).toBe("100")
    expect(fetch).toHaveBeenLastCalledWith(expect.stringContaining("limit=100"), expect.anything())

    await limit.setValue("12.8")
    await flushPromises()
    expect((limit.element as HTMLInputElement).value).toBe("12")
    expect(fetch).toHaveBeenLastCalledWith(expect.stringContaining("limit=12"), expect.anything())
  })

  it("hides time and aggregation while retaining their values for other layers", async () => {
    wrapper = mount(App)
    await flushPromises()
    await wrapper.get('[aria-label="开始时间"]').setValue('7')
    await wrapper.get('[aria-label="结束时间"]').setValue('9')
    await wrapper.get('[aria-label="聚合口径"]').setValue('sum')
    await wrapper.get('[data-layer="sequences"]').trigger('click')
    expect(wrapper.find('[aria-label="开始时间"]').exists()).toBe(false)
    expect(wrapper.find('[aria-label="聚合口径"]').exists()).toBe(false)
    await wrapper.get('[data-layer="source-sink"]').trigger('click')
    expect((wrapper.get('[aria-label="开始时间"]').element as HTMLInputElement).value).toBe('7')
    expect((wrapper.get('[aria-label="结束时间"]').element as HTMLInputElement).value).toBe('9')
    expect((wrapper.get('[aria-label="聚合口径"]').element as HTMLSelectElement).value).toBe('sum')
  })

  it("synchronizes list, chain, and step selection while stopping map click propagation", async () => {
    wrapper = mount(App)
    await flushPromises()
    await wrapper.get('[data-layer="sequences"]').trigger("click")
    await flushPromises()

    await wrapper.findAll(".sequence-list button")[1].trigger("click")
    expect(wrapper.get('[aria-current="true"]').text()).toContain("R-2 → R-3")
    expect(wrapper.get(".sequence-details").text()).toContain("支持度（区域序列条数）7")

    const lineClick = leaflet.polyline.mock.results.at(-2)?.value.on.mock.calls[0][1]
    const originalEvent = new Event("click")
    lineClick({ originalEvent })
    await wrapper.vm.$nextTick()
    expect(leaflet.DomEvent.stopPropagation).toHaveBeenCalledWith(originalEvent)
    expect(wrapper.get('[aria-current="true"]').text()).toContain("R-1 → R-2 → R-3")

    const stepClick = leaflet.circleMarker.mock.results.at(-1)?.value.on.mock.calls[0][1]
    stepClick({ originalEvent })
    await wrapper.vm.$nextTick()
    expect(wrapper.get('[aria-current="true"]').text()).toContain("R-2 → R-3")
  })

  it("keeps API support values unchanged under the global aggregation state", async () => {
    wrapper = mount(App)
    await flushPromises()
    await wrapper.get('[aria-label="聚合口径"]').setValue("sum")
    await wrapper.get('[data-layer="sequences"]').trigger("click")
    await flushPromises()

    expect(wrapper.get(".sequence-details").text()).toContain("支持度（区域序列条数）8")
    expect(wrapper.get(".sequence-details").text()).toContain("连续支持度6")
    expect(wrapper.get(".sequence-note").text()).toContain("同一条序列内重复出现只计一次")
    expect(wrapper.get(".sequence-note").text()).toContain("不受跨日平均或合计影响")
  })

  it("isolates sequence loading, empty, and retry errors from the region map", async () => {
    let sequenceAttempt = 0
    let finishFirstRequest: (response: Response) => void = () => undefined
    vi.mocked(fetch).mockImplementation(async (input) => {
      if (String(input).startsWith("/api/sequences")) {
        sequenceAttempt += 1
        if (sequenceAttempt === 1) {
          return new Promise<Response>((resolve) => {
            finishFirstRequest = resolve
          })
        }
        return {
          ok: true,
          json: async () => ({ ...sequenceResponse, patterns: [] }),
        } as Response
      }
      return {
        ok: true,
        json: async () =>
          String(input) === "/api/regions" ? regionResponse : { status: "ok", components: {} },
      } as Response
    })
    wrapper = mount(App)
    await flushPromises()
    await wrapper.get('[data-layer="sequences"]').trigger("click")
    expect(wrapper.text()).toContain("正在加载通勤链")
    const regionDrawCount = leaflet.geoJSON.mock.calls.length
    finishFirstRequest({ ok: false, json: async () => ({ detail: "secret" }) } as Response)
    await flushPromises()

    expect(wrapper.text()).toContain("通勤链暂不可用，区域地图仍可查看")
    expect(wrapper.text()).not.toContain("secret")
    expect(leaflet.geoJSON).toHaveBeenCalledTimes(regionDrawCount)
    expect(leaflet.map).toHaveBeenCalledTimes(1)

    await wrapper.get('.sequence-panel [role="alert"] button').trigger("click")
    await flushPromises()
    expect(wrapper.findAll(".sequence-list li")).toHaveLength(0)
    expect(wrapper.find(".sequence-panel [role=alert]").exists()).toBe(false)
    expect(leaflet.geoJSON).toHaveBeenCalledTimes(regionDrawCount)
    expect(leaflet.map).toHaveBeenCalledTimes(1)
  })
})
