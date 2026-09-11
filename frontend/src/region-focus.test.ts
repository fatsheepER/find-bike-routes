import { selectDate, currentDate, setFocusHour, focusHour } from "./app-test-support"
import { flushPromises, mount, type VueWrapper } from "@vue/test-utils"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import App from "./App.vue"

const leaflet = vi.hoisted(() => {
  const regionClicks = new Map<number, () => void>()
  const mapHandlers = new Map<string, () => void>()
  const mapInstance = {
    createPane: vi.fn(() => document.createElement("div")),
    fitBounds: vi.fn(),
    getCenter: vi.fn(() => ({ lat: 24.1, lng: 118.1 })),
    getZoom: vi.fn(() => 12),
    invalidateSize: vi.fn(),
    on: vi.fn((event: string, handler: () => void) => mapHandlers.set(event, handler)),
    remove: vi.fn(),
    removeLayer: vi.fn(),
    setMaxBounds: vi.fn(),
    setView: vi.fn(() => mapHandlers.get("movestart")?.()),
  }
  const tile = { addTo: vi.fn(), on: vi.fn(), setOpacity: vi.fn() }
  const focusGroup = { addTo: vi.fn(), clearLayers: vi.fn() }
  focusGroup.addTo.mockReturnValue(focusGroup)
  const geometryLayer = { addTo: vi.fn(() => geometryLayer) }
  const markerLayer = { addTo: vi.fn(() => markerLayer) }
  const sequenceObjects: Array<{ addTo: ReturnType<typeof vi.fn>; on: ReturnType<typeof vi.fn> }> = []
  const sequenceObject = () => {
    const object = {
      addTo: vi.fn(() => object),
      on: vi.fn(() => object),
    }
    sequenceObjects.push(object)
    return object
  }

  return {
    divIcon: vi.fn((options) => options),
    focusGroup,
    geoJSON: vi.fn((data: { features?: Array<{ properties?: Record<string, unknown>; geometry?: unknown }> }, options?: {
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
      return geometryLayer
    }),
    layerGroup: vi.fn(() => focusGroup),
    latLngBounds: vi.fn((southWest: number[], northEast: number[]) => [southWest, northEast]),
    map: vi.fn(() => mapInstance),
    mapHandlers,
    mapInstance,
    marker: vi.fn(() => markerLayer),
    circleMarker: vi.fn(sequenceObject),
    polyline: vi.fn(sequenceObject),
    regionClicks,
    sequenceObjects,
    tile,
    tileLayer: vi.fn(() => tile),
  }
})

vi.mock("leaflet", () => ({ default: leaflet }))

const metric = (date: string, hour: number) => {
  const value = hour - 5
  return {
  date,
  hour,
  unlocks: value,
  locks: value * 3,
  net_inflow: value * 2,
  net_inflow_per_km2: value,
  order_events_per_km2: value * 2,
  tracks_visiting: value * 8,
  tracks_transit: value * 2,
  pi_r: 0.25,
  chords: value * 2,
  sum_cos: value,
  sum_sin: 0,
  sum_cos2: value,
  sum_sin2: 0,
  r: 0.5,
  r_axial: 0.5,
  mean_bearing_deg: 0,
  axis_bearing_deg: 0,
  sectors: [1, ...Array(15).fill(0)],
  }
}

const region = {
  type: "Feature",
  id: 7,
  geometry: {
    type: "Polygon",
    coordinates: [[[118, 24], [118.1, 24], [118.1, 24.1], [118, 24.1], [118, 24]]],
  },
  properties: {
    region_id: 7,
    region_code: "思明-7",
    district_id: 2,
    cells: 3,
    area_km2: 2,
    functional_composition: { residential: 0.1, employment: 0.2, education: 0.3, transport: 0.4 },
    classified_share: 0.75,
    bus_stops_per_km2: 6,
    map_anchor: { type: "Point", coordinates: [118.05, 24.05] },
    metrics: ["2020-12-21", "2020-12-22", "2020-12-23", "2020-12-24", "2020-12-25"]
      .flatMap((date) => [6, 7, 8, 9].map((hour) => metric(date, hour))),
  },
}

const regionResponse = {
  release_digest: "a".repeat(64),
  island_boundary: region,
  island_bounds: { west: 118, south: 24, east: 118.2, north: 24.2 },
  map_bounds: { west: 117.9, south: 23.9, east: 118.3, north: 24.3 },
  districts: {
    type: "FeatureCollection",
    features: [{ type: "Feature", geometry: region.geometry, properties: { district_id: 2, label: "思明片区" } }],
  },
  regions: { type: "FeatureCollection", features: [region] },
}

let wrapper: VueWrapper | undefined
let resizeCallback: ResizeObserverCallback

function trackFeature(trackId: string) {
  return {
    type: "Feature",
    id: trackId,
    geometry: { type: "LineString", coordinates: [[117.8, 23.8], [118.2, 24.2]] },
    properties: { track_id: trackId },
  }
}

function trackCalls() {
  return vi.mocked(fetch).mock.calls.filter(([input]) => input === "/api/tracks/query")
}

async function mountApp() {
  wrapper = mount(App)
  await flushPromises()
  return wrapper
}

async function enterFocus() {
  leaflet.regionClicks.get(7)?.()
  await flushPromises()
}

beforeEach(() => {
  vi.clearAllMocks()
  leaflet.regionClicks.clear()
  leaflet.mapHandlers.clear()
  leaflet.sequenceObjects.length = 0
  class ResizeObserverStub {
    constructor(callback: ResizeObserverCallback) {
      resizeCallback = callback
    }
    observe = vi.fn()
    disconnect = vi.fn()
  }
  vi.stubGlobal("ResizeObserver", ResizeObserverStub)
  vi.stubGlobal("fetch", vi.fn(async (input, init) => {
    if (String(input) === "/api/regions") return { ok: true, status: 200, json: async () => regionResponse } as Response
    if (String(input) === "/api/health") {
      return { ok: true, status: 200, json: async () => ({ status: "ok", components: {} }) } as Response
    }
    if (String(input).startsWith("/api/sequences")) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          scope: "clear-days",
          min_contiguous_support: 0.001,
          min_contiguous_support_count: 1,
          valid_tracks: 10,
          limit: 20,
          patterns: ["first", "second"].map((name, index) => ({
            region_ids: [7],
            region_codes: [name],
            district_ids: [2],
            length: 1,
            support: 2 - index,
            contiguous_support: 2 - index,
          })),
        }),
      } as Response
    }
    const body = JSON.parse(String(init?.body)) as { start: string }
    const date = body.start.slice(0, 10)
    const day = Number(date.slice(-2))
    return {
      ok: true,
      status: 200,
      json: async () => ({
        release_digest: "a".repeat(64),
        total_count: day,
        samples: {
          type: "FeatureCollection",
          features: [5, 1, 4, 2, 3, 0].map((id) => trackFeature(`${date}-${id}`)),
        },
      }),
    } as Response
  }))
})

afterEach(() => {
  wrapper?.unmount()
  wrapper = undefined
  vi.unstubAllGlobals()
})

describe("region focus", () => {
  it("enters immediately from a region click and locks the background controls", async () => {
    wrapper = await mountApp()
    await enterFocus()

    expect(wrapper.get('[aria-label="区域聚焦详情"]').text()).toContain("思明-7")
    expect((wrapper.get('[role="tab"][aria-selected="true"]').element as HTMLButtonElement).disabled).toBe(true)
    expect(fetch).toHaveBeenCalledWith(
      "/api/tracks/query",
      expect.objectContaining({ method: "POST" }),
    )
    expect(trackCalls()).toHaveLength(4)
    expect(wrapper.get('[aria-label="区域聚焦详情"]').text()).toContain("唯一有效轨迹总数92")
    expect(wrapper.get('[aria-label="区域聚焦详情"]').text()).toContain("实际样例数20")
    expect(wrapper.get('[aria-label="区域聚焦详情"]').text()).toContain("样例最多 20 条")
    const samples = leaflet.geoJSON.mock.calls.find(([data]) => data.features?.length === 20)?.[0].features
    expect(samples?.map((feature) => feature.properties?.request_date)).toEqual([
      ...Array(6).fill("2020-12-21"),
      ...Array(6).fill("2020-12-22"),
      ...Array(6).fill("2020-12-24"),
      ...Array(2).fill("2020-12-25"),
    ])
    expect(samples?.[0].properties?.track_id).toBe("2020-12-21-0")
    expect(samples?.[0].geometry).toEqual(trackFeature("x").geometry)
    expect(leaflet.marker).toHaveBeenCalledWith(
      [24.05, 118.05],
      expect.objectContaining({ icon: expect.objectContaining({ html: "<span>92</span>" }) }),
    )
    await wrapper.get('[aria-label="区域聚焦详情"] .focus-heading button').trigger("click")
    expect(wrapper.find('[aria-label="区域聚焦详情"]').exists()).toBe(false)
  })

  it("uses the full study window when focus starts from the commute-sequence layer", async () => {
    wrapper = await mountApp()
    await wrapper.get('[aria-label="开始时间"]').setValue("7")
    await wrapper.get('[aria-label="结束时间"]').setValue("9")
    await wrapper.get('[data-layer="sequences"]').trigger("click")
    await flushPromises()
    await wrapper.findAll(".sequence-list button")[1].trigger("click")

    await enterFocus()

    expect(focusHour(wrapper, "start")).toBe("6")
    expect(focusHour(wrapper, "end")).toBe("10")
    expect(JSON.parse(String(trackCalls()[0][1]?.body))).toEqual(expect.objectContaining({
      start: "2020-12-21T06:00:00+08:00",
      end: "2020-12-21T10:00:00+08:00",
    }))
    const firstSequenceClick = leaflet.sequenceObjects[0].on.mock.calls[0][1]
    firstSequenceClick()
    await wrapper.get('[aria-label="区域聚焦详情"] .focus-heading button').trigger("click")

    expect(wrapper.get('[role="tab"][aria-selected="true"]').attributes("data-layer")).toBe("sequences")
    expect(wrapper.get('.sequence-list [aria-current="true"]').text()).toBe("second")
  })

  it("copies a single-day scope and changes only the local focus time", async () => {
    wrapper = await mountApp()
    await selectDate(wrapper, "2020-12-23")
    await wrapper.get('[aria-label="开始时间"]').setValue("7")
    await wrapper.get('[aria-label="结束时间"]').setValue("9")
    await wrapper.get('[aria-label="聚合口径"]').setValue("sum")

    await enterFocus()

    expect(trackCalls()).toHaveLength(1)
    expect(JSON.parse(String(trackCalls()[0][1]?.body))).toEqual({
      selection: { type: "region", region_id: 7 },
      start: "2020-12-23T07:00:00+08:00",
      end: "2020-12-23T09:00:00+08:00",
      sample_limit: 20,
    })
    expect(wrapper.get('[aria-label="区域聚焦详情"]').text()).toContain("2020-12-23")
    expect(wrapper.get('[aria-label="区域聚焦详情"]').text()).toContain("跨日合计")
    expect(wrapper.get('[aria-label="区域画像"]').text()).toContain("解锁5")

    await setFocusHour(wrapper, "8")
    await flushPromises()

    expect(trackCalls()).toHaveLength(2)
    expect(JSON.parse(String(trackCalls()[1][1]?.body)).start).toBe("2020-12-23T08:00:00+08:00")
    expect(wrapper.get('[aria-label="区域画像"]').text()).toContain("解锁3")
    expect((wrapper.get('[aria-label="开始时间"]').element as HTMLInputElement).value).toBe("7")
    expect((wrapper.get('[aria-label="聚合口径"]').element as HTMLSelectElement).value).toBe("sum")
  })

  it("shows the complete static and dynamic region profile", async () => {
    wrapper = await mountApp()
    await enterFocus()

    expect(wrapper.get('.focus-heading').text()).toContain('思明片区 · 2 km²')
    const profile = wrapper.get('[aria-label="区域画像"]').text()
    for (const text of [
      "解锁", "上锁", "净流入", "净流入强度",
      "订单事件密度", "访问轨迹", "过境轨迹", "过境率", "过境弦", "方向集中度0.5",
      "轴向集中度0.5", "方向角", "方向分布", "住宅10%", "已分类面积75%", "公交站密度6",
    ]) expect(profile).toContain(text)
  })

  it("shows a legal empty result without leaving focus", async () => {
    vi.mocked(fetch).mockImplementation(async (input) => ({
      ok: true,
      status: 200,
      json: async () => String(input) === "/api/regions"
        ? regionResponse
        : String(input) === "/api/health"
          ? { status: "ok", components: {} }
          : { total_count: 0, samples: { type: "FeatureCollection", features: [] } },
    } as Response))
    wrapper = await mountApp()
    await enterFocus()

    expect(wrapper.get('[aria-label="区域聚焦详情"]').text()).toContain("唯一有效轨迹总数0")
    expect(wrapper.get('[aria-label="区域聚焦详情"]').text()).toContain("共有 0 条唯一有效轨迹，样例为空")
  })

  it("clears old results while loading and ignores a late response", async () => {
    const finishes: Array<(response: Response) => void> = []
    vi.mocked(fetch).mockImplementation(async (input) => {
      if (String(input) === "/api/regions") return { ok: true, status: 200, json: async () => regionResponse } as Response
      if (String(input) === "/api/health") {
        return { ok: true, status: 200, json: async () => ({ status: "ok", components: {} }) } as Response
      }
      return new Promise<Response>((resolve) => finishes.push(resolve))
    })
    wrapper = await mountApp()
    await selectDate(wrapper, "2020-12-23")
    leaflet.regionClicks.get(7)?.()
    await wrapper.vm.$nextTick()
    expect(finishes).toHaveLength(1)

    finishes[0]({
      ok: true,
      status: 200,
      json: async () => ({ total_count: 9, samples: { type: "FeatureCollection", features: [trackFeature("old")] } }),
    } as Response)
    await flushPromises()
    expect(wrapper.get('[aria-label="区域聚焦详情"]').text()).toContain("唯一有效轨迹总数9")

    await setFocusHour(wrapper, "7")
    await wrapper.vm.$nextTick()
    expect(wrapper.get('[aria-label="区域聚焦详情"]').text()).toContain("唯一有效轨迹总数—")
    expect(finishes).toHaveLength(2)

    await setFocusHour(wrapper, "8")
    await wrapper.vm.$nextTick()
    expect(finishes).toHaveLength(3)
    finishes[2]({
      ok: true,
      status: 200,
      json: async () => ({ total_count: 2, samples: { type: "FeatureCollection", features: [trackFeature("new")] } }),
    } as Response)
    await flushPromises()
    finishes[1]({
      ok: true,
      status: 200,
      json: async () => ({ total_count: 7, samples: { type: "FeatureCollection", features: [trackFeature("late")] } }),
    } as Response)
    await flushPromises()

    expect(wrapper.get('[aria-label="区域聚焦详情"]').text()).toContain("唯一有效轨迹总数2")
    const latestSamples = leaflet.geoJSON.mock.calls.filter(([data]) => data.features?.length === 1).at(-1)?.[0].features
    expect(latestSamples?.[0].properties?.track_id).toBe("new")
  })

  it.each([
    [422, "输入范围无效"],
    [404, "所选区域不存在"],
    [503, "轨迹查询依赖暂不可用"],
    [500, "轨迹查询服务暂不可用"],
  ])("keeps focus and safely retries a %i response", async (status, message) => {
    let attempts = 0
    vi.mocked(fetch).mockImplementation(async (input) => {
      if (String(input) === "/api/regions") return { ok: true, status: 200, json: async () => regionResponse } as Response
      if (String(input) === "/api/health") {
        return { ok: true, status: 200, json: async () => ({ status: "ok", components: {} }) } as Response
      }
      attempts += 1
      return attempts <= 4
        ? { ok: false, status, json: async () => ({ detail: "private DSN and stack" }) } as Response
        : { ok: true, status: 200, json: async () => ({ total_count: 1, samples: { type: "FeatureCollection", features: [] } }) } as Response
    })
    wrapper = await mountApp()
    await enterFocus()

    expect(wrapper.get('[aria-label="区域聚焦详情"] [role="alert"]').text()).toContain(message)
    expect(wrapper.text()).not.toContain("private DSN and stack")
    expect(wrapper.get('[aria-label="区域聚焦详情"]').text()).toContain("思明-7")

    await wrapper.get('[aria-label="区域聚焦详情"] [role="alert"] button').trigger("click")
    await flushPromises()
    expect(wrapper.get('[aria-label="区域聚焦详情"]').text()).toContain("唯一有效轨迹总数4")
  })

  it("clears focus with Escape and restores the exact analysis state and map view", async () => {
    wrapper = await mountApp()
    await selectDate(wrapper, "2020-12-23")
    await wrapper.get('[aria-label="开始时间"]').setValue("7")
    await wrapper.get('[aria-label="结束时间"]').setValue("9")
    await wrapper.get('[aria-label="聚合口径"]').setValue("sum")
    await enterFocus()
    await setFocusHour(wrapper, "8")
    await flushPromises()
    leaflet.mapHandlers.get("movestart")?.()
    const fitsBeforeExit = leaflet.mapInstance.fitBounds.mock.calls.length

    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }))
    await wrapper.vm.$nextTick()
    resizeCallback([], {} as ResizeObserver)

    expect(wrapper.find('[aria-label="区域聚焦详情"]').exists()).toBe(false)
    expect(currentDate(wrapper)).toBe("2020-12-23")
    expect((wrapper.get('[aria-label="开始时间"]').element as HTMLInputElement).value).toBe("7")
    expect((wrapper.get('[aria-label="结束时间"]').element as HTMLInputElement).value).toBe("9")
    expect((wrapper.get('[aria-label="聚合口径"]').element as HTMLSelectElement).value).toBe("sum")
    expect(leaflet.focusGroup.clearLayers).toHaveBeenCalled()
    expect(leaflet.mapInstance.setView).toHaveBeenCalledWith(
      { lat: 24.1, lng: 118.1 },
      12,
      { animate: false },
    )
    expect(leaflet.mapInstance.fitBounds).toHaveBeenCalledTimes(fitsBeforeExit + 1)
  })
})
