import { flushPromises, mount, type VueWrapper } from "@vue/test-utils"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import App from "./App.vue"
import { leaflet, ResizeObserverStub, testRegionContext } from "./app-test-support"

vi.mock("leaflet", async () => ({ default: (await import("./app-test-support")).leaflet }))
vi.mock("echarts", () => ({ init: vi.fn(() => ({ dispose: vi.fn(), resize: vi.fn(), setOption: vi.fn() })) }))

const regions = testRegionContext()

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

let wrapper: VueWrapper | undefined

beforeEach(() => {
  vi.clearAllMocks()
  vi.stubEnv("VITE_CARTO_API_KEY", "test-key")
  leaflet.regionClicks.clear()
  vi.stubGlobal("ResizeObserver", ResizeObserverStub)
})

afterEach(() => {
  wrapper?.unmount()
  wrapper = undefined
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe("failure isolation", () => {
  it("blocks only on region failure and accepts an empty release after retry", async () => {
    let failRegions = true
    vi.stubGlobal("fetch", vi.fn(async (input) => {
      if (String(input) === "/api/health") return { ok: false, json: async () => ({ detail: "secret" }) } as Response
      return failRegions
        ? { ok: false, json: async () => ({ detail: "database password" }) } as Response
        : { ok: true, json: async () => ({ ...regions, regions: { type: "FeatureCollection", features: [] } }) } as Response
    }))
    wrapper = mount(App)
    expect(wrapper.text()).toContain("正在加载区域地图")
    expect(wrapper.text()).toContain("组件状态：检查中")
    await flushPromises()

    expect(wrapper.text()).toContain("区域地图暂不可用")
    expect(wrapper.text()).not.toContain("database password")
    expect(leaflet.map).not.toHaveBeenCalled()

    failRegions = false
    await wrapper.get(".map-status button").trigger("click")
    await flushPromises()
    expect(wrapper.text()).toContain("区域数据为空，共 0 个区域")
  })

  it("keeps the business map when health or tiles fail", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input) => String(input) === "/api/regions"
      ? { ok: true, json: async () => regions } as Response
      : { ok: false, json: async () => ({ detail: "internal host" }) } as Response))
    wrapper = mount(App)
    await flushPromises()

    expect(leaflet.map).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).toContain("组件状态：部分不可用")
    expect(wrapper.text()).not.toContain("internal host")
    const tileError = leaflet.tile.on.mock.calls.find(([event]) => event === "tileerror")?.[1]
    tileError()
    await wrapper.vm.$nextTick()
    expect(wrapper.text()).toContain("底图暂不可用，本地地图仍可查看")
    expect(leaflet.mapInstance.removeLayer).toHaveBeenCalledWith(leaflet.tile)
  })

  it("isolates flow, sequence, and track errors and lets each retry reach an empty result", async () => {
    const flow = deferred<Response>()
    const sequence = deferred<Response>()
    const track = deferred<Response>()
    let mode: "flow-loading" | "sequence-loading" | "track-loading" | "empty" = "flow-loading"
    vi.stubGlobal("fetch", vi.fn((input) => {
      const url = String(input)
      if (url === "/api/regions") return Promise.resolve({ ok: true, json: async () => regions } as Response)
      if (url === "/api/health") {
        return Promise.resolve({ ok: true, json: async () => ({ status: "ok", components: {} }) } as Response)
      }
      if (mode === "flow-loading") return flow.promise
      if (mode === "sequence-loading") return sequence.promise
      if (mode === "track-loading") return track.promise
      if (url.startsWith("/api/flows")) return Promise.resolve({ ok: true, json: async () => ({ flows: [] }) } as Response)
      if (url.startsWith("/api/sequences")) return Promise.resolve({ ok: true, json: async () => ({ patterns: [] }) } as Response)
      return Promise.resolve({
        ok: true,
        json: async () => ({ total_count: 0, samples: { type: "FeatureCollection", features: [] } }),
      } as Response)
    }))
    wrapper = mount(App)
    await flushPromises()

    await wrapper.get('[data-layer="flows"]').trigger("click")
    expect(wrapper.text()).toContain("正在加载区域流")
    flow.resolve({ ok: false, status: 503, json: async () => ({ detail: "private error" }) } as Response)
    await flushPromises()
    expect(wrapper.text()).toContain("区域流暂不可用，区域地图仍可查看")
    expect(leaflet.map).toHaveBeenCalledTimes(1)
    mode = "empty"
    await wrapper.get('[aria-label="区域间流动"] [role="alert"] button').trigger("click")
    await flushPromises()
    expect(wrapper.text()).toContain("当前条件下没有区域流")

    mode = "sequence-loading"
    await wrapper.get('[data-layer="sequences"]').trigger("click")
    expect(wrapper.text()).toContain("正在加载通勤链")
    sequence.resolve({ ok: false, status: 503, json: async () => ({ detail: "private" }) } as Response)
    await flushPromises()
    expect(wrapper.text()).toContain("通勤链暂不可用，区域地图仍可查看")
    mode = "empty"
    await wrapper.get('[aria-label="典型通勤链"] [role="alert"] button').trigger("click")
    await flushPromises()
    expect(wrapper.findAll(".sequence-list li")).toHaveLength(0)
    expect(wrapper.find(".sequence-panel [role=alert]").exists()).toBe(false)

    mode = "track-loading"
    leaflet.regionClicks.get(7)?.()
    await wrapper.vm.$nextTick()
    expect(wrapper.text()).toContain("正在查询有效轨迹")
    track.resolve({ ok: false, status: 503, json: async () => ({ detail: "private error" }) } as Response)
    await flushPromises()
    expect(wrapper.get('[aria-label="区域聚焦详情"] [role="alert"]').text()).toContain("轨迹查询依赖暂不可用")
    expect(wrapper.get('[aria-label="区域聚焦详情"]').text()).toContain("R-7")
    mode = "empty"
    await wrapper.get('[aria-label="区域聚焦详情"] [role="alert"] button').trigger("click")
    await flushPromises()
    expect(wrapper.get('[aria-label="区域聚焦详情"]').text()).toContain("共有 0 条唯一有效轨迹，样例为空")
    expect(wrapper.text()).not.toContain("private error")
  })
})
