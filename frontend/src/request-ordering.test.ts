import { flushPromises, mount, type VueWrapper } from "@vue/test-utils"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import App from "./App.vue"
import { leaflet, ResizeObserverStub, testRegionContext } from "./app-test-support"

vi.mock("leaflet", async () => ({ default: (await import("./app-test-support")).leaflet }))
vi.mock("echarts", () => ({ init: vi.fn(() => ({ dispose: vi.fn(), setOption: vi.fn() })) }))

function regionResponse(digest: string) {
  return testRegionContext([7], digest)
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

let wrapper: VueWrapper | undefined

beforeEach(() => {
  vi.clearAllMocks()
  leaflet.regionClicks.clear()
  vi.stubGlobal("ResizeObserver", ResizeObserverStub)
})

afterEach(() => {
  wrapper?.unmount()
  wrapper = undefined
  vi.unstubAllGlobals()
})

describe("request ordering", () => {
  it("aborts an older region retry and rejects its late response", async () => {
    const firstRetry = deferred<Response>()
    const secondRetry = deferred<Response>()
    let regionCalls = 0
    vi.stubGlobal("fetch", vi.fn((input, init) => {
      if (String(input) === "/api/health") {
        return Promise.resolve({ ok: true, json: async () => ({ status: "ok", components: {} }) } as Response)
      }
      regionCalls += 1
      if (regionCalls === 1) return Promise.resolve({ ok: false } as Response)
      const pending = regionCalls === 2 ? firstRetry : secondRetry
      ;(pending as typeof pending & { signal?: AbortSignal }).signal = init?.signal as AbortSignal
      return pending.promise
    }))
    wrapper = mount(App)
    await flushPromises()

    const retry = wrapper.get(".map-status button")
    const firstClick = retry.trigger("click")
    const secondClick = retry.trigger("click")
    await Promise.all([firstClick, secondClick])

    expect((firstRetry as typeof firstRetry & { signal: AbortSignal }).signal.aborted).toBe(true)
    secondRetry.resolve({ ok: true, json: async () => regionResponse("new") } as Response)
    await flushPromises()
    firstRetry.resolve({ ok: true, json: async () => regionResponse("old") } as Response)
    await flushPromises()

    expect(wrapper.text()).toContain("new")
    expect(wrapper.text()).not.toContain("old")
  })

  it("cancels superseded flow, sequence, and track requests so late results cannot write back", async () => {
    const flowRequests: Array<ReturnType<typeof deferred<Response>> & { signal?: AbortSignal }> = []
    const sequenceRequests: Array<ReturnType<typeof deferred<Response>> & { signal?: AbortSignal }> = []
    const trackRequests: Array<ReturnType<typeof deferred<Response>> & { signal?: AbortSignal }> = []
    vi.stubGlobal("fetch", vi.fn((input, init) => {
      const url = String(input)
      if (url === "/api/regions") {
        return Promise.resolve({ ok: true, json: async () => regionResponse("release") } as Response)
      }
      if (url === "/api/health") {
        return Promise.resolve({ ok: true, json: async () => ({ status: "ok", components: {} }) } as Response)
      }
      const requests = url.startsWith("/api/flows")
        ? flowRequests
        : url.startsWith("/api/sequences") ? sequenceRequests : trackRequests
      const request = deferred<Response>() as ReturnType<typeof deferred<Response>> & { signal?: AbortSignal }
      request.signal = init?.signal as AbortSignal
      requests.push(request)
      return request.promise
    }))
    wrapper = mount(App)
    await flushPromises()

    await wrapper.get('[aria-label="内容图层"]').setValue("flows")
    await wrapper.get('[aria-label="日期"]').setValue("2020-12-23")
    expect(flowRequests.slice(0, 4).every((request) => request.signal?.aborted)).toBe(true)
    flowRequests.slice(4).forEach((request) => request.resolve({
      ok: true,
      json: async () => ({ flows: [{
        matrix: "od", scope: "2020-12-23", hour: 6, from_region: 7, to_region: 8,
        weight: 1, observed: 1, is_tested: true, is_significant: true, gated: false, is_self_loop: false,
      }] }),
    } as Response))
    await flushPromises()
    flowRequests.slice(0, 4).forEach((request) => request.resolve({ ok: true, json: async () => ({ flows: [] }) } as Response))
    await flushPromises()
    expect(wrapper.text()).toContain("Top 50 区域流列表")

    const exitRequestStart = flowRequests.length
    await wrapper.get('[aria-label="流矩阵"]').setValue("channel")
    const exitRequests = flowRequests.slice(exitRequestStart)
    await wrapper.get('[aria-label="内容图层"]').setValue("source-sink")
    expect(exitRequests.every((request) => request.signal?.aborted)).toBe(true)
    exitRequests.forEach((request) => request.resolve({ ok: true, json: async () => ({ flows: [] }) } as Response))
    await flushPromises()
    expect(wrapper.find('[aria-label="区域间流动"]').exists()).toBe(false)

    await wrapper.get('[aria-label="内容图层"]').setValue("sequences")
    await wrapper.get('[aria-label="日期"]').setValue("2020-12-24")
    expect(sequenceRequests[0].signal?.aborted).toBe(true)
    sequenceRequests[1].resolve({ ok: true, json: async () => ({
      scope: "2020-12-24", min_contiguous_support: 0.001, min_contiguous_support_count: 1,
      valid_tracks: 1, limit: 20,
      patterns: [{ region_ids: [7], region_codes: ["NEW"], district_ids: [1], length: 1, support: 1, contiguous_support: 1 }],
    }) } as Response)
    await flushPromises()
    sequenceRequests[0].resolve({ ok: true, json: async () => ({ patterns: [] }) } as Response)
    await flushPromises()
    expect(wrapper.text()).toContain("NEW")

    leaflet.regionClicks.get(7)?.()
    await wrapper.vm.$nextTick()
    ;(wrapper.get('[aria-label="聚焦开始时间"]').element as HTMLInputElement).value = "7"
    await wrapper.get('[aria-label="聚焦开始时间"]').trigger("change")
    expect(trackRequests[0].signal?.aborted).toBe(true)
    trackRequests.slice(1).forEach((request) => request.resolve({
      ok: true,
      json: async () => ({ total_count: 2, samples: { type: "FeatureCollection", features: [] } }),
    } as Response))
    await flushPromises()
    trackRequests[0].resolve({
      ok: true,
      json: async () => ({ total_count: 99, samples: { type: "FeatureCollection", features: [] } }),
    } as Response)
    await flushPromises()
    expect(wrapper.get('[aria-label="区域聚焦详情"]').text()).toContain("唯一有效轨迹总数2")
  })
})
