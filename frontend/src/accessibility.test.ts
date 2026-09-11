import { flushPromises, mount, type VueWrapper } from "@vue/test-utils"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import App from "./App.vue"
import { leaflet, ResizeObserverStub, testRegionContext } from "./app-test-support"

vi.mock("leaflet", async () => ({ default: (await import("./app-test-support")).leaflet }))
vi.mock("echarts", () => ({ init: vi.fn(() => ({ dispose: vi.fn(), setOption: vi.fn() })) }))

const regions = testRegionContext([7, 8])

let wrapper: VueWrapper | undefined

beforeEach(() => {
  vi.clearAllMocks()
  vi.stubGlobal("ResizeObserver", ResizeObserverStub)
  vi.stubGlobal("fetch", vi.fn(async (input) => {
    const url = String(input)
    if (url === "/api/regions") return { ok: true, json: async () => regions } as Response
    if (url === "/api/health") return { ok: true, json: async () => ({ status: "ok", components: {} }) } as Response
    if (url.startsWith("/api/flows")) {
      return { ok: true, json: async () => ({ flows: [{
        matrix: "od", scope: "clear-days-stable", hour: 6, from_region: 7, to_region: 8,
        weight: 4, observed: 16, is_tested: true, is_significant: true, gated: false, is_self_loop: false,
      }] }) } as Response
    }
    if (url.startsWith("/api/sequences")) {
      return { ok: true, json: async () => ({
        scope: "clear-days", min_contiguous_support: 0.001, min_contiguous_support_count: 1,
        valid_tracks: 2, limit: 20,
        patterns: [{ region_ids: [7, 8], region_codes: ["R-7", "R-8"], district_ids: [1, 1], length: 2, support: 1, contiguous_support: 1 }],
      }) } as Response
    }
    return {
      ok: true,
      json: async () => ({ total_count: 0, samples: { type: "FeatureCollection", features: [] } }),
    } as Response
  }))
})

afterEach(() => {
  wrapper?.unmount()
  wrapper = undefined
  vi.unstubAllGlobals()
})

describe("basic accessibility", () => {
  it("gives every form control a visible label and a keyboard focus target", async () => {
    wrapper = mount(App, { attachTo: document.body })
    await flushPromises()
    const app = wrapper

    const checkControls = (names: string[]) => names.forEach((name) => {
      const control = app.get(`[aria-label="${name}"]`)
      expect(control.element.closest("label") ?? control.element.closest("fieldset")?.querySelector("legend")).not.toBeNull()
      ;(control.element as HTMLElement).focus()
      expect(document.activeElement).toBe(control.element)
    })
    checkControls(["内容图层", "日期", "开始时间", "结束时间", "聚合口径"])

    await wrapper.get('[aria-label="内容图层"]').setValue("flows")
    await flushPromises()
    checkControls(["流矩阵", "显著性"])
    await wrapper.get('[aria-label="内容图层"]').setValue("sequences")
    await flushPromises()
    checkControls(["连续支持度门槛", "Top-N"])

    await wrapper.get('[aria-label="区域地图等价操作"] button').trigger("click")
    await flushPromises()
    checkControls(["聚焦开始时间", "聚焦结束时间"])
  })

  it("provides keyboard equivalents for region, flow, and sequence map objects", async () => {
    wrapper = mount(App)
    await flushPromises()

    const regionActions = wrapper.get('[aria-label="区域地图等价操作"]')
    expect(regionActions.findAll("button").map((button) => button.text())).toEqual(["聚焦 R-7", "聚焦 R-8"])
    await regionActions.findAll("button")[0].trigger("click")
    await flushPromises()
    expect(wrapper.find('[aria-label="区域聚焦详情"]').exists()).toBe(true)
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }))
    await wrapper.vm.$nextTick()

    await wrapper.get('[aria-label="内容图层"]').setValue("flows")
    await flushPromises()
    expect(wrapper.get('[aria-label="Top 50 区域流列表"]').findAll("button")).toHaveLength(1)
    await wrapper.get('[aria-label="内容图层"]').setValue("sequences")
    await flushPromises()
    expect(wrapper.get('[aria-label="典型通勤链列表"]').findAll("button")).toHaveLength(1)
    expect(wrapper.get('[aria-label="框选范围"]').element.tagName).toBe("BUTTON")
  })

  it("supports Escape and supplies non-color source, sink, significance, and error cues", async () => {
    wrapper = mount(App)
    await flushPromises()

    expect(wrapper.get('[aria-label="净流入强度图例"]').text()).toContain("源 − ｜ 平衡 0 ｜ 汇 +")
    await wrapper.get('[aria-label="框选范围"]').trigger("click")
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }))
    await wrapper.vm.$nextTick()
    expect(wrapper.get('[aria-label="框选范围"]').text()).toBe("框选范围")

    await wrapper.get('[aria-label="内容图层"]').setValue("flows")
    await flushPromises()
    expect(wrapper.get('[aria-label="所选流对详情"]').text()).toContain("显著是")
    expect(wrapper.get('[aria-label="显著性"]').element.closest("label")?.textContent).toContain("显著性")

    vi.mocked(fetch).mockResolvedValue({ ok: false, json: async () => ({ detail: "private error" }) } as Response)
    await wrapper.get('[aria-label="流矩阵"]').setValue("channel")
    await flushPromises()
    expect(wrapper.get('[aria-label="区域间流动"] [role="alert"]').text()).toContain("区域流暂不可用")
    expect(wrapper.text()).not.toContain("private error")
  })
})
