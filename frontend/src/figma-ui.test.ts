import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import App from './App.vue'
import fogIcon from './assets/symbols/cloud.fog.svg'
import RangeControl from './RangeControl.vue'
import { leaflet, ResizeObserverStub, testRegionContext } from './app-test-support'
import { datesForScope } from './region-aggregation'
import { aggregateFlows, flowRequests } from './flow-aggregation'

vi.mock('leaflet', async () => ({ default: (await import('./app-test-support')).leaflet }))
vi.mock('echarts', () => ({ init: vi.fn(() => ({ dispose: vi.fn(), resize: vi.fn(), setOption: vi.fn() })) }))
let app: VueWrapper | undefined
beforeEach(() => {
  vi.clearAllMocks()
  vi.stubGlobal('ResizeObserver', ResizeObserverStub)
  Object.defineProperties(HTMLDialogElement.prototype, {
    showModal: { configurable: true, value: function (this: HTMLDialogElement) { this.open = true } },
    close: { configurable: true, value: function (this: HTMLDialogElement) { this.open = false } },
  })
  vi.stubGlobal('fetch', vi.fn(async (url) => ({ ok: true, json: async () =>
    url === '/api/regions' ? testRegionContext() : url === '/api/health' ? { status: 'ok', components: {} }
      : String(url).startsWith('/api/sequences') ? { patterns: [] }
      : { total_count: 0, samples: { type: 'FeatureCollection', features: [] } },
  } as Response)))
})
afterEach(() => { app?.unmount(); app = undefined; vi.restoreAllMocks(); vi.unstubAllGlobals() })

it('selects a continuous date range including rain and restores the unfilled clear-day state', async () => {
  app = mount(App)
  await flushPromises()
  expect(app.findAll('[role="tab"]')).toHaveLength(3)
  expect(app.find('h1').exists()).toBe(false)
  expect(app.find('footer').exists()).toBe(false)
  expect(app.get('.date-selector').find('.range-track span').exists()).toBe(false)
  await app.get('[aria-label="结束日期"]').setValue('2')
  expect(app.get('.date-selector').find('.range-track span').exists()).toBe(true)
  expect(app.get('.detail-body').text()).toContain('12/21、12/22、12/23')
  expect(app.get('.detail-body').text()).toContain('3 日平均')
  await app.findAll('button').find(button => button.text() === '重置为晴天集')!.trigger('click')
  expect(app.find('.restore-button').exists()).toBe(false)
  expect(app.get('.date-selector').find('.range-track span').exists()).toBe(false)
  expect(datesForScope('2020-12-21..2020-12-23')).toEqual(['2020-12-21', '2020-12-22', '2020-12-23'])
})

it('clamps the two ends to a nonempty hourly interval on one track', async () => {
  app = mount(RangeControl, { props: { min: 6, max: 10, start: 7, end: 9, gap: 1, startLabel: '开始', endLabel: '结束' } })
  expect(app.findAll('.range-track')).toHaveLength(1)
  await app.get('[aria-label="开始"]').setValue('10')
  expect(app.emitted('change')?.at(-1)).toEqual([8, 9])
  expect((app.get('[aria-label="开始"]').element as HTMLInputElement).value).toBe('8')
  await app.get('[aria-label="结束"]').setValue('6')
  expect(app.emitted('change')?.at(-1)).toEqual([7, 8])
})

it('queries every selected flow date and divides by the selected day count', () => {
  const requests = flowRequests('od', 'all', '2020-12-22..2020-12-23', 7, 9)
  expect(requests.map(item => item.url)).toEqual([
    '/api/flows?matrix=od&hour=7&date=2020-12-22', '/api/flows?matrix=od&hour=8&date=2020-12-22',
    '/api/flows?matrix=od&hour=7&date=2020-12-23', '/api/flows?matrix=od&hour=8&date=2020-12-23',
  ])
  const slices = requests.map(({ scope, hour }) => ({ scope, hour, rows: [{
    scope, hour, matrix: 'od' as const, from_region: 1, to_region: 2, weight: 6, observed: 6,
    is_tested: true, is_significant: true, gated: false, is_self_loop: false,
  }] }))
  expect(aggregateFlows(slices, { dateScope: '2020-12-22..2020-12-23', aggregation: 'average', significance: 'all' })[0].weight).toBe(12)
  expect(aggregateFlows(slices, { dateScope: '2020-12-22..2020-12-23', aggregation: 'sum', significance: 'all' })[0].weight).toBe(24)
})

it('enters sequences with an untouched single-date slider and resets on every entry', async () => {
  app = mount(App)
  await flushPromises()
  expect(app.findAll('[role="tab"]').map(tab => tab.attributes('data-layer'))).toEqual(['source-sink', 'flows', 'sequences'])
  await app.get('[aria-label="结束日期"]').setValue('2')
  await app.get('[data-layer="sequences"]').trigger('click')
  await flushPromises()
  expect(app.find('.sequence-date.no-date').exists()).toBe(true)
  expect(app.find('.date-capsule .restore-button').exists()).toBe(false)
  expect(app.findAll('.date-selector input[type="range"]')).toHaveLength(1)
  expect(app.get('.sequence-panel').text()).not.toContain('没有通勤链')
  expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).startsWith('/api/sequences?scope=clear-days'))).toBe(true)
  const slider = app.get('[aria-label="通勤链日期"]')
  Object.defineProperty(slider.element, 'getBoundingClientRect', { value: () => ({ left: 0, width: 208 }) })
  slider.element.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, clientX: 104 }))
  await flushPromises()
  expect(slider.attributes('aria-valuetext')).toBe('2020-12-23')
  expect(app.find('.sequence-date.no-date').exists()).toBe(false)
  expect(app.find('.date-capsule .restore-button').exists()).toBe(true)
  expect(String(vi.mocked(fetch).mock.calls.at(-1)?.[0])).toContain('scope=2020-12-23')
  await app.get('.date-capsule .restore-button').trigger('click')
  expect(app.find('.sequence-date.no-date').exists()).toBe(true)
  await slider.setValue('1')
  await app.get('[data-layer="flows"]').trigger('click')
  await app.get('[data-layer="sequences"]').trigger('click')
  expect(app.find('.sequence-date.no-date').exists()).toBe(true)
  expect(app.get('.detail-panel').find('.sequence-panel').exists()).toBe(true)
  await app.get('[aria-controls="layer-details"]').trigger('click')
  expect(app.get('.sequence-panel').isVisible()).toBe(false)
})

it('collapses the legend and opens a dismissible info dialog without exiting a selected region', async () => {
  app = mount(App, { attachTo: document.body })
  await flushPromises()
  await app.get('[aria-controls="layer-details"]').trigger('click')
  expect(app.get('#layer-details').isVisible()).toBe(false)
  expect(app.get('[aria-controls="layer-details"]').text()).toBe('展开')
  leaflet.regionClicks.get(7)?.()
  await flushPromises()
  await app.get('[aria-label="应用信息"]').trigger('click')
  expect((app.get('dialog').element as HTMLDialogElement).open).toBe(true)
  window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
  await flushPromises()
  expect((app.get('dialog').element as HTMLDialogElement).open).toBe(false)
  expect(app.find('.focus-panel').exists()).toBe(true)
  expect(document.activeElement).toBe(app.get('[aria-label="应用信息"]').element)
})

it('reserves right-side map space, brings eastern selection into view, and restores the saved view', async () => {
  app = mount(App)
  Object.defineProperties(app.get('#map').element, { clientWidth: { value: 1200 }, clientHeight: { value: 800 } })
  await flushPromises()
  const initialEast = leaflet.mapInstance.setMaxBounds.mock.calls.at(-1)![0][1][1]
  leaflet.regionClicks.get(7)?.()
  await flushPromises()
  expect(leaflet.mapInstance.setMaxBounds.mock.calls.at(-1)![0][1][1]).toBeGreaterThan(initialEast)
  expect(leaflet.mapInstance.panInside).toHaveBeenCalledWith([24.007, 118.007], expect.objectContaining({ paddingBottomRight: [420, 60] }))
  await app.get('[aria-label="本岛居中"]').trigger('click')
  expect(leaflet.mapInstance.fitBounds).toHaveBeenLastCalledWith(expect.anything(), expect.objectContaining({ paddingBottomRight: [412, 210] }))
  await app.get('.focus-heading button').trigger('click')
  expect(leaflet.mapInstance.setView).toHaveBeenCalledWith({ lat: 24, lng: 118 }, 12, { animate: false })
  await app.get('[aria-label="放大地图"]').trigger('click')
  expect(leaflet.mapInstance.setZoom).toHaveBeenCalledWith(13)
})

it('lets coincident date handles expand in either direction and responds to track clicks', async () => {
  app = mount(RangeControl, { props: { min: 0, max: 4, start: 2, end: 2, startLabel: '开始', endLabel: '结束' } })
  Object.defineProperty(app.element, 'getBoundingClientRect', { value: () => ({ left: 0, width: 208 }) })
  app.element.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, clientX: 104 }))
  app.element.dispatchEvent(new MouseEvent('pointermove', { bubbles: true, clientX: 54 }))
  expect(app.emitted('change')?.at(-1)).toEqual([1, 2])
  app.element.dispatchEvent(new MouseEvent('pointerup', { bubbles: true }))
  app.element.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, clientX: 154 }))
  expect(app.emitted('change')?.at(-1)).toEqual([2, 3])
})

it('switches focused regions, highlights the district, and restores the original snapshot', async () => {
  const context = testRegionContext([7, 8])
  const district = { type: 'Feature', geometry: context.regions.features[0].geometry, properties: { district_id: 1, label: '片区一' } }
  vi.mocked(fetch).mockImplementation(async (url) => ({ ok: true, json: async () =>
    url === '/api/regions' ? { ...context, districts: { type: 'FeatureCollection', features: [district] } }
      : url === '/api/health' ? { status: 'ok', components: {} }
      : { total_count: 0, samples: { type: 'FeatureCollection', features: [] } },
  } as Response))
  app = mount(App)
  await flushPromises()
  leaflet.regionClicks.get(7)?.()
  await flushPromises()
  leaflet.mapInstance.getCenter.mockReturnValue({ lat: 25, lng: 119 })
  leaflet.mapInstance.getZoom.mockReturnValue(14)
  leaflet.regionClicks.get(8)?.()
  await flushPromises()
  expect(app.get('.focus-heading').text()).toContain('R-8')
  expect(app.get('.focus-heading').text()).toContain('片区一 · 1 km²')
  expect(leaflet.geoJSON).toHaveBeenCalledWith(district, expect.objectContaining({ interactive: false, style: expect.objectContaining({ dashArray: '7 6', fill: false }) }))
  expect(JSON.parse(String(vi.mocked(fetch).mock.calls.at(-1)?.[1]?.body)).selection.region_id).toBe(8)
  leaflet.mapInstance.getCenter.mockReturnValue({ lat: 24, lng: 118 })
  leaflet.mapInstance.getZoom.mockReturnValue(12)
  await app.get('.focus-heading button').trigger('click')
  expect(leaflet.mapInstance.setView).toHaveBeenCalledWith({ lat: 24, lng: 118 }, 12, { animate: false })
})

it('uses the source weather conditions and collapses flow contents in the left panel', async () => {
  app = mount(App)
  await flushPromises()
  expect(app.findAll('.weather-symbol').map(icon => icon.attributes('alt'))).toEqual(['阴', '阴转晴', '小雨、阵雨', '雾', '阴'])
  expect(app.get('[aria-label="2020-12-24"] img').attributes('src')).toBe(fogIcon)
  await app.get('[data-layer="flows"]').trigger('click')
  await flushPromises()
  expect(app.get('.detail-panel').find('.flow-panel').exists()).toBe(true)
  expect(app.get('.detail-panel').text()).not.toContain('箭头表示')
  expect(app.get('.detail-panel').text()).not.toContain('所选时段累计')
  await app.get('[aria-controls="layer-details"]').trigger('click')
  expect(app.get('.flow-panel').isVisible()).toBe(false)
})
