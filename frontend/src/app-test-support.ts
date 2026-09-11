import { vi } from "vitest"

const mapObject = () => {
  const value = { addTo: vi.fn(() => value), on: vi.fn(() => value) }
  return value
}

const map = {
  createPane: vi.fn(),
  dragging: { disable: vi.fn(), enable: vi.fn() },
  fitBounds: vi.fn(),
  panInside: vi.fn(),
  panBy: vi.fn(),
  setZoom: vi.fn(),
  getCenter: vi.fn(() => ({ lat: 24, lng: 118 })),
  getZoom: vi.fn(() => 12),
  invalidateSize: vi.fn(),
  on: vi.fn(),
  remove: vi.fn(),
  removeLayer: vi.fn(),
  setMaxBounds: vi.fn(),
  setView: vi.fn(),
}

const regionClicks = new Map<number, () => void>()
const tile = { addTo: vi.fn(), on: vi.fn() }

export const leaflet = {
  DomEvent: { stopPropagation: vi.fn() },
  circleMarker: vi.fn(mapObject),
  divIcon: vi.fn((options) => options),
  geoJSON: vi.fn((data: { features?: Array<{ properties?: { region_id?: number } }> }, options?: {
    onEachFeature?: (feature: { properties?: { region_id?: number } }, layer: object) => void
  }) => {
    data.features?.forEach((feature) => options?.onEachFeature?.(feature, {
      bindTooltip: vi.fn(),
      on: vi.fn((event: string, handler: () => void) => {
        if (event === "click" && feature.properties?.region_id) regionClicks.set(feature.properties.region_id, handler)
      }),
    }))
    return mapObject()
  }),
  latLngBounds: vi.fn((southWest, northEast) => [southWest, northEast]),
  layerGroup: vi.fn(() => {
    const group = { addTo: vi.fn(() => group), clearLayers: vi.fn() }
    return group
  }),
  map: vi.fn(() => map),
  mapInstance: map,
  marker: vi.fn(mapObject),
  polyline: vi.fn(mapObject),
  rectangle: vi.fn(mapObject),
  regionClicks,
  tile,
  tileLayer: vi.fn(() => tile),
}

export function testRegion(id: number) {
  return {
    type: "Feature",
    geometry: { type: "Point", coordinates: [118 + id / 1000, 24 + id / 1000] },
    properties: {
      region_id: id,
      region_code: `R-${id}`,
      district_id: 1,
      cells: 1,
      area_km2: 1,
      functional_composition: {},
      classified_share: 0,
      bus_stops_per_km2: 0,
      map_anchor: { type: "Point", coordinates: [118 + id / 1000, 24 + id / 1000] },
      metrics: [],
    },
  }
}

export function testRegionContext(ids = [7], releaseDigest = "release") {
  return {
    release_digest: releaseDigest,
    island_boundary: testRegion(1),
    island_bounds: { west: 118, south: 24, east: 118.2, north: 24.2 },
    map_bounds: { west: 117.9, south: 23.9, east: 118.3, north: 24.3 },
    districts: { type: "FeatureCollection", features: [] },
    regions: { type: "FeatureCollection", features: ids.map(testRegion) },
  }
}

export class ResizeObserverStub {
  observe = vi.fn()
  disconnect = vi.fn()
}

// UI adapters keep data/request regression tests independent of control markup.
import type { VueWrapper } from '@vue/test-utils'
export async function selectDate(app: VueWrapper, date: string) {
  if (date === 'clear-days') {
    const reset = app.findAll('button').find(button => button.text() === '重置为晴天集')
    if (reset) await reset.trigger('click')
  } else await app.get(`[aria-label="${date}"]`).trigger('click')
}
export function currentDate(app: VueWrapper) {
  const start = Number((app.get('[aria-label="开始日期"]').element as HTMLInputElement).value)
  const end = Number((app.get('[aria-label="结束日期"]').element as HTMLInputElement).value)
  return start === end ? `2020-12-${21 + start}` : 'clear-days'
}
// Local time UI is deferred; retain checks for the existing query lifecycle.
export function focusHour(app: VueWrapper, edge: 'start' | 'end') {
  return String((app.vm as unknown as { geographicFocus: { selection: Record<string, number> } }).geographicFocus.selection[`${edge}Hour`])
}
export async function setFocusHour(app: VueWrapper, hour: string) {
  (app.vm as unknown as { geographicFocus: { selection: { startHour: number } } }).geographicFocus.selection.startHour = Number(hour)
  await app.vm.$nextTick()
}
