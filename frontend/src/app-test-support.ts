import { vi } from "vitest"

const mapObject = () => {
  const value = { addTo: vi.fn(() => value), on: vi.fn(() => value) }
  return value
}

const map = {
  createPane: vi.fn(),
  dragging: { disable: vi.fn(), enable: vi.fn() },
  fitBounds: vi.fn(),
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
