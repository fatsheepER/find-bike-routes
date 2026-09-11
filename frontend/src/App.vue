<script setup lang="ts">
import RangeControl from "./RangeControl.vue"
import { init as initChart, type ECharts } from "echarts"
import L, { type GeoJSON as LeafletGeoJSON, type LayerGroup, type Map as LeafletMap, type TileLayer } from "leaflet"
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue"
import {
  aggregateDistrictFlows,
  aggregateFlows,
  flowRequests,
  topRegionFlows,
  type AggregatedFlow,
  type FlowMatrix,
  type FlowSignificance,
  type FlowSlice,
} from "./flow-aggregation"
import {
  ALL_DAYS,
  aggregateRegion,
  datesForScope,
  type Aggregation,
  type DateScope,
  type RegionFeature,
  type RegionSelection,
} from "./region-aggregation"
import {
  aggregationNote,
  colorScaleLimit,
  formatMetric,
  sourceSinkColor,
  regionProfile,
  sourceSinkTooltip,
} from "./source-sink-layer"

type Layer = "source-sink" | "flows" | "sequences"
type Bounds = { west: number; south: number; east: number; north: number }
type DistrictFeature = GeoJSON.Feature<GeoJSON.Geometry, { district_id: number; label: string }>
type RegionContext = {
  release_digest: string
  island_boundary: object
  island_bounds: Bounds
  map_bounds: Bounds
  districts: { type: "FeatureCollection"; features: DistrictFeature[] }
  regions: { type: "FeatureCollection"; features: RegionFeature[] }
}
type SequencePattern = {
  region_ids: number[]
  region_codes: string[]
  district_ids: number[]
  length: number
  support: number
  contiguous_support: number
}
type SequenceResponse = {
  scope: DateScope
  min_contiguous_support: number
  min_contiguous_support_count: number
  valid_tracks: number
  limit: number
  patterns: SequencePattern[]
}
type TrackFeature = GeoJSON.Feature<GeoJSON.Geometry, { track_id: string | number; request_date?: string }>
type TrackResponse = {
  total_count: number
  samples: { type: "FeatureCollection"; features: TrackFeature[] }
}
type FocusSnapshot = {
  layer: Layer
  selection: RegionSelection
  flowMatrix: FlowMatrix
  flowSignificance: FlowSignificance
  selectedFlowKey: string | null
  selectedSequenceIndex: number
  center: L.LatLng
  zoom: number
  userAdjustedView: boolean
}
type FocusTarget =
  | { type: "region"; regionId: number }
  | { type: "bounds"; bounds: Bounds }
type GeographicFocus = {
  target: FocusTarget
  selection: RegionSelection
  status: "loading" | "ready" | "error"
  error: string | null
  totalCount: number
  samples: TrackFeature[]
  snapshot: FocusSnapshot
}
const SUPPORT_LEVELS = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01] as const
const HEALTH_COMPONENTS = [
  ["database", "数据库"],
  ["extensions", "空间扩展"],
  ["dataset_release", "活动发布"],
  ["island_boundary", "本岛边界"],
  ["sequences", "频繁区域序列"],
  ["digest_match", "划分摘要一致性"],
] as const

const layer = ref<Layer>("source-sink")
const dateScope = ref<DateScope>("clear-days")
const startHour = ref(6)
const endHour = ref(10)
const aggregation = ref<Aggregation>("average")
const flowMatrix = ref<FlowMatrix>("od")
const flowSignificance = ref<FlowSignificance>("significant")
const flowStatus = ref<"idle" | "loading" | "ready" | "empty" | "error">("idle")
const flowSlices = ref<FlowSlice[]>([])
const selectedFlowKey = ref<string | null>(null)
const sequenceSupport = ref<number>(0.001)
const sequenceLimit = ref(20)
const sequenceStatus = ref<"idle" | "loading" | "ready" | "empty" | "error">("idle")
const sequenceResponse = ref<SequenceResponse | null>(null)
const selectedSequenceIndex = ref(0)
const currentSequence = computed(
  () => sequenceResponse.value?.patterns[selectedSequenceIndex.value] ?? null,
)
const analysisControlsDisabled = computed(() => layer.value === "sequences")
const selection = computed<RegionSelection>(() => ({
  dateScope: dateScope.value,
  startHour: startHour.value,
  endHour: endHour.value,
  aggregation: aggregation.value,
}))
const symbol = (name: string) => new URL(`./assets/symbols/${name}.svg`, import.meta.url).href
// Raw Weather data.csv, 06:00–10:00: 4/4/4/4; 4/4/2/2; 7/7/7/17; 5/5/5/5; 4/4/4/4.
// COCO definitions: https://dev.meteostat.net/formats.html#weather-condition-codes
const dateWeather = [
  { icon: 'cloud', label: '阴' }, { icon: 'cloud.sun', label: '阴转晴' },
  { icon: 'cloud.rain', label: '小雨、阵雨' }, { icon: 'cloud.fog', label: '雾' }, { icon: 'cloud', label: '阴' },
]
function selectSequenceDate(event: Event) {
  if (event.type === 'pointerdown' && (event as PointerEvent).button > 0) return
  const input = event.target as HTMLInputElement
  const rect = input.getBoundingClientRect()
  const index = event.type === 'pointerdown'
    ? Math.max(0, Math.min(4, Math.round(((event as PointerEvent).clientX - rect.left - 4) / Math.max(1, rect.width - 8) * 4)))
    : Number(input.value)
  dateScope.value = ALL_DAYS[index]
}
const detailCollapsed = ref(false)
const panelTop = ref(180)
const infoDialog = ref<HTMLDialogElement | null>(null)
const infoButton = ref<HTMLButtonElement | null>(null)
const dateStart = computed(() => dateScope.value === 'clear-days' ? 0 : ALL_DAYS.indexOf(datesForScope(dateScope.value)[0] as typeof ALL_DAYS[number]))
const dateEnd = computed(() => dateScope.value === 'clear-days' ? 4 : ALL_DAYS.indexOf(datesForScope(dateScope.value).at(-1) as typeof ALL_DAYS[number]))
function selectDates(start: number, end: number) {
  dateScope.value = start === end ? ALL_DAYS[start] : `${ALL_DAYS[start]}..${ALL_DAYS[end]}`
}
function selectHours(start: number, end: number) { startHour.value = start; endHour.value = end }
function switchLayer(next: Layer) {
  if (interactionLocked.value || next === layer.value) return
  if (next === 'sequences') dateScope.value = 'clear-days'
  layer.value = next
}
function navigateTabs(event: KeyboardEvent) {
  const tabs = ['source-sink', 'flows', 'sequences'] as const
  const offset = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0
  if (!offset && event.key !== 'Home' && event.key !== 'End') return
  event.preventDefault()
  switchLayer(tabs[event.key === 'Home' ? 0 : event.key === 'End' ? 2 : (tabs.indexOf(layer.value) + offset + 3) % 3])
  nextTick(() => document.querySelector<HTMLButtonElement>('[role="tab"][aria-selected="true"]')?.focus())
}
function openInfo() { infoDialog.value?.showModal() }
function closeInfo() { infoDialog.value?.close(); infoButton.value?.focus() }
const mapElement = ref<HTMLElement | null>(null)
const regionContext = ref<RegionContext | null>(null)
const regionStatus = ref<"loading" | "ready" | "empty" | "error">("loading")
const healthStatus = ref<"loading" | "ok" | "unavailable">("loading")
const healthComponents = ref<Record<string, string>>({})
const basemapAvailable = ref(true)
const aggregatedRegions = computed(() =>
  regionContext.value?.regions.features.map((feature) => aggregateRegion(feature, selection.value)) ?? [],
)
const sourceSinkLimit = computed(() => colorScaleLimit(aggregatedRegions.value))

const aggregatedFlows = computed(() => aggregateFlows(flowSlices.value, {
  dateScope: dateScope.value,
  aggregation: aggregation.value,
  significance: flowSignificance.value,
}))
const visibleFlows = computed(() => aggregatedFlows.value.filter(
  (flow) => !flow.is_self_loop && flow.from_region !== flow.to_region,
))
const topFlows = computed(() => topRegionFlows(visibleFlows.value))
const flowKey = (flow: Pick<AggregatedFlow, "from_region" | "to_region">) =>
  `${flow.from_region}/${flow.to_region}`
const currentFlow = computed(() =>
  topFlows.value.find((flow) => flowKey(flow) === selectedFlowKey.value) ?? topFlows.value[0] ?? null,
)
const districtByRegion = computed(() => new Map(
  regionContext.value?.regions.features.map((feature) => [
    feature.properties.region_id,
    feature.properties.district_id,
  ]) ?? [],
))
const districtFlows = computed(() => aggregateDistrictFlows(visibleFlows.value, districtByRegion.value))
const geographicFocus = ref<GeographicFocus | null>(null)
const drawingBounds = ref(false)
const focusActive = computed(() => geographicFocus.value !== null)
const interactionLocked = computed(() => focusActive.value || drawingBounds.value)
const focusedRegion = computed(() => regionContext.value?.regions.features.find(
  (feature) => geographicFocus.value?.target.type === "region" &&
    feature.properties.region_id === geographicFocus.value.target.regionId,
) ?? null)
const focusSelection = computed(() => geographicFocus.value?.selection ?? null)
const focusedProfile = computed(() =>
  focusedRegion.value && focusSelection.value
    ? aggregateRegion(focusedRegion.value, focusSelection.value)
    : null,
)
const focusedProfileHtml = computed(() => {
  if (!focusedProfile.value || !focusSelection.value) return ""
  return regionProfile(
    focusedProfile.value,
    focusSelection.value,
  )
})

let map: LeafletMap | null = null
let tileLayer: TileLayer | null = null
let regionLayer: LeafletGeoJSON | null = null
let flowLayer: LayerGroup | null = null
let sequenceLayer: LayerGroup | null = null
let focusLayer: LayerGroup | null = null
let drawLayer: LayerGroup | null = null
let regionRequest: AbortController | null = null
let flowRequest: AbortController | null = null
let sequenceRequest: AbortController | null = null
let trackRequest: AbortController | null = null
let flowChart: ECharts | null = null
const flowChartElement = ref<HTMLElement | null>(null)
let resizeObserver: ResizeObserver | null = null
let userAdjustedView = false
let fittingView = false
let stopped = false
let drawStart: L.LatLng | null = null

function leafletBounds(bounds: Bounds): L.LatLngBounds {
  return L.latLngBounds(
    [bounds.south, bounds.west],
    [bounds.north, bounds.east],
  )
}

function fitIsland(): void {
  if (!map || !mapElement.value || !regionContext.value) return
  const shortSide = Math.min(mapElement.value.clientWidth, mapElement.value.clientHeight)
  const padding = Math.min(64, Math.max(24, shortSide * 0.04))
  fittingView = true
  map.fitBounds(leafletBounds(regionContext.value.island_bounds), {
    animate: false,
    paddingTopLeft: [layer.value !== "source-sink" && !detailCollapsed.value ? 400 : padding, panelTop.value],
    paddingBottomRight: [sidebarWidth() + padding, detailCollapsed.value ? 70 : 210],
  })
  fittingView = false
}

function measureToolbar(): void {
  const bottom = mapElement.value?.closest('.app-shell')?.querySelector('header')?.getBoundingClientRect().bottom
  panelTop.value = bottom ? bottom + 20 : 180
}

function sidebarWidth(): number {
  if (!focusActive.value) return 0
  return mapElement.value?.closest('.app-shell')?.querySelector<HTMLElement>('.focus-panel')?.getBoundingClientRect().width || 380
}

function updateMapBounds(): void {
  if (!map || !regionContext.value) return
  const bounds = regionContext.value.map_bounds
  const span = bounds.east - bounds.west
  const extra = span * (1 + sidebarWidth() / Math.max(mapElement.value?.clientWidth ?? 1024, 1024))
  map.setMaxBounds(leafletBounds({ ...bounds, west: bounds.west - extra, east: bounds.east + extra }))
}

function centerIsland(): void { userAdjustedView = false; fitIsland() }
function zoomMap(delta: number): void { map?.setZoom(map.getZoom() + delta) }

function renderRegions(): void {
  if (!map || !regionContext.value) return
  if (regionLayer) map.removeLayer(regionLayer)
  const byId = new Map(aggregatedRegions.value.map((region) => [region.region_id, region]))
  const showSourceSink = layer.value === "source-sink"
  const regionFor = (feature?: GeoJSON.Feature) =>
    feature?.properties ? byId.get(feature.properties.region_id as number) : undefined
  regionLayer = L.geoJSON(regionContext.value.regions as GeoJSON.FeatureCollection, {
    interactive: !drawingBounds.value,
    style: (feature) => {
      const region = regionFor(feature)
      return {
        color: "#64748b",
        fillColor: showSourceSink && region ? sourceSinkColor(region.net_inflow_per_km2, sourceSinkLimit.value) : "#f8fafc",
        fillOpacity: showSourceSink ? 0.82 : 0.08,
        weight: 0.8,
      }
    },
    onEachFeature: (feature, leafletLayer) => {
      const region = regionFor(feature)
      if (region) {
        leafletLayer.on?.("click", () => {
          if (!drawingBounds.value) {
            void enterGeographicFocus({ type: "region", regionId: region.region_id })
          }
        })
      }
      if (region) leafletLayer.on('add', () => {
        const element = (leafletLayer as L.Path).getElement?.()
        if (!element) return
        element.setAttribute('tabindex', drawingBounds.value ? '-1' : '0')
        element.setAttribute('role', 'button')
        element.setAttribute('aria-label', region.region_code)
        element.addEventListener('keydown', (event) => {
          if (((event as KeyboardEvent).key === 'Enter' || (event as KeyboardEvent).key === ' ') && !drawingBounds.value) {
            event.preventDefault()
            void enterGeographicFocus({ type: 'region', regionId: region.region_id })
          }
        })
      })
      if (showSourceSink && region) {
        leafletLayer.bindTooltip(sourceSinkTooltip(region), { sticky: true, className: "region-tooltip", opacity: 1 })
      }
    },
  }).addTo(map)
}

function initializeMap(context: RegionContext): void {
  if (map || !mapElement.value) return

  map = L.map(mapElement.value, { attributionControl: true, zoomControl: false })
  map.createPane?.("focusPane")
  updateMapBounds()
  const cartoKey = import.meta.env.VITE_CARTO_API_KEY
  if (!cartoKey) {
    basemapAvailable.value = false
  } else {
    tileLayer = L.tileLayer(
      `https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png?key=${encodeURIComponent(cartoKey)}`,
      {
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: "abcd",
        maxZoom: 20,
      },
    )
    tileLayer.addTo(map)
    tileLayer.on("tileerror", () => {
      if (!map || !tileLayer) return
      map.removeLayer(tileLayer)
      tileLayer = null
      basemapAvailable.value = false
    })
  }

  L.geoJSON(context.island_boundary as GeoJSON.GeoJsonObject, {
    style: { color: "#164e63", fillOpacity: 0.04, weight: 2 },
  }).addTo(map)
  L.geoJSON(context.districts as GeoJSON.GeoJsonObject, {
    style: { color: "#64748b", fillOpacity: 0, weight: 1.5 },
  }).addTo(map)
  renderRegions()

  map.on("movestart", () => {
    if (!fittingView) userAdjustedView = true
  })
  map.on("mousedown", beginBoundsDrag)
  map.on("mousemove", updateBoundsDrag)
  map.on("mouseup", finishBoundsDrag)
  resizeObserver = new ResizeObserver(() => {
    map?.invalidateSize()
    measureToolbar()
    updateMapBounds()
    flowChart?.resize()
    if (!userAdjustedView) fitIsland()
  })
  resizeObserver.observe(mapElement.value)
  const header = mapElement.value.closest('.app-shell')?.querySelector('header')
  if (header) resizeObserver.observe(header)
  measureToolbar()
  fitIsland()
  drawFlows()
  drawSequences()
}

function focusError(status: number): string {
  if (status === 422) return "输入范围无效，请调整后重试。"
  if (status === 404) return "所选区域不存在，请退出后重新选择。"
  if (status === 503) return "轨迹查询依赖暂不可用，请稍后重试。"
  return "轨迹查询服务暂不可用，请重试。"
}

function focusTimestamp(date: string, hour: number): string {
  return `${date}T${String(hour).padStart(2, "0")}:00:00+08:00`
}

function boundsFromDrag(start: L.LatLng, end: L.LatLng): Bounds {
  return { west: start.lng, south: start.lat, east: end.lng, north: end.lat }
}

function validBounds(bounds: Bounds): boolean {
  const allowed = regionContext.value?.map_bounds
  return Boolean(
    allowed &&
    Number.isFinite(bounds.west) && Number.isFinite(bounds.south) &&
    Number.isFinite(bounds.east) && Number.isFinite(bounds.north) &&
    allowed.west <= bounds.west && bounds.west < bounds.east && bounds.east <= allowed.east &&
    allowed.south <= bounds.south && bounds.south < bounds.north && bounds.north <= allowed.north,
  )
}

function startBoundsDrawing(): void {
  if (!map || geographicFocus.value) return
  drawingBounds.value = true
  drawStart = null
  drawLayer ??= L.layerGroup().addTo(map)
  drawLayer.clearLayers()
  map.dragging.disable()
}

function cancelBoundsDrawing(): void {
  const wasDrawing = drawingBounds.value
  drawingBounds.value = false
  drawStart = null
  drawLayer?.clearLayers()
  if (wasDrawing) map?.dragging.enable()
}

function beginBoundsDrag(event: L.LeafletMouseEvent): void {
  if (!drawingBounds.value || geographicFocus.value) return
  drawStart = event.latlng
}

function updateBoundsDrag(event: L.LeafletMouseEvent): void {
  if (!drawingBounds.value || !drawStart || !map) return
  drawLayer ??= L.layerGroup().addTo(map)
  drawLayer.clearLayers()
  L.rectangle(L.latLngBounds(drawStart, event.latlng), {
    color: "#f97316",
    fillColor: "#fff7ed",
    fillOpacity: 0.12,
    weight: 3,
  }).addTo(drawLayer)
}

function finishBoundsDrag(event: L.LeafletMouseEvent): void {
  if (!drawingBounds.value || !drawStart) return
  const bounds = boundsFromDrag(drawStart, event.latlng)
  cancelBoundsDrawing()
  if (validBounds(bounds)) void enterGeographicFocus({ type: "bounds", bounds })
}

function formatBounds(bounds: Bounds): string {
  return `西 ${bounds.west.toFixed(5)}，南 ${bounds.south.toFixed(5)}，东 ${bounds.east.toFixed(5)}，北 ${bounds.north.toFixed(5)}`
}

function drawGeographicFocus(): void {
  focusLayer?.clearLayers()
  const focus = geographicFocus.value
  if (!map || !focus) return
  focusLayer ??= L.layerGroup().addTo(map)
  if (focus.target.type === "region" && focusedRegion.value) {
    const district = regionContext.value?.districts.features.find(
      feature => feature.properties.district_id === focusedRegion.value?.properties.district_id,
    )
    if (district) L.geoJSON(district, {
      pane: 'focusPane', interactive: false,
      style: { color: '#8065a8', fill: false, weight: 3, dashArray: '7 6' },
    }).addTo(focusLayer)
    L.geoJSON(focusedRegion.value as GeoJSON.Feature, {
      pane: "focusPane", interactive: false,
      style: { color: "#f97316", fillColor: "#fff7ed", fillOpacity: 0.12, weight: 4 },
    }).addTo(focusLayer)
  } else if (focus.target.type === "bounds") {
    L.rectangle(leafletBounds(focus.target.bounds), {
      pane: "focusPane", interactive: false,
      color: "#f97316",
      fillColor: "#fff7ed",
      fillOpacity: 0.12,
      weight: 4,
    }).addTo(focusLayer)
  }
  if (focus.status === "ready" && focus.samples.length) {
    L.geoJSON({ type: "FeatureCollection", features: focus.samples } as GeoJSON.FeatureCollection, {
      pane: "focusPane", interactive: false,
      style: { color: "#0369a1", opacity: 0.75, weight: 3 },
    }).addTo(focusLayer)
  }
  const anchor = focus.target.type === "region"
    ? focusedRegion.value?.properties.map_anchor
    : {
        type: "Point",
        coordinates: [
          (focus.target.bounds.west + focus.target.bounds.east) / 2,
          (focus.target.bounds.south + focus.target.bounds.north) / 2,
        ],
      }
  if (focus.status === "ready" && anchor?.type === "Point") {
    L.marker([anchor.coordinates[1], anchor.coordinates[0]], {
      pane: "focusPane", interactive: false,
      icon: L.divIcon({
        className: "focus-count-marker",
        html: `<span>${focus.totalCount}</span>`,
      }),
    }).addTo(focusLayer)
  }
}

async function loadGeographicFocus(): Promise<void> {
  const focus = geographicFocus.value
  if (!focus) return
  trackRequest?.abort()
  const request = new AbortController()
  trackRequest = request
  focus.status = "loading"
  focus.error = null
  focus.totalCount = 0
  focus.samples = []
  try {
    const results = await Promise.all(datesForScope(focus.selection.dateScope).map(async (date) => {
      const response = await fetch("/api/tracks/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: request.signal,
        body: JSON.stringify({
          selection: focus.target.type === "region"
            ? { type: "region", region_id: focus.target.regionId }
            : { type: "bounds", ...focus.target.bounds },
          start: focusTimestamp(date, focus.selection.startHour),
          end: focusTimestamp(date, focus.selection.endHour),
          sample_limit: 20,
        }),
      })
      if (!response.ok) throw Object.assign(new Error("track request failed"), { status: response.status })
      const payload = (await response.json()) as TrackResponse
      if (!Array.isArray(payload.samples?.features) || !Number.isInteger(payload.total_count)) {
        throw new Error("invalid track response")
      }
      return {
        totalCount: payload.total_count,
        samples: payload.samples.features.map((feature) => ({
          ...feature,
          properties: { ...feature.properties, request_date: date },
        })),
      }
    }))
    if (stopped || request !== trackRequest || focus !== geographicFocus.value) return
    focus.totalCount = results.reduce((total, result) => total + result.totalCount, 0)
    focus.samples = results.flatMap((result) => result.samples)
      .sort((left, right) =>
        String(left.properties.request_date).localeCompare(String(right.properties.request_date)) ||
        String(left.properties.track_id).localeCompare(String(right.properties.track_id)),
      )
      .slice(0, 20)
    focus.status = "ready"
  } catch (problem) {
    if (!stopped && request === trackRequest && focus === geographicFocus.value && (problem as Error).name !== "AbortError") {
      focus.status = "error"
      focus.error = focusError(Number((problem as Error & { status?: number }).status ?? 500))
    }
  }
}

async function enterGeographicFocus(target: FocusTarget): Promise<void> {
  if (!map || drawingBounds.value) return
  const previousFocus = geographicFocus.value
  geographicFocus.value = {
    target,
    selection: previousFocus ? { ...previousFocus.selection } : {
      ...selection.value,
      startHour: layer.value === "sequences" ? 6 : startHour.value,
      endHour: layer.value === "sequences" ? 10 : endHour.value,
    },
    status: "loading",
    error: null,
    totalCount: 0,
    samples: [],
    snapshot: previousFocus?.snapshot ?? {
      layer: layer.value,
      selection: { ...selection.value },
      flowMatrix: flowMatrix.value,
      flowSignificance: flowSignificance.value,
      selectedFlowKey: selectedFlowKey.value,
      selectedSequenceIndex: selectedSequenceIndex.value,
      center: map.getCenter(),
      zoom: map.getZoom(),
      userAdjustedView,
    },
  }
  await nextTick()
  updateMapBounds()
  const anchor = target.type === 'region' ? focusedRegion.value?.properties.map_anchor : null
  const point: L.LatLngTuple | null = anchor?.type === 'Point' ? [anchor.coordinates[1], anchor.coordinates[0]]
    : target.type === 'bounds' ? [(target.bounds.south + target.bounds.north) / 2, (target.bounds.west + target.bounds.east) / 2] : null
  if (point) map.panInside?.(point, { paddingTopLeft: [layer.value !== "source-sink" && !detailCollapsed.value ? 420 : 40, panelTop.value], paddingBottomRight: [sidebarWidth() + 40, 60], animate: false })
  drawGeographicFocus()
  await loadGeographicFocus()
}

function exitGeographicFocus(): void {
  const snapshot = geographicFocus.value?.snapshot
  if (!snapshot) return
  trackRequest?.abort()
  focusLayer?.clearLayers()
  geographicFocus.value = null
  layer.value = snapshot.layer
  dateScope.value = snapshot.selection.dateScope
  startHour.value = snapshot.selection.startHour
  endHour.value = snapshot.selection.endHour
  aggregation.value = snapshot.selection.aggregation
  flowMatrix.value = snapshot.flowMatrix
  flowSignificance.value = snapshot.flowSignificance
  selectedFlowKey.value = snapshot.selectedFlowKey
  selectedSequenceIndex.value = snapshot.selectedSequenceIndex
  userAdjustedView = snapshot.userAdjustedView
  if (map) {
    fittingView = true
    map.setView(snapshot.center, snapshot.zoom, { animate: false })
    fittingView = false
  }
}

function handleKeydown(event: KeyboardEvent): void {
  if (event.key !== "Escape") return
  if (infoDialog.value?.open) { event.preventDefault(); closeInfo(); return }
  if (drawingBounds.value) cancelBoundsDrawing()
  else if (geographicFocus.value) exitGeographicFocus()
}

function districtName(id: number): string {
  const district = regionContext.value?.districts.features.find(
    (feature) => feature.properties.district_id === id,
  )
  return String(district?.properties.label ?? id)
}

function regionCode(id: number): string {
  const region = regionContext.value?.regions.features.find(
    (feature) => feature.properties.region_id === id,
  )
  return region?.properties.region_code ?? String(id)
}

function regionDistrictName(id: number): string {
  const districtId = districtByRegion.value.get(id)
  return districtId === undefined ? "未知" : districtName(districtId)
}

function selectFlow(flow: AggregatedFlow, event?: L.LeafletMouseEvent): void {
  if (event?.originalEvent) L.DomEvent.stopPropagation(event.originalEvent)
  if (interactionLocked.value) return
  selectedFlowKey.value = flowKey(flow)
}

function flowArc(from: L.LatLngTuple, to: L.LatLngTuple): L.LatLngTuple[] {
  const deltaLat = to[0] - from[0]
  const deltaLng = to[1] - from[1]
  const distance = Math.hypot(deltaLat, deltaLng)
  if (distance === 0) return [from, to]
  const bend = distance * 0.18
  const middle: L.LatLngTuple = [
    (from[0] + to[0]) / 2 + deltaLng / distance * bend,
    (from[1] + to[1]) / 2 - deltaLat / distance * bend,
  ]
  return Array.from({ length: 17 }, (_, index) => {
    const progress = index / 16
    const remaining = 1 - progress
    return [
      remaining * remaining * from[0] + 2 * remaining * progress * middle[0] + progress * progress * to[0],
      remaining * remaining * from[1] + 2 * remaining * progress * middle[1] + progress * progress * to[1],
    ]
  })
}

function drawFlows(): void {
  flowLayer?.clearLayers()
  if (
    !map ||
    layer.value !== "flows" ||
    flowStatus.value !== "ready" ||
    !regionContext.value
  ) return

  flowLayer ??= L.layerGroup().addTo(map)
  const anchors = new Map(
    regionContext.value.regions.features.flatMap((feature) => {
      if (feature.properties.map_anchor?.type !== "Point") return []
      const coordinates = feature.properties.map_anchor.coordinates
      return [[feature.properties.region_id, [coordinates[1], coordinates[0]] as L.LatLngTuple] as const]
    }),
  )
  const maximum = Math.max(...topFlows.value.map((flow) => flow.weight), 1)
  for (const flow of topFlows.value) {
    const from = anchors.get(flow.from_region)
    const to = anchors.get(flow.to_region)
    if (!from || !to) continue
    const selected = flowKey(flow) === flowKey(currentFlow.value ?? flow)
    const color = selected ? "#c2410c" : "#0369a1"
    const weight = 1.5 + 5 * Math.sqrt(flow.weight / maximum)
    L.polyline(flowArc(from, to), {
      className: "flow-arc",
      color,
      interactive: !interactionLocked.value,
      opacity: selected ? 0.95 : 0.58,
      weight,
    })
      .addTo(flowLayer)
      .on("click", (event) => selectFlow(flow, event))

    const rotation = Math.atan2(-(to[0] - from[0]), to[1] - from[1]) * 180 / Math.PI
    L.marker(to, {
      icon: L.divIcon({
        className: "flow-arrow",
        html: `<span style="color: ${color}; transform: rotate(${rotation}deg)">➤</span>`,
      }),
      interactive: !interactionLocked.value,
    })
      .addTo(flowLayer)
      .on("click", (event) => selectFlow(flow, event))
  }
}

async function loadFlows(): Promise<void> {
  flowRequest?.abort()
  const request = new AbortController()
  flowRequest = request
  flowStatus.value = "loading"
  flowSlices.value = []
  selectedFlowKey.value = null
  const requests = flowRequests(
    flowMatrix.value,
    flowSignificance.value,
    dateScope.value,
    startHour.value,
    endHour.value,
  )
  try {
    const slices = await Promise.all(requests.map(async (item) => {
      const response = await fetch(item.url, { signal: request.signal })
      if (!response.ok) throw new Error("flow request failed")
      const payload = (await response.json()) as { flows?: unknown }
      if (!Array.isArray(payload.flows)) throw new Error("invalid flow response")
      return { scope: item.scope, hour: item.hour, rows: payload.flows } as FlowSlice
    }))
    if (stopped || request !== flowRequest) return
    flowSlices.value = slices
    flowStatus.value = topRegionFlows(aggregateFlows(slices, {
      dateScope: dateScope.value,
      aggregation: aggregation.value,
      significance: flowSignificance.value,
    })).length ? "ready" : "empty"
  } catch (problem) {
    if (!stopped && request === flowRequest && (problem as Error).name !== "AbortError") {
      flowStatus.value = "error"
    }
  }
}

function renderFlowChart(): void {
  if (layer.value !== "flows" || flowStatus.value !== "ready" || !flowChartElement.value) {
    flowChart?.dispose()
    flowChart = null
    return
  }
  flowChart ??= initChart(flowChartElement.value)
  const districtIds = [...new Set(districtFlows.value.flatMap(
    (flow) => [flow.from_district, flow.to_district],
  ))]
  const selectedDistricts = currentFlow.value && [
    districtByRegion.value.get(currentFlow.value.from_region),
    districtByRegion.value.get(currentFlow.value.to_region),
  ]
  flowChart.setOption({
    tooltip: { trigger: "item" },
    series: [{
      type: "chord",
      data: districtIds.map((id) => ({ id: String(id), name: districtName(id) })),
      links: districtFlows.value.map((flow) => {
        const selected = selectedDistricts?.[0] === flow.from_district && selectedDistricts[1] === flow.to_district
        return {
          source: String(flow.from_district),
          target: String(flow.to_district),
          value: flow.weight,
          name: flow.is_internal ? "片区内部区域间流动" : `${districtName(flow.from_district)} → ${districtName(flow.to_district)}`,
          lineStyle: selected ? { color: "#c2410c", opacity: 0.9 } : undefined,
        }
      }),
      label: { show: true },
      lineStyle: { color: "source", opacity: 0.45 },
      emphasis: { focus: "adjacency" },
    }],
  })
}

function selectSequence(index: number, event?: L.LeafletMouseEvent): void {
  if (event?.originalEvent) L.DomEvent.stopPropagation(event.originalEvent)
  if (interactionLocked.value) return
  selectedSequenceIndex.value = index
}

function drawSequences(): void {
  sequenceLayer?.clearLayers()
  if (
    !map ||
    layer.value !== "sequences" ||
    sequenceStatus.value !== "ready" ||
    !sequenceResponse.value ||
    !regionContext.value
  ) return

  sequenceLayer ??= L.layerGroup().addTo(map)
  const anchors = new Map(
    regionContext.value.regions.features.map((feature) => {
      const coordinates = (feature.properties.map_anchor as GeoJSON.Point).coordinates
      return [Number(feature.properties.region_id), [coordinates[1], coordinates[0]] as L.LatLngTuple]
    }),
  )
  sequenceResponse.value.patterns.forEach((pattern, index) => {
    const positions = pattern.region_ids.map((id) => anchors.get(id)).filter((point) => point !== undefined)
    if (positions.length !== pattern.region_ids.length) return
    const selected = index === selectedSequenceIndex.value
    L.polyline(positions, {
      color: selected ? "#c2410c" : "#64748b",
      interactive: !interactionLocked.value,
      opacity: selected ? 1 : 0.35,
      weight: selected ? 5 : 2,
    })
      .addTo(sequenceLayer!)
      .on("click", (event) => selectSequence(index, event))
    positions.forEach((position) => {
      L.circleMarker(position, {
        color: selected ? "#9a3412" : "#64748b",
        fillColor: selected ? "#fb923c" : "#cbd5e1",
        fillOpacity: selected ? 1 : 0.5,
        interactive: !interactionLocked.value,
        radius: selected ? 6 : 4,
        weight: 2,
      })
        .addTo(sequenceLayer!)
        .on("click", (event) => selectSequence(index, event))
    })
  })
}

async function loadSequences(): Promise<void> {
  sequenceRequest?.abort()
  const request = new AbortController()
  sequenceRequest = request
  sequenceStatus.value = "loading"
  sequenceResponse.value = null
  selectedSequenceIndex.value = 0
  if (dateScope.value.includes('..')) {
    sequenceStatus.value = 'empty'
    return
  }
  const query = new URLSearchParams({
    scope: dateScope.value,
    min_contiguous_support: String(sequenceSupport.value),
    limit: String(sequenceLimit.value),
  })
  try {
    const response = await fetch(`/api/sequences?${query}`, { signal: request.signal })
    if (!response.ok) throw new Error("sequence request failed")
    const payload = (await response.json()) as SequenceResponse
    if (!Array.isArray(payload.patterns)) throw new Error("invalid sequence response")
    if (stopped || request !== sequenceRequest) return
    sequenceResponse.value = payload
    sequenceStatus.value = payload.patterns.length ? "ready" : "empty"
  } catch (problem) {
    if (!stopped && request === sequenceRequest && (problem as Error).name !== "AbortError") {
      sequenceStatus.value = "error"
    }
  }
}

function setSequenceLimit(event: Event): void {
  const value = Math.trunc(Number((event.target as HTMLInputElement).value))
  sequenceLimit.value = Number.isFinite(value) ? Math.min(100, Math.max(1, value)) : 20
}

async function loadRegions(): Promise<void> {
  regionRequest?.abort()
  const request = new AbortController()
  regionRequest = request
  regionStatus.value = "loading"
  try {
    const response = await fetch("/api/regions", { signal: request.signal })
    if (!response.ok) throw new Error("region request failed")
    const context = (await response.json()) as RegionContext
    if (stopped || request !== regionRequest) return
    regionContext.value = context
    regionStatus.value = context.regions.features.length === 0 ? "empty" : "ready"
    initializeMap(context)
  } catch (problem) {
    if (!stopped && request === regionRequest && (problem as Error).name !== "AbortError") {
      regionStatus.value = "error"
    }
  }
}

async function loadHealth(): Promise<void> {
  healthStatus.value = "loading"
  try {
    const response = await fetch("/api/health")
    const health = (await response.json()) as {
      status?: string
      components?: Record<string, { status?: string }>
    }
    if (stopped) return
    healthComponents.value = Object.fromEntries(
      HEALTH_COMPONENTS.map(([key]) => [key, health.components?.[key]?.status ?? "unavailable"]),
    )
    healthStatus.value = response.ok && health.status === "ok" ? "ok" : "unavailable"
  } catch {
    if (!stopped) {
      healthComponents.value = {}
      healthStatus.value = "unavailable"
    }
  }
}

watch([layer, dateScope, sequenceSupport, sequenceLimit], () => {
  if (layer.value === "sequences") void loadSequences()
  else {
    sequenceRequest?.abort()
    sequenceRequest = null
    sequenceStatus.value = "idle"
    sequenceResponse.value = null
    selectedSequenceIndex.value = 0
  }
})
watch([layer, dateScope, startHour, endHour, flowMatrix, flowSignificance], () => {
  if (layer.value === "flows") void loadFlows()
  else {
    flowRequest?.abort()
    flowRequest = null
    flowStatus.value = "idle"
    flowSlices.value = []
    selectedFlowKey.value = null
  }
})
watch([layer, flowStatus, topFlows, currentFlow, regionContext, focusActive], drawFlows)
watch([layer, flowStatus, districtFlows, currentFlow, regionContext, focusActive, detailCollapsed], async () => {
  await nextTick()
  renderFlowChart()
  if (!detailCollapsed.value) flowChart?.resize()
}, { flush: "post" })
watch([layer, sequenceStatus, sequenceResponse, selectedSequenceIndex, regionContext, focusActive], drawSequences)

onMounted(() => {
  window.addEventListener("keydown", handleKeydown)
  void loadRegions()
  void loadHealth()
})

watch([layer, dateScope, startHour, endHour, aggregation], renderRegions)
watch(focusActive, renderRegions)
watch([layer, focusActive], async ([nextLayer, nextFocus], [previousLayer, previousFocus]) => {
  await nextTick()
  updateMapBounds()
  if (!nextFocus && !previousFocus && nextLayer !== previousLayer) {
    if (!userAdjustedView) fitIsland()
    else map?.panBy?.([(nextLayer === "source-sink" ? 1 : previousLayer === "source-sink" ? -1 : 0) * (detailCollapsed.value ? 0 : 190), 0], { animate: false })
  }
})
watch(drawingBounds, () => {
  renderRegions()
  drawFlows()
  drawSequences()
})
watch(geographicFocus, drawGeographicFocus, { deep: true })
watch(
  () => geographicFocus.value && [geographicFocus.value.selection.startHour, geographicFocus.value.selection.endHour],
  (current, previous) => {
    if (current && previous && (current[0] !== previous[0] || current[1] !== previous[1])) {
      void loadGeographicFocus()
    }
  },
)

onBeforeUnmount(() => {
  stopped = true
  regionRequest?.abort()
  flowRequest?.abort()
  sequenceRequest?.abort()
  trackRequest?.abort()
  cancelBoundsDrawing()
  window.removeEventListener("keydown", handleKeydown)
  flowChart?.dispose()
  resizeObserver?.disconnect()
  map?.remove()
  map = null
  regionLayer = null
})
</script>

<template>
  <main class="app-shell" :style="{ '--panel-top': panelTop + 'px' }" :class="{ 'focus-active': focusActive, 'bounds-drawing': drawingBounds }">
    <header>
      <form class="toolbar" :inert="focusActive" aria-label="全局工具栏" @submit.prevent>
        <div class="capsule layer-tabs" role="tablist" aria-label="内容图层" @keydown="navigateTabs">
          <button v-for="tab in ([['source-sink', '源汇'], ['flows', '区域间流动'], ['sequences', '通勤链']] as const)"
            :key="tab[0]" type="button" role="tab" :data-layer="tab[0]" :aria-selected="layer === tab[0]"
            :tabindex="layer === tab[0] ? 0 : -1" :disabled="interactionLocked" @click="switchLayer(tab[0])">{{ tab[1] }}</button>
        </div>
        <div class="capsule date-capsule">
          <div class="date-selector">
            <div class="range-labels date-labels">
              <button v-for="(date, index) in ALL_DAYS" :key="date" type="button" :aria-label="date" :disabled="interactionLocked"
                :class="{ rainy: index === 2, selected: dateScope !== 'clear-days' && index >= dateStart && index <= dateEnd }"
                @click="selectDates(index, index)"><img class="sf-symbol weather-symbol" :src="symbol(dateWeather[index].icon)" :alt="dateWeather[index].label" />{{ date.slice(5).replace('-', '/') }}</button>
            </div>
            <div v-if="layer === 'sequences'" class="range-control sequence-date" :class="{ 'no-date': dateScope === 'clear-days' }">
              <div class="range-track" />
              <input type="range" min="0" max="4" step="1" :value="dateStart" aria-label="通勤链日期"
                :aria-valuetext="dateScope === 'clear-days' ? '晴天集' : dateScope" :disabled="interactionLocked"
                @input="selectSequenceDate" @pointerdown="selectSequenceDate" />
            </div>
            <RangeControl v-else :min="0" :max="4" :start="dateStart" :end="dateEnd" start-label="开始日期" end-label="结束日期" :start-text="ALL_DAYS[dateStart]" :end-text="ALL_DAYS[dateEnd]"
              :unfilled="dateScope === 'clear-days'" :disabled="interactionLocked" @change="selectDates" />
          </div>
          <Transition name="restore"><button v-if="dateScope !== 'clear-days'" class="restore-button" type="button"
            :disabled="interactionLocked" @click="dateScope = 'clear-days'">重置为晴天集</button></Transition>
        </div>
        <div v-if="!analysisControlsDisabled" class="capsule time-capsule">
          <div class="time-selector">
            <div class="range-labels"><span v-for="hour in [6, 7, 8, 9, 10]" :key="hour">{{ hour }}:00</span></div>
            <RangeControl :min="6" :max="10" :gap="1" :start="startHour" :end="endHour" start-label="开始时间" end-label="结束时间" :start-text="startHour + ':00'" :end-text="endHour + ':00'"
              :disabled="interactionLocked" @change="selectHours" />
          </div>
          <Transition name="restore"><button v-if="startHour !== 6 || endHour !== 10" class="restore-button" type="button"
            :disabled="interactionLocked" @click="selectHours(6, 10)">恢复全选</button></Transition>
        </div>
        <div :key="layer" class="capsule layer-options">
          <template v-if="layer !== 'sequences'">
            <label>聚合口径<select v-model="aggregation" aria-label="聚合口径" :disabled="interactionLocked"><option value="average">跨日平均</option><option value="sum">跨日合计</option></select></label>
            <template v-if="layer === 'flows'">
              <label>矩阵<select v-model="flowMatrix" aria-label="流矩阵" :disabled="interactionLocked"><option value="od">出行流</option><option value="channel">通道流</option></select></label>
              <label>显著性<select v-model="flowSignificance" aria-label="显著性" :disabled="interactionLocked"><option value="significant">仅显著</option><option value="all">全部流对</option></select></label>
            </template>
          </template>
          <template v-else>
            <label>连续支持度<select v-model.number="sequenceSupport" aria-label="连续支持度门槛" :disabled="interactionLocked"><option v-for="level in SUPPORT_LEVELS" :key="level" :value="level">{{ level }}</option></select></label>
            <label>数量<input :value="sequenceLimit" aria-label="Top-N" type="number" min="1" max="100" step="1" :disabled="interactionLocked" @change="setSequenceLimit" /></label>
          </template>
        </div>
      </form>
      <nav class="map-toolbar" aria-label="地图操作">
        <div class="capsule"><button type="button" aria-label="放大地图" @click="zoomMap(1)"><img class="sf-symbol" :src="symbol('plus')" alt="" /></button><button type="button" aria-label="缩小地图" @click="zoomMap(-1)"><img class="sf-symbol" :src="symbol('minus')" alt="" /></button></div>
        <button class="capsule" type="button" aria-label="本岛居中" @click="centerIsland">居中</button>
        <button class="capsule" type="button" aria-label="框选范围" :aria-pressed="drawingBounds" :disabled="focusActive"
          @click="drawingBounds ? cancelBoundsDrawing() : startBoundsDrawing()"><span v-if="drawingBounds">× 取消框选</span><img v-else class="sf-symbol" :src="symbol('crop')" alt="" /></button>
      </nav>
    </header>

    <section class="workspace">
      <section class="map-panel" aria-label="业务地图">
        <p v-if="regionStatus === 'loading'" class="map-status" role="status">正在加载区域地图…</p>
        <div v-if="regionStatus === 'error'" class="map-status" role="alert">
          <p>区域地图暂不可用。</p>
          <button type="button" @click="loadRegions">重试</button>
        </div>
        <p v-if="regionStatus === 'empty'" class="map-status" role="status">区域数据为空，共 0 个区域。</p>
        <div id="map" ref="mapElement" />
      </section>

      <aside class="detail-panel" :class="{ 'analysis-panel': layer !== 'source-sink' }" aria-label="图层说明">
        <div class="detail-heading"><button type="button" :aria-expanded="!detailCollapsed" aria-controls="layer-details" @click="detailCollapsed = !detailCollapsed"><img class="sf-symbol" :src="symbol(detailCollapsed ? 'chevron.up' : 'chevron.down')" alt="" />{{ detailCollapsed ? '展开' : '收起' }}</button><button ref="infoButton" type="button" class="info-button" aria-label="应用信息" @click="openInfo"><img class="sf-symbol" :src="symbol('info')" alt="" /></button></div>
        <div v-show="!detailCollapsed" id="layer-details" class="detail-body">
          <template v-if="layer === 'source-sink'">
            <div class="source-sink-legend" aria-label="净流入强度图例"><div class="legend-title"><strong>净流入强度</strong><span>单/km²</span></div>
              <div class="legend-gradient" :style="{ background: `linear-gradient(90deg, ${sourceSinkColor(-1, 1)}, #f8fafc, ${sourceSinkColor(1, 1)})` }" />
              <div class="legend-ticks"><span>源 {{ sourceSinkLimit === null ? '—' : formatMetric(-sourceSinkLimit) }}</span><span>0</span><span>汇 {{ sourceSinkLimit === null ? '—' : '+' + formatMetric(sourceSinkLimit) }}</span></div>
            </div>
          </template>
          <p v-if="layer === 'source-sink'" class="metric-scope">{{ dateScope === 'clear-days' ? '晴天集' : datesForScope(dateScope).map(date => date.slice(5).replace('-', '/')).join('、') }} · {{ aggregationNote(selection) }}</p>
          <section v-if="layer === 'flows'" class="flow-panel" aria-label="区域间流动" :inert="interactionLocked">
            <p v-if="flowStatus === 'loading'" role="status">正在加载区域流…</p>
            <div v-else-if="flowStatus === 'error'" role="alert">
              <p>区域流暂不可用，区域地图仍可查看。</p>
              <button type="button" @click="loadFlows">重试</button>
            </div>
            <p v-else-if="flowStatus === 'empty'" role="status">当前条件下没有区域流。</p>
            <template v-else-if="flowStatus === 'ready'">
              <figure class="flow-chart-figure">
                <div ref="flowChartElement" class="flow-chart" aria-label="片区流动弦图" />
                <figcaption>片区间流量</figcaption>
              </figure>
              <h3>区域流量 <span class="muted">前 {{ topFlows.length }} 条</span></h3>
              <ol class="flow-list" aria-label="Top 50 区域流列表">
                <li v-for="flow in topFlows" :key="flowKey(flow)">
                  <button
                    type="button"
                    :aria-current="flowKey(flow) === flowKey(currentFlow ?? flow) ? 'true' : undefined"
                    @click="selectFlow(flow)"
                  >
                    <span>{{ regionCode(flow.from_region) }} → {{ regionCode(flow.to_region) }}</span><strong>{{ formatMetric(flow.weight) }}</strong>
                  </button>
                </li>
              </ol>
              <dl v-if="currentFlow" class="flow-details" aria-label="所选流对详情">
                <dt>矩阵</dt>
                <dd>{{ currentFlow.matrix === "od" ? "出行流" : "通道流" }}</dd>
                <dt>起终区域编码</dt>
                <dd>{{ regionCode(currentFlow.from_region) }} → {{ regionCode(currentFlow.to_region) }}</dd>
                <dt>起终片区</dt>
                <dd>{{ regionDistrictName(currentFlow.from_region) }} → {{ regionDistrictName(currentFlow.to_region) }}</dd>
                <dt>权重</dt>
                <dd>{{ formatMetric(currentFlow.weight) }}</dd>
              </dl><details v-if="currentFlow" class="flow-diagnostics" aria-label="流对检验详情"><summary>检验详情</summary><dl>
                <dt>已检验</dt>
                <dd>{{ currentFlow.is_tested ? "是" : "否" }}</dd>
                <dt>显著</dt>
                <dd>{{ currentFlow.is_significant === null ? "未检验" : currentFlow.is_significant ? "是" : "否" }}</dd>
                <dt>被门槛挡住</dt>
                <dd>{{ currentFlow.gated ? "是" : "否" }}</dd>
              </dl></details>
            </template>
          </section>

          <section v-if="layer === 'sequences'" class="sequence-panel" aria-label="典型通勤链" :inert="interactionLocked">
            <p v-if="sequenceStatus === 'loading'" role="status">正在加载通勤链…</p>
            <div v-else-if="sequenceStatus === 'error'" role="alert">
              <p>通勤链暂不可用，区域地图仍可查看。</p>
              <button type="button" @click="loadSequences">重试</button>
            </div>

            <template v-else-if="sequenceStatus === 'ready' && sequenceResponse">
              <ol class="sequence-list" aria-label="典型通勤链列表">
                <li v-for="(pattern, index) in sequenceResponse.patterns" :key="pattern.region_ids.join('-')">
                  <button
                    type="button"
                    :aria-current="index === selectedSequenceIndex ? 'true' : undefined"
                    @click="selectSequence(index)"
                  >
                    {{ pattern.region_codes.join(" → ") }}
                  </button>
                </li>
              </ol>
              <dl v-if="currentSequence" class="sequence-details">
                <dt>区域编码</dt>
                <dd>{{ currentSequence.region_codes.join(" → ") }}</dd>
                <dt>所属片区</dt>
                <dd>{{ currentSequence.district_ids.map(districtName).join(" → ") }}</dd>
                <dt>长度</dt>
                <dd>{{ currentSequence.length }}</dd>
                <dt>支持度（区域序列条数）</dt>
                <dd>{{ currentSequence.support }}</dd>
                <dt>连续支持度</dt>
                <dd>{{ currentSequence.contiguous_support }}</dd>
              </dl><details v-if="currentSequence" class="sequence-diagnostics"><summary>门槛与样本</summary><dl>
                <dt>绝对连续支持度门槛</dt>
                <dd>{{ sequenceResponse.min_contiguous_support_count }}</dd>
                <dt>有效轨迹分母</dt>
                <dd>{{ sequenceResponse.valid_tracks }}</dd>
              </dl></details>
              <details><summary>统计口径</summary><p class="sequence-note">
                支持度按含该模式的区域序列条数计算，同一条序列内重复出现只计一次；一条有效轨迹可能贡献多条区域序列。绘图资格使用 API 已审计的连续支持度与区域相邻性，不受跨日平均或合计影响。
              </p></details>
            </template>
          </section>

        </div>
      </aside>

      <aside
        v-if="geographicFocus"
        class="focus-panel"
        :aria-label="geographicFocus.target.type === 'region' ? '区域聚焦详情' : '矩形聚焦详情'"
      >
        <div class="focus-heading">
          <div>
            <strong class="region-name">{{ geographicFocus.target.type === "region" ? focusedProfile?.region_code : "矩形范围" }}</strong>
            <span v-if="focusedProfile" class="muted">{{ districtName(focusedProfile.district_id) }} · {{ formatMetric(focusedProfile.area_km2) }} km²</span>
          </div>
          <button type="button" @click="exitGeographicFocus">退出聚焦</button>
        </div>

        <section v-if="focusedProfile" aria-label="区域画像">
          <div v-html="focusedProfileHtml" />
        </section>
        <dl class="focus-totals">
          <dt>唯一有效轨迹总数</dt>
          <dd>{{ geographicFocus.status === "ready" ? geographicFocus.totalCount : "—" }}</dd>
          <dt>实际样例数</dt>
          <dd>{{ geographicFocus.status === "ready" ? geographicFocus.samples.length : "—" }}</dd>
        </dl>
        <details class="focus-query-details"><summary>查询详情</summary><dl class="focus-details">
          <dt>选择类型</dt>
          <dd>{{ geographicFocus.target.type === "region" ? "区域" : "矩形" }}</dd>
          <template v-if="geographicFocus.target.type === 'region' && focusedProfile">
            <dt>区域编码</dt>
            <dd>{{ focusedProfile.region_code }}</dd>
          </template>
          <template v-else-if="geographicFocus.target.type === 'bounds'">
            <dt>矩形坐标</dt>
            <dd>{{ formatBounds(geographicFocus.target.bounds) }}</dd>
          </template>
          <dt>日期</dt>
          <dd>{{ geographicFocus.selection.dateScope === "clear-days" ? "晴天集（4 日）" : geographicFocus.selection.dateScope }}</dd>
          <dt>局部时间</dt>
          <dd>{{ String(geographicFocus.selection.startHour).padStart(2, "0") }}:00–{{ String(geographicFocus.selection.endHour).padStart(2, "0") }}:00</dd>
          <dt>聚合口径</dt>
          <dd v-if="geographicFocus.target.type === 'bounds'">
            {{ geographicFocus.selection.aggregation === "average" ? "跨日平均（有效轨迹计数不缩放）" : "跨日合计（唯一有效轨迹数按日相加）" }}
          </dd>
          <dd v-else>{{ geographicFocus.selection.aggregation === "average" ? "跨日平均（仅用于区域动态指标）" : "跨日合计" }}</dd>

        </dl>
        <p class="focus-sample-note">样例最多 20 条。</p></details>


        <p v-if="geographicFocus.status === 'loading'" role="status">正在查询有效轨迹…</p>
        <div v-else-if="geographicFocus.status === 'error'" role="alert">
          <p>{{ geographicFocus.error }}</p>
          <button type="button" @click="loadGeographicFocus">重试</button>
        </div>
        <p v-else-if="geographicFocus.totalCount === 0" role="status">当前条件下共有 0 条唯一有效轨迹，样例为空。</p>


      </aside>
    </section>

    <dialog ref="infoDialog" class="info-dialog" aria-labelledby="info-title" @cancel.prevent="closeInfo" @click="($event.target === infoDialog) && closeInfo()">
      <div class="dialog-heading"><h2 id="info-title">关于数据</h2><button type="button" aria-label="关闭应用信息" @click="closeInfo">关闭</button></div>
      <p>数据来源：厦门共享单车 GPS 与订单数据</p>
      <p>研究范围：2020-12-21 至 2020-12-25，06:00–10:00，厦门本岛</p>
      <p>结论边界：历史工作日早高峰研究结果，不代表实时、全天或厦门全市情况。</p>
      <details class="diagnostics"><summary>技术诊断</summary><p>组件状态：{{ healthStatus === "ok" ? "正常" : healthStatus === "loading" ? "检查中" : "部分不可用" }}</p>
      <ul v-if="Object.keys(healthComponents).length" class="health-components">
        <li v-for="[key, label] in HEALTH_COMPONENTS" :key="key">
          {{ label }}：{{ healthComponents[key] === "ok" ? "正常" : "不可用" }}
        </li>
      </ul>
      <details v-if="regionContext">
        <summary>活动发布摘要</summary>
        <code>{{ regionContext.release_digest }}</code>
      </details>
      </details>
      <p v-if="!basemapAvailable">底图暂不可用，本地地图仍可查看。</p>
      <p class="attribution">
        地图数据 © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors，
        底图样式 © <a href="https://carto.com/attributions">CARTO</a>
      </p>
    </dialog>
  </main>
</template>
