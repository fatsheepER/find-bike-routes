<script setup lang="ts">
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
  sourceSinkLegend,
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
type RegionFocus = {
  regionId: number
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
const sourceSinkLegendText = computed(() => sourceSinkLegend(sourceSinkLimit.value, selection.value))
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
const regionFocus = ref<RegionFocus | null>(null)
const focusActive = computed(() => regionFocus.value !== null)
const focusedRegion = computed(() => regionContext.value?.regions.features.find(
  (feature) => feature.properties.region_id === regionFocus.value?.regionId,
) ?? null)
const focusSelection = computed(() => regionFocus.value?.selection ?? null)
const focusedProfile = computed(() =>
  focusedRegion.value && focusSelection.value
    ? aggregateRegion(focusedRegion.value, focusSelection.value)
    : null,
)
const focusedProfileHtml = computed(() => {
  if (!focusedProfile.value || !focusSelection.value) return ""
  return sourceSinkTooltip(
    focusedProfile.value,
    focusSelection.value,
    districtName(focusedProfile.value.district_id),
  )
})

let map: LeafletMap | null = null
let tileLayer: TileLayer | null = null
let regionLayer: LeafletGeoJSON | null = null
let flowLayer: LayerGroup | null = null
let sequenceLayer: LayerGroup | null = null
let focusLayer: LayerGroup | null = null
let flowRequest: AbortController | null = null
let sequenceRequest: AbortController | null = null
let trackRequest: AbortController | null = null
let flowChart: ECharts | null = null
const flowChartElement = ref<HTMLElement | null>(null)
let resizeObserver: ResizeObserver | null = null
let userAdjustedView = false
let fittingView = false
let stopped = false

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
    padding: [padding, padding],
  })
  fittingView = false
}

function renderRegions(): void {
  if (!map || !regionContext.value) return
  if (regionLayer) map.removeLayer(regionLayer)
  const byId = new Map(aggregatedRegions.value.map((region) => [region.region_id, region]))
  const districtLabels = new Map(
    regionContext.value.districts.features.map((feature) => [feature.properties.district_id, feature.properties.label]),
  )
  const showSourceSink = layer.value === "source-sink"
  const regionFor = (feature?: GeoJSON.Feature) =>
    feature?.properties ? byId.get(feature.properties.region_id as number) : undefined
  regionLayer = L.geoJSON(regionContext.value.regions as GeoJSON.FeatureCollection, {
    interactive: !focusActive.value,
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
          if (!focusActive.value) void enterRegionFocus(region.region_id)
        })
      }
      if (showSourceSink && region) {
        const districtLabel = districtLabels.get(region.district_id) ?? String(region.district_id)
        leafletLayer.bindTooltip(sourceSinkTooltip(region, selection.value, districtLabel), { sticky: true })
      }
    },
  }).addTo(map)
}

function initializeMap(context: RegionContext): void {
  if (map || !mapElement.value) return

  map = L.map(mapElement.value, { attributionControl: true })
  map.createPane?.("focusPane")
  map.setMaxBounds(leafletBounds(context.map_bounds))
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
  resizeObserver = new ResizeObserver(() => {
    map?.invalidateSize()
    if (!userAdjustedView) fitIsland()
  })
  resizeObserver.observe(mapElement.value)
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

function drawRegionFocus(): void {
  focusLayer?.clearLayers()
  if (!map || !regionFocus.value || !focusedRegion.value) return
  focusLayer ??= L.layerGroup().addTo(map)
  L.geoJSON(focusedRegion.value as GeoJSON.Feature, {
    pane: "focusPane",
    style: { color: "#f97316", fillColor: "#fff7ed", fillOpacity: 0.12, weight: 4 },
  }).addTo(focusLayer)
  if (regionFocus.value.status === "ready" && regionFocus.value.samples.length) {
    L.geoJSON({ type: "FeatureCollection", features: regionFocus.value.samples } as GeoJSON.FeatureCollection, {
      pane: "focusPane",
      style: { color: "#0369a1", opacity: 0.75, weight: 3 },
    }).addTo(focusLayer)
  }
  const anchor = focusedRegion.value.properties.map_anchor
  if (regionFocus.value.status === "ready" && anchor?.type === "Point") {
    L.marker([anchor.coordinates[1], anchor.coordinates[0]], {
      pane: "focusPane",
      icon: L.divIcon({
        className: "focus-count-marker",
        html: `<span>${regionFocus.value.totalCount}</span>`,
      }),
    }).addTo(focusLayer)
  }
}

async function loadRegionFocus(): Promise<void> {
  const focus = regionFocus.value
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
          selection: { type: "region", region_id: focus.regionId },
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
    if (stopped || request !== trackRequest || focus !== regionFocus.value) return
    focus.totalCount = results.reduce((total, result) => total + result.totalCount, 0)
    focus.samples = results.flatMap((result) => result.samples)
      .sort((left, right) =>
        String(left.properties.request_date).localeCompare(String(right.properties.request_date)) ||
        String(left.properties.track_id).localeCompare(String(right.properties.track_id)),
      )
      .slice(0, 20)
    focus.status = "ready"
  } catch (problem) {
    if (!stopped && request === trackRequest && focus === regionFocus.value && (problem as Error).name !== "AbortError") {
      focus.status = "error"
      focus.error = focusError(Number((problem as Error & { status?: number }).status ?? 500))
    }
  }
}

async function enterRegionFocus(regionId: number): Promise<void> {
  if (!map || regionFocus.value) return
  regionFocus.value = {
    regionId,
    selection: {
      ...selection.value,
      startHour: layer.value === "sequences" ? 6 : startHour.value,
      endHour: layer.value === "sequences" ? 10 : endHour.value,
    },
    status: "loading",
    error: null,
    totalCount: 0,
    samples: [],
    snapshot: {
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
  drawRegionFocus()
  await loadRegionFocus()
}

function exitRegionFocus(): void {
  const snapshot = regionFocus.value?.snapshot
  if (!snapshot) return
  trackRequest?.abort()
  focusLayer?.clearLayers()
  regionFocus.value = null
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
  if (event.key === "Escape" && regionFocus.value) exitRegionFocus()
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
  if (focusActive.value) return
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
      interactive: !focusActive.value,
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
      interactive: !focusActive.value,
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
  if (focusActive.value) return
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
      interactive: !focusActive.value,
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
        interactive: !focusActive.value,
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
  regionStatus.value = "loading"
  try {
    const response = await fetch("/api/regions")
    if (!response.ok) throw new Error("region request failed")
    const context = (await response.json()) as RegionContext
    if (stopped) return
    regionContext.value = context
    regionStatus.value = context.regions.features.length === 0 ? "empty" : "ready"
    initializeMap(context)
  } catch {
    if (!stopped) regionStatus.value = "error"
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

function reset(): void {
  layer.value = "source-sink"
  dateScope.value = "clear-days"
  startHour.value = 6
  endHour.value = 10
  aggregation.value = "average"
  flowMatrix.value = "od"
  flowSignificance.value = "significant"
  sequenceSupport.value = 0.001
  sequenceLimit.value = 20
  userAdjustedView = false
  fitIsland()
  if (regionStatus.value === "error") void loadRegions()
  if (healthStatus.value === "unavailable") void loadHealth()
}

watch([layer, dateScope, sequenceSupport, sequenceLimit], () => {
  if (layer.value === "sequences") void loadSequences()
  else {
    sequenceRequest?.abort()
    sequenceStatus.value = "idle"
    sequenceResponse.value = null
    selectedSequenceIndex.value = 0
  }
})
watch([layer, dateScope, startHour, endHour, flowMatrix, flowSignificance], () => {
  if (layer.value === "flows") void loadFlows()
  else {
    flowRequest?.abort()
    flowStatus.value = "idle"
    flowSlices.value = []
    selectedFlowKey.value = null
  }
})
watch([layer, flowStatus, topFlows, currentFlow, regionContext, focusActive], drawFlows)
watch([layer, flowStatus, districtFlows, currentFlow, regionContext], async () => {
  await nextTick()
  renderFlowChart()
}, { flush: "post" })
watch([layer, sequenceStatus, sequenceResponse, selectedSequenceIndex, regionContext, focusActive], drawSequences)

onMounted(() => {
  window.addEventListener("keydown", handleKeydown)
  void loadRegions()
  void loadHealth()
})

watch([layer, dateScope, startHour, endHour, aggregation], renderRegions)
watch(focusActive, renderRegions)
watch(regionFocus, drawRegionFocus, { deep: true })
watch(
  () => regionFocus.value && [regionFocus.value.selection.startHour, regionFocus.value.selection.endHour],
  (current, previous) => {
    if (current && previous && (current[0] !== previous[0] || current[1] !== previous[1])) {
      void loadRegionFocus()
    }
  },
)

onBeforeUnmount(() => {
  stopped = true
  flowRequest?.abort()
  sequenceRequest?.abort()
  trackRequest?.abort()
  window.removeEventListener("keydown", handleKeydown)
  flowChart?.dispose()
  resizeObserver?.disconnect()
  map?.remove()
  map = null
  regionLayer = null
})
</script>

<template>
  <main class="app-shell" :class="{ 'focus-active': focusActive }">
    <header :inert="focusActive">
      <div>
        <p class="eyebrow">课程数据演示</p>
        <h1>厦门本岛早高峰共享单车流动</h1>
      </div>

      <form class="toolbar" aria-label="全局工具栏" @submit.prevent>
        <label>
          内容图层
          <select v-model="layer" aria-label="内容图层" :disabled="focusActive">
            <option value="source-sink">源汇</option>
            <option value="flows">区域间流动</option>
            <option value="sequences">典型通勤链</option>
          </select>
        </label>

        <label>
          日期
          <select v-model="dateScope" aria-label="日期" :disabled="focusActive">
            <option value="clear-days">晴天集（12-21、12-22、12-24、12-25）</option>
            <option value="2020-12-21">2020-12-21</option>
            <option value="2020-12-22">2020-12-22</option>
            <option value="2020-12-23">2020-12-23（雨天）</option>
            <option value="2020-12-24">2020-12-24</option>
            <option value="2020-12-25">2020-12-25</option>
          </select>
        </label>

        <fieldset :disabled="analysisControlsDisabled || focusActive">
          <legend>连续整点范围</legend>
          <label>
            开始
            <input
              v-model.number="startHour"
              aria-label="开始时间"
              type="range"
              min="6"
              :max="endHour - 1"
              step="1"
              :disabled="analysisControlsDisabled || focusActive"
            />
          </label>
          <label>
            结束
            <input
              v-model.number="endHour"
              aria-label="结束时间"
              type="range"
              :min="startHour + 1"
              max="10"
              step="1"
              :disabled="analysisControlsDisabled || focusActive"
            />
          </label>
          <output>{{ String(startHour).padStart(2, "0") }}:00–{{ String(endHour).padStart(2, "0") }}:00</output>
        </fieldset>

        <label>
          聚合口径
          <select v-model="aggregation" aria-label="聚合口径" :disabled="analysisControlsDisabled || focusActive">
            <option value="average">跨日平均</option>
            <option value="sum">跨日合计</option>
          </select>
        </label>

        <fieldset v-if="layer === 'flows'" class="flow-context">
          <legend>流动上下文</legend>
          <label>
            矩阵
            <select v-model="flowMatrix" aria-label="流矩阵" :disabled="focusActive">
              <option value="od">出行流</option>
              <option value="channel">通道流</option>
            </select>
          </label>
          <label>
            显著性
            <select v-model="flowSignificance" aria-label="显著性" :disabled="focusActive">
              <option value="significant">仅显著</option>
              <option value="all">全部流对</option>
            </select>
          </label>
        </fieldset>

        <button type="button" :disabled="focusActive" @click="reset">重置</button>
      </form>
    </header>

    <section class="workspace">
      <section class="map-panel" aria-label="业务地图">
        <p v-if="regionStatus === 'loading'" class="map-status" role="status">正在加载区域地图…</p>
        <div v-if="regionStatus === 'error'" class="map-status" role="alert">
          <p>区域地图暂不可用。</p>
          <button type="button" @click="loadRegions">重试</button>
        </div>
        <p v-if="regionStatus === 'empty'" class="map-status" role="status">区域数据为空，共 0 个区域。</p>
        <p v-if="!basemapAvailable" class="basemap-status" role="status">外部底图不可用，本地业务地图仍可查看。</p>
        <aside v-if="layer === 'source-sink' && regionStatus === 'ready'" class="source-sink-legend" aria-label="净流入强度图例">
          <strong>净流入强度</strong>
          <span>源 − ｜ 平衡 0 ｜ 汇 +</span>
          <span>{{ sourceSinkLegendText }}</span>
          <span>灰色表示不可计算</span>
        </aside>
        <div id="map" ref="mapElement" />
      </section>

      <aside v-if="layer === 'flows'" class="flow-panel" aria-label="区域间流动" :inert="focusActive">
        <h2>区域间流动</h2>
        <p class="flow-scope">{{ aggregationNote(selection) }}</p>
        <p v-if="flowStatus === 'loading'" role="status">正在加载区域流…</p>
        <div v-else-if="flowStatus === 'error'" role="alert">
          <p>区域流暂不可用，区域地图仍可查看。</p>
          <button type="button" @click="loadFlows">重试</button>
        </div>
        <p v-else-if="flowStatus === 'empty'" role="status">当前条件下没有区域流。</p>
        <template v-else-if="flowStatus === 'ready'">
          <figure class="flow-chart-figure">
            <div ref="flowChartElement" class="flow-chart" aria-label="片区流动弦图" />
            <figcaption>片区弦图使用全部过滤后的非自环区域流。片区自连接表示不同区域之间的片区内部区域间流动。</figcaption>
          </figure>
          <h3>Top 50 区域流列表</h3>
          <ol class="flow-list" aria-label="Top 50 区域流列表">
            <li v-for="flow in topFlows" :key="flowKey(flow)">
              <button
                type="button"
                :aria-current="flowKey(flow) === flowKey(currentFlow ?? flow) ? 'true' : undefined"
                @click="selectFlow(flow)"
              >
                {{ regionCode(flow.from_region) }} → {{ regionCode(flow.to_region) }} · {{ formatMetric(flow.weight) }}（{{ aggregationNote(selection) }}）
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
            <dt>日期时段口径</dt>
            <dd>{{ aggregationNote(selection) }}</dd>
            <dt>已检验</dt>
            <dd>{{ currentFlow.is_tested ? "是" : "否" }}</dd>
            <dt>显著</dt>
            <dd>{{ currentFlow.is_significant === null ? "未检验" : currentFlow.is_significant ? "是" : "否" }}</dd>
            <dt>被门槛挡住</dt>
            <dd>{{ currentFlow.gated ? "是" : "否" }}</dd>
          </dl>
        </template>
      </aside>

      <aside v-if="layer === 'sequences'" class="sequence-panel" aria-label="典型通勤链" :inert="focusActive">
        <h2>典型通勤链</h2>
        <label>
          连续支持度门槛
          <select v-model.number="sequenceSupport" aria-label="连续支持度门槛">
            <option v-for="level in SUPPORT_LEVELS" :key="level" :value="level">{{ level }}</option>
          </select>
        </label>
        <label>
          Top-N
          <input
            :value="sequenceLimit"
            aria-label="Top-N"
            type="number"
            min="1"
            max="100"
            step="1"
            @change="setSequenceLimit"
          />
        </label>
        <p v-if="sequenceStatus === 'loading'" role="status">正在加载通勤链…</p>
        <div v-else-if="sequenceStatus === 'error'" role="alert">
          <p>通勤链暂不可用，区域地图仍可查看。</p>
          <button type="button" @click="loadSequences">重试</button>
        </div>
        <p v-else-if="sequenceStatus === 'empty'" role="status">当前条件下没有通勤链。</p>
        <template v-else-if="sequenceResponse">
          <ol class="sequence-list">
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
            <dt>绝对连续支持度门槛</dt>
            <dd>{{ sequenceResponse.min_contiguous_support_count }}</dd>
            <dt>有效轨迹分母</dt>
            <dd>{{ sequenceResponse.valid_tracks }}</dd>
          </dl>
          <p class="sequence-note">
            支持度按含该模式的区域序列条数计算，同一条序列内重复出现只计一次；一条有效轨迹可能贡献多条区域序列。绘图资格使用 API 已审计的连续支持度与区域相邻性，不受跨日平均或合计影响。
          </p>
        </template>
      </aside>

      <aside v-if="regionFocus && focusedProfile" class="focus-panel" aria-label="区域聚焦详情">
        <div class="focus-heading">
          <div>
            <p class="eyebrow">独立地理聚焦</p>
            <h2>{{ focusedProfile.region_code }}</h2>
          </div>
          <button type="button" @click="exitRegionFocus">退出聚焦</button>
        </div>

        <dl class="focus-details">
          <dt>选择类型</dt>
          <dd>区域</dd>
          <dt>区域编码</dt>
          <dd>{{ focusedProfile.region_code }}</dd>
          <dt>日期</dt>
          <dd>{{ regionFocus.selection.dateScope === "clear-days" ? "晴天集（4 日）" : regionFocus.selection.dateScope }}</dd>
          <dt>聚合口径</dt>
          <dd>{{ regionFocus.selection.aggregation === "average" ? "跨日平均（仅用于区域动态指标）" : "跨日合计" }}</dd>
          <dt>唯一有效轨迹总数</dt>
          <dd>{{ regionFocus.status === "ready" ? regionFocus.totalCount : "—" }}</dd>
          <dt>实际样例数</dt>
          <dd>{{ regionFocus.status === "ready" ? regionFocus.samples.length : "—" }}</dd>
        </dl>
        <p class="focus-sample-note">样例最多 20 条；样例几何保留 API 返回的完整查询时段上下文。</p>

        <fieldset class="focus-time">
          <legend>聚焦局部时间</legend>
          <label>
            开始
            <input
              v-model.lazy.number="regionFocus.selection.startHour"
              aria-label="聚焦开始时间"
              type="range"
              min="6"
              :max="regionFocus.selection.endHour - 1"
              step="1"
            />
          </label>
          <label>
            结束
            <input
              v-model.lazy.number="regionFocus.selection.endHour"
              aria-label="聚焦结束时间"
              type="range"
              :min="regionFocus.selection.startHour + 1"
              max="10"
              step="1"
            />
          </label>
          <output>{{ String(regionFocus.selection.startHour).padStart(2, "0") }}:00–{{ String(regionFocus.selection.endHour).padStart(2, "0") }}:00</output>
        </fieldset>

        <p v-if="regionFocus.status === 'loading'" role="status">正在查询有效轨迹…</p>
        <div v-else-if="regionFocus.status === 'error'" role="alert">
          <p>{{ regionFocus.error }}</p>
          <button type="button" @click="loadRegionFocus">重试</button>
        </div>
        <p v-else-if="regionFocus.totalCount === 0" role="status">当前条件下共有 0 条唯一有效轨迹，样例为空。</p>

        <section aria-label="区域画像">
          <h3>完整区域画像</h3>
          <div v-html="focusedProfileHtml" />
        </section>
      </aside>
    </section>

    <footer :inert="focusActive">
      <p>数据来源：厦门共享单车 GPS 与订单数据</p>
      <p>研究范围：2020-12-21 至 2020-12-25，06:00–10:00，厦门本岛</p>
      <p>结论边界：历史工作日早高峰研究结果，不代表实时、全天或厦门全市情况。</p>
      <p>组件状态：{{ healthStatus === "ok" ? "正常" : healthStatus === "loading" ? "检查中" : "部分不可用" }}</p>
      <ul v-if="Object.keys(healthComponents).length" class="health-components">
        <li v-for="[key, label] in HEALTH_COMPONENTS" :key="key">
          {{ label }}：{{ healthComponents[key] === "ok" ? "正常" : "不可用" }}
        </li>
      </ul>
      <details v-if="regionContext">
        <summary>活动发布摘要</summary>
        <code>{{ regionContext.release_digest }}</code>
      </details>
      <p class="attribution">
        地图数据 © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors，
        底图样式 © <a href="https://carto.com/attributions">CARTO</a>
      </p>
    </footer>
  </main>
</template>
