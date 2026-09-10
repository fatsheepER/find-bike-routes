<script setup lang="ts">
import L, { type GeoJSON as LeafletGeoJSON, type Map as LeafletMap, type TileLayer } from "leaflet"
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue"
import {
  aggregateRegion,
  type Aggregation,
  type DateScope,
  type RegionFeature,
  type RegionSelection,
} from "./region-aggregation"
import {
  colorScaleLimit,
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

let map: LeafletMap | null = null
let tileLayer: TileLayer | null = null
let regionLayer: LeafletGeoJSON | null = null
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
  map.setMaxBounds(leafletBounds(context.map_bounds))
  tileLayer = L.tileLayer(
    "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png",
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
  userAdjustedView = false
  fitIsland()
  if (regionStatus.value === "error") void loadRegions()
  if (healthStatus.value === "unavailable") void loadHealth()
}

onMounted(() => {
  void loadRegions()
  void loadHealth()
})

watch([layer, dateScope, startHour, endHour, aggregation], renderRegions)

onBeforeUnmount(() => {
  stopped = true
  resizeObserver?.disconnect()
  map?.remove()
  map = null
  regionLayer = null
})
</script>

<template>
  <main class="app-shell">
    <header>
      <div>
        <p class="eyebrow">课程数据演示</p>
        <h1>厦门本岛早高峰共享单车流动</h1>
      </div>

      <form class="toolbar" aria-label="全局工具栏" @submit.prevent>
        <label>
          内容图层
          <select v-model="layer" aria-label="内容图层">
            <option value="source-sink">源汇</option>
            <option value="flows">区域间流动</option>
            <option value="sequences">典型通勤链</option>
          </select>
        </label>

        <label>
          日期
          <select v-model="dateScope" aria-label="日期">
            <option value="clear-days">晴天集（12-21、12-22、12-24、12-25）</option>
            <option value="2020-12-21">2020-12-21</option>
            <option value="2020-12-22">2020-12-22</option>
            <option value="2020-12-23">2020-12-23（雨天）</option>
            <option value="2020-12-24">2020-12-24</option>
            <option value="2020-12-25">2020-12-25</option>
          </select>
        </label>

        <fieldset :disabled="analysisControlsDisabled">
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
              :disabled="analysisControlsDisabled"
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
              :disabled="analysisControlsDisabled"
            />
          </label>
          <output>{{ String(startHour).padStart(2, "0") }}:00–{{ String(endHour).padStart(2, "0") }}:00</output>
        </fieldset>

        <label>
          聚合口径
          <select v-model="aggregation" aria-label="聚合口径" :disabled="analysisControlsDisabled">
            <option value="average">跨日平均</option>
            <option value="sum">跨日合计</option>
          </select>
        </label>

        <button type="button" @click="reset">重置</button>
      </form>
    </header>

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

    <footer>
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
