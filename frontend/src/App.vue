<script setup lang="ts">
import L, { type LayerGroup, type Map as LeafletMap, type TileLayer } from "leaflet"
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue"

type Layer = "source-sink" | "flows" | "sequences"
type DateScope =
  | "clear-days"
  | "2020-12-21"
  | "2020-12-22"
  | "2020-12-23"
  | "2020-12-24"
  | "2020-12-25"
type Aggregation = "average" | "sum"
type Bounds = { west: number; south: number; east: number; north: number }
type MapFeature = {
  properties: Record<string, unknown>
  geometry: object
}
type FeatureCollection = { type: "FeatureCollection"; features: MapFeature[] }
type RegionContext = {
  release_digest: string
  island_boundary: object
  island_bounds: Bounds
  map_bounds: Bounds
  districts: FeatureCollection
  regions: FeatureCollection
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
const sequenceSupport = ref<number>(0.001)
const sequenceLimit = ref(20)
const sequenceStatus = ref<"idle" | "loading" | "ready" | "empty" | "error">("idle")
const sequenceResponse = ref<SequenceResponse | null>(null)
const selectedSequenceIndex = ref(0)
const currentSequence = computed(
  () => sequenceResponse.value?.patterns[selectedSequenceIndex.value] ?? null,
)
const analysisControlsDisabled = computed(() => layer.value === "sequences")
const mapElement = ref<HTMLElement | null>(null)
const regionContext = ref<RegionContext | null>(null)
const regionStatus = ref<"loading" | "ready" | "empty" | "error">("loading")
const healthStatus = ref<"loading" | "ok" | "unavailable">("loading")
const healthComponents = ref<Record<string, string>>({})
const basemapAvailable = ref(true)

let map: LeafletMap | null = null
let tileLayer: TileLayer | null = null
let sequenceLayer: LayerGroup | null = null
let sequenceRequest: AbortController | null = null
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
  L.geoJSON(context.regions as GeoJSON.GeoJsonObject, {
    style: { color: "#94a3b8", fillOpacity: 0.08, weight: 0.8 },
  }).addTo(map)

  map.on("movestart", () => {
    if (!fittingView) userAdjustedView = true
  })
  resizeObserver = new ResizeObserver(() => {
    map?.invalidateSize()
    if (!userAdjustedView) fitIsland()
  })
  resizeObserver.observe(mapElement.value)
  fitIsland()
  drawSequences()
}

function districtName(id: number): string {
  const district = regionContext.value?.districts.features.find(
    (feature) => feature.properties.district_id === id,
  )
  return String(district?.properties.label ?? id)
}

function selectSequence(index: number, event?: L.LeafletMouseEvent): void {
  if (event?.originalEvent) L.DomEvent.stopPropagation(event.originalEvent)
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
      const coordinates = (feature.properties.map_anchor as { coordinates: [number, number] }).coordinates
      return [Number(feature.properties.region_id), [coordinates[1], coordinates[0]] as L.LatLngTuple]
    }),
  )
  sequenceResponse.value.patterns.forEach((pattern, index) => {
    const positions = pattern.region_ids.map((id) => anchors.get(id)).filter((point) => point !== undefined)
    if (positions.length !== pattern.region_ids.length) return
    const selected = index === selectedSequenceIndex.value
    L.polyline(positions, {
      color: selected ? "#c2410c" : "#64748b",
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
watch([layer, sequenceStatus, sequenceResponse, selectedSequenceIndex, regionContext], drawSequences)

onMounted(() => {
  void loadRegions()
  void loadHealth()
})

onBeforeUnmount(() => {
  stopped = true
  sequenceRequest?.abort()
  resizeObserver?.disconnect()
  map?.remove()
  map = null
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

    <section class="workspace">
      <section class="map-panel" aria-label="业务地图">
        <p v-if="regionStatus === 'loading'" class="map-status" role="status">正在加载区域地图…</p>
        <div v-if="regionStatus === 'error'" class="map-status" role="alert">
          <p>区域地图暂不可用。</p>
          <button type="button" @click="loadRegions">重试</button>
        </div>
        <p v-if="regionStatus === 'empty'" class="map-status" role="status">区域数据为空，共 0 个区域。</p>
        <p v-if="!basemapAvailable" class="basemap-status" role="status">外部底图不可用，本地业务地图仍可查看。</p>
        <div id="map" ref="mapElement" />
      </section>

      <aside v-if="layer === 'sequences'" class="sequence-panel" aria-label="典型通勤链">
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
