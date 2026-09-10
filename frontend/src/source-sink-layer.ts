import { datesForScope, type AggregatedRegion, type RegionSelection } from "./region-aggregation"

export function formatMetric(value: number | null, maximumFractionDigits = 3): string {
  return value === null || !Number.isFinite(value)
    ? "不可计算"
    : (Object.is(value, -0) ? 0 : value).toLocaleString("zh-CN", { maximumFractionDigits })
}

export function colorScaleLimit(regions: AggregatedRegion[]): number | null {
  return regions.reduce<number | null>((limit, region) => {
    const value = region.net_inflow_per_km2
    return value === null || !Number.isFinite(value) ? limit : Math.max(limit ?? 0, Math.abs(value))
  }, null)
}

export function sourceSinkColor(value: number | null, limit: number | null): string {
  if (value === null || limit === null || !Number.isFinite(value)) return "#cbd5e1"
  if (value === 0 || limit === 0) return "#f8fafc"
  const lightness = Math.round(92 - Math.min(1, Math.abs(value) / limit) * 48)
  return `hsl(${value > 0 ? 199 : 347} 72% ${lightness}%)`
}

export function sourceSinkState(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "不可计算"
  if (value === 0) return "平衡 0"
  return `${value > 0 ? "汇 +" : "源 −"}${formatMetric(Math.abs(value))}`
}

export function aggregationNote(selection: RegionSelection): string {
  const dates = datesForScope(selection.dateScope)
  const hours = `${String(selection.startHour).padStart(2, "0")}:00–${String(selection.endHour).padStart(2, "0")}:00`
  const mode = dates.length === 1 ? "单日" : `${dates.length} 日${selection.aggregation === "average" ? "平均" : "合计"}`
  return `${mode}，${hours} 所选时段累计`
}

export function sourceSinkLegend(limit: number | null, selection: RegionSelection): string {
  const dateLabel = selection.dateScope === "clear-days" ? "晴天集" : selection.dateScope
  const bounds = limit === null ? "无可计算数值" : `${formatMetric(-limit)} 至 +${formatMetric(limit)} 单/平方公里`
  return `${bounds}；${dateLabel}；${aggregationNote(selection)}`
}

function percentage(value: number): string {
  return `${formatMetric(value * 100, 1)}%`
}

function degrees(value: number | null): string {
  return value === null ? "不可计算" : `${formatMetric(value)}°`
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character] as string)
}

function directionRose(sectors: number[] | null): string {
  if (sectors === null) return "<p>方向玫瑰（16 扇区） 不可计算</p>"
  const maximum = Math.max(...sectors)
  const spokes = sectors.map((value, index) => {
    const angle = index * Math.PI / 8 - Math.PI / 2
    const radius = maximum > 0 ? 38 * value / maximum : 0
    const x = (50 + Math.cos(angle) * radius).toFixed(2)
    const y = (50 + Math.sin(angle) * radius).toFixed(2)
    return `<line x1="50" y1="50" x2="${x}" y2="${y}" />`
  }).join("")
  return `<figure class="direction-rose"><svg viewBox="0 0 100 100" role="img" aria-label="16 扇区方向玫瑰">${spokes}</svg><figcaption>方向玫瑰（16 扇区） ${sectors.map((value) => formatMetric(value)).join(" / ")}</figcaption></figure>`
}

export function sourceSinkTooltip(region: AggregatedRegion, selection: RegionSelection, districtLabel: string): string {
  const composition = region.functional_composition
  return `
    <div class="source-sink-tooltip">
      <strong>${escapeHtml(region.region_code)}</strong>
      <p>片区 ${escapeHtml(districtLabel)}；面积 ${formatMetric(region.area_km2)} 平方公里</p>
      <p class="metric-scope">${aggregationNote(selection)}</p>
      <p>源汇状态 ${sourceSinkState(region.net_inflow_per_km2)} 单/平方公里</p>
      <p>解锁 ${formatMetric(region.unlocks)}；上锁 ${formatMetric(region.locks)}；净流入 ${formatMetric(region.net_inflow)}；净流入强度 ${formatMetric(region.net_inflow_per_km2)} 单/平方公里</p>
      <p>订单事件密度 ${formatMetric(region.order_events_per_km2)} 单/平方公里；访问轨迹 ${formatMetric(region.tracks_visiting)}；过境轨迹 ${formatMetric(region.tracks_transit)}；PI_r ${formatMetric(region.pi_r)}</p>
      <p>过境弦 ${formatMetric(region.chords)}；R ${formatMetric(region.r)}；R_axial ${formatMetric(region.r_axial)}；方向角 ${degrees(region.mean_bearing_deg)}；轴向角 ${degrees(region.axis_bearing_deg)}</p>
      ${directionRose(region.sectors)}
      <p>静态画像（不随筛选变化）：住宅 ${percentage(composition.residential)}；就业 ${percentage(composition.employment)}；教育 ${percentage(composition.education)}；交通 ${percentage(composition.transport)}；已分类面积 ${percentage(region.classified_share)}；公交站密度 ${formatMetric(region.bus_stops_per_km2)} 站/平方公里</p>
    </div>
  `
}
