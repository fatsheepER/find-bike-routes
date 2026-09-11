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
  const lightness = Math.round(96 - Math.min(1, Math.abs(value) / limit) * 26)
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

export function directionRose(sectors: number[] | null): string {
  if (sectors === null) return '<p class="muted">暂无方向数据</p>'
  const maximum = Math.max(...sectors)
  const points = sectors.map((value, index) => {
    const angle = index * Math.PI / 8 - Math.PI / 2
    const radius = maximum > 0 ? 72 * value / maximum : 0
    return `${(100 + Math.cos(angle) * radius).toFixed(2)},${(100 + Math.sin(angle) * radius).toFixed(2)}`
  })
  const wedges = points.map((point, index) =>
    `<polygon points="100,100 ${point} ${points[(index + 1) % points.length]}" fill="${index % 2 ? '#9bb9c7' : '#48798f'}" />`,
  ).join("")
  return `<figure class="direction-rose"><svg viewBox="0 0 200 200" role="img" aria-label="16 扇区方向玫瑰">
    <circle cx="100" cy="100" r="72" fill="none" stroke="#e7e7e7" />
    <circle cx="100" cy="100" r="36" fill="none" stroke="#e7e7e7" />
    ${wedges}<polygon points="${points.join(' ')}" fill="none" stroke="#48798f" stroke-width="1.5" />
    <g text-anchor="middle" fill="#888" font-size="10"><text x="100" y="16">北</text><text x="190" y="104">东</text><text x="100" y="192">南</text><text x="10" y="104">西</text></g>
    </svg><figcaption>方向分布</figcaption></figure>`
}

export function sourceSinkTooltip(region: AggregatedRegion): string {
  return `<div class="source-sink-tooltip"><strong>${escapeHtml(region.region_code)}</strong><span>${formatMetric(region.net_inflow_per_km2)}<small> 单/km²</small></span></div>`
}

export function regionProfile(region: AggregatedRegion, selection: RegionSelection, districtLabel: string): string {
  const composition = region.functional_composition
  const rows = (items: [string, string][]) => `<dl>${items.map(([label, value]) => `<dt>${label}</dt><dd>${value}</dd>`).join('')}</dl>`
  return `<div class="region-profile">
    <p class="muted">${escapeHtml(districtLabel)} · ${formatMetric(region.area_km2)} km²</p>
    <div class="hero-metric"><span>净流入强度</span><strong>${formatMetric(region.net_inflow_per_km2)}<small> 单/km²</small></strong></div>
    <p class="metric-scope">${aggregationNote(selection)}</p>
    <section><h3>骑行活动</h3>${rows([
      ['解锁', formatMetric(region.unlocks)], ['上锁', formatMetric(region.locks)], ['净流入', formatMetric(region.net_inflow)],
      ['订单事件密度', `${formatMetric(region.order_events_per_km2)} 单/km²`], ['访问轨迹', formatMetric(region.tracks_visiting)],
      ['过境轨迹', formatMetric(region.tracks_transit)], ['过境率', region.pi_r === null ? '不可计算' : percentage(region.pi_r)],
    ])}</section>
    <section>${directionRose(region.sectors)}${rows([
      ['方向集中度', formatMetric(region.r)], ['轴向集中度', formatMetric(region.r_axial)],
      ['方向角', degrees(region.mean_bearing_deg)], ['轴向角', degrees(region.axis_bearing_deg)], ['过境弦', formatMetric(region.chords)],
    ])}</section>
    <section><h3>功能构成</h3>${rows([
      ['住宅', percentage(composition.residential)], ['就业', percentage(composition.employment)],
      ['教育', percentage(composition.education)], ['交通', percentage(composition.transport)],
      ['已分类面积', percentage(region.classified_share)], ['公交站密度', `${formatMetric(region.bus_stops_per_km2)} 站/km²`],
    ])}</section></div>`
}
