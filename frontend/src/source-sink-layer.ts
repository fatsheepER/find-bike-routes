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
  return `hsl(${value > 0 ? 347 : 199} 72% ${lightness}%)`
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
    </svg></figure>`
}

export const COMPOSITION_COLORS = {
  residential: "#95acc6",
  employment: "#cb9aa9",
  education: "#95bba7",
  transport: "#bab4ab",
} as const

const COMPOSITION_LABELS: [keyof typeof COMPOSITION_COLORS, string][] = [
  ["residential", "住宅"], ["employment", "就业"], ["education", "教育"], ["transport", "交通"],
]

export function compositionBar(composition: AggregatedRegion["functional_composition"]): string {
  const parts = COMPOSITION_LABELS.map(([key, label]) => {
    const share = composition[key]
    return { label, color: COMPOSITION_COLORS[key], share: share !== null && Number.isFinite(share) ? Math.max(0, share) : 0 }
  })
  const total = parts.reduce((sum, part) => sum + part.share, 0)
  if (total === 0) return '<p class="muted">暂无已分类面积构成</p>'
  const segments = parts.filter((part) => part.share > 0).map((part) =>
    `<i style="flex: ${(part.share / total).toFixed(6)}; background: ${part.color}" title="${part.label} ${percentage(part.share / total)}"></i>`,
  ).join("")
  const keys = parts.map((part) =>
    `<span><i style="background: ${part.color}"></i>${part.label}<b>${percentage(part.share / total)}</b></span>`,
  ).join("")
  const description = parts.map((part) => `${part.label} ${percentage(part.share / total)}`).join("，")
  return `<div class="composition-legend">
    <div class="composition-bar" role="img" aria-label="已分类面积构成：${description}">${segments}</div>
    <div class="composition-keys">${keys}</div>
  </div>`
}

export function sourceSinkTooltip(region: AggregatedRegion): string {
  return `<div class="source-sink-tooltip"><strong>${escapeHtml(region.region_code)}</strong><span style="background:${region.net_inflow_per_km2 === null ? '#eeeeee' : region.net_inflow_per_km2 > 0 ? '#f6cfd8' : region.net_inflow_per_km2 < 0 ? '#d1e5ef' : '#eeeeee'}">${region.net_inflow_per_km2 !== null && region.net_inflow_per_km2 > 0 ? '+' : ''}${formatMetric(region.net_inflow_per_km2, 2)}<small> 单/km²</small></span></div>`
}

export function regionProfile(region: AggregatedRegion, selection: RegionSelection): string {
  const composition = region.functional_composition
  const rows = (items: [string, string][]) => `<dl>${items.map(([label, value]) => `<dt>${label}</dt><dd>${value}</dd>`).join('')}</dl>`
  return `<div class="region-profile">
    <div class="hero-metric"><span>净流入强度</span><strong>${formatMetric(region.net_inflow_per_km2)}<small> 单/km²</small></strong></div>
    <p class="metric-scope">${aggregationNote(selection)}</p>
    <section><h3>骑行活动</h3>${rows([
      ['解锁', formatMetric(region.unlocks)], ['上锁', formatMetric(region.locks)], ['净流入', formatMetric(region.net_inflow)],
      ['订单事件密度', `${formatMetric(region.order_events_per_km2)} 单/km²`], ['访问轨迹', formatMetric(region.tracks_visiting)],
      ['过境轨迹', formatMetric(region.tracks_transit)], ['过境率', region.pi_r === null ? '不可计算' : percentage(region.pi_r)],
    ])}</section>
    <section><h3>方向分布</h3>${directionRose(region.sectors)}${rows([
      ['方向集中度', formatMetric(region.r)], ['轴向集中度', formatMetric(region.r_axial)],
      ['方向角', degrees(region.mean_bearing_deg)], ['轴向角', degrees(region.axis_bearing_deg)], ['过境弦', formatMetric(region.chords)],
    ])}</section>
    <section><h3>功能构成</h3>${rows([
      ['已分类面积', percentage(region.classified_share)], ['公交站密度', `${formatMetric(region.bus_stops_per_km2)} 站/km²`],
    ])}${compositionBar(composition)}</section></div>`
}
