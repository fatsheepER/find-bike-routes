type Sample = GeoJSON.Feature<GeoJSON.Geometry, { track_id: string | number; request_date?: string }>
export const SAMPLE_COLORS = ['#2478b5', '#468bdd', '#7363cd', '#ce4c8c', '#c62635']

export function sampleOverlap(samples: Sample[]): GeoJSON.Feature<GeoJSON.LineString, { count: number }>[] {
  const edges = new Map<string, { coordinates: GeoJSON.Position[]; tracks: Set<string> }>()
  samples.forEach((sample) => {
    const lines = sample.geometry.type === 'LineString' ? [sample.geometry.coordinates]
      : sample.geometry.type === 'MultiLineString' ? sample.geometry.coordinates : []
    const track = `${sample.properties.request_date ?? ''}/${sample.properties.track_id}`
    for (const line of lines) for (let i = 1; i < line.length; i++) {
      const endpoints = [line[i - 1].slice(0, 2), line[i].slice(0, 2)]
      if (endpoints.some(point => point.length !== 2 || !point.every(Number.isFinite))) continue
      const keys = endpoints.map(point => point.join(','))
      if (keys[0] === keys[1]) continue
      // ponytail: identical endpoint pairs only; use road segment IDs to count partially shared edges.
      const key = keys.sort().join('|')
      const edge = edges.get(key) ?? { coordinates: endpoints, tracks: new Set<string>() }
      edge.tracks.add(track)
      edges.set(key, edge)
    }
  })
  return [...edges.values()].filter(edge => edge.tracks.size > 1)
    .sort((a, b) => a.tracks.size - b.tracks.size)
    .map(edge => ({ type: 'Feature', geometry: { type: 'LineString', coordinates: edge.coordinates }, properties: { count: edge.tracks.size } }))
}

export function sampleStroke(count: number, maximum: number) {
  const strength = maximum > 1 ? Math.max(0, Math.min(1, (count - 1) / (maximum - 1))) : 0
  return { color: SAMPLE_COLORS[Math.round(strength * (SAMPLE_COLORS.length - 1))], weight: 2 + 4 * strength, opacity: .85 + .1 * strength }
}
