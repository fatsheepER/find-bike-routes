import { expect, it } from 'vitest'
import { sampleOverlap, sampleStroke } from './focus-tracks'

it('counts distinct dated tracks on shared edges, ignoring direction, repeat visits and disconnected pieces', () => {
  const a = [118, 24], b = [118.01, 24], c = [118.02, 24], d = [118.03, 24]
  const sample = (id: number, date: string, lines: number[][][]): GeoJSON.Feature<GeoJSON.MultiLineString, { track_id: number; request_date: string }> => ({
    type: 'Feature', properties: { track_id: id, request_date: date }, geometry: { type: 'MultiLineString', coordinates: lines },
  })
  const first = sample(1, '2020-12-21', [[a, b, a, b], [c, d]])
  const edges = sampleOverlap([first, first, sample(2, '2020-12-21', [[b, a]]), sample(1, '2020-12-22', [[a, b]])])
  expect(edges).toEqual([{ type: 'Feature', properties: { count: 3 }, geometry: { type: 'LineString', coordinates: [a, b] } }])
  expect(sampleStroke(1, 3)).toEqual({ color: '#2478b5', weight: 2, opacity: .85 })
  expect(sampleStroke(3, 3)).toEqual({ color: '#c62635', weight: 6, opacity: .95 })
  expect(sampleOverlap([])).toEqual([])
  expect(sampleStroke(1, 1).color).toBe('#2478b5')
})
