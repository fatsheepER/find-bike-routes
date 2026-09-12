<script setup lang="ts">
import { computed, ref } from 'vue'
const props = withDefaults(defineProps<{
  min: number; max: number; start: number; end: number; gap?: number
  startLabel: string; endLabel: string; startText?: string; endText?: string; disabled?: boolean; unfilled?: boolean
}>(), { gap: 0 })
const emit = defineEmits<{ change: [start: number, end: number] }>()
const position = (value: number) => `${(value - props.min) / (props.max - props.min) * 100}%`
const fill = computed(() => ({ left: position(props.start), right: position(props.max - props.end + props.min) }))
function selectValue(value: number, edge: 'start' | 'end'): number {
  const start = edge === 'start' ? Math.min(value, props.end - props.gap) : props.start
  const end = edge === 'end' ? Math.max(value, props.start + props.gap) : props.end
  emit('change', start, end)
  return edge === 'start' ? start : end
}
function update(event: Event, edge: 'start' | 'end') {
  const input = event.target as HTMLInputElement
  input.value = String(selectValue(Number(input.value), edge))
}
const drag = ref<{ edge: 'start' | 'end' | null; origin: number } | null>(null)
function pointerValue(event: PointerEvent): number {
  const rect = (event.currentTarget as HTMLElement).getBoundingClientRect()
  return Math.max(props.min, Math.min(props.max, Math.round(props.min + (event.clientX - rect.left - 4) / Math.max(1, rect.width - 8) * (props.max - props.min))))
}
function beginDrag(event: PointerEvent) {
  if (props.disabled || event.button > 0) return
  const value = pointerValue(event)
  drag.value = { origin: props.start, edge: props.start === props.end ? null : Math.abs(value - props.start) < Math.abs(value - props.end) ? 'start' : 'end' }
  ;(event.currentTarget as HTMLElement).setPointerCapture?.(event.pointerId)
  moveDrag(event)
}
function moveDrag(event: PointerEvent) {
  if (!drag.value) return
  const value = pointerValue(event)
  if (!drag.value.edge && value !== drag.value.origin) drag.value.edge = value < drag.value.origin ? 'start' : 'end'
  const edge = drag.value.edge ?? 'end'
  ;(event.currentTarget as HTMLElement).querySelector<HTMLInputElement>(`.range-${edge}`)?.focus()
  selectValue(value, edge)
}
function endDrag() { drag.value = null }
</script>
<template>
  <div class="range-control" :class="{ 'is-disabled': disabled }" @pointerdown.prevent="beginDrag" @pointermove="moveDrag" @pointerup="endDrag" @pointercancel="endDrag" @lostpointercapture="endDrag">
    <div class="range-track"><span v-if="!unfilled" :style="fill" /></div>
    <input type="range" :aria-label="startLabel" :aria-valuetext="startText" :aria-valuemin="min" :aria-valuemax="end - gap"
      :min="min" :max="max" :value="start" :disabled="disabled" step="1" class="range-start"
      :style="{ zIndex: start === max ? 3 : 1 }" @input="update($event, 'start')" />
    <input type="range" :aria-label="endLabel" :aria-valuetext="endText" :aria-valuemin="start + gap" :aria-valuemax="max"
      :min="min" :max="max" :value="end" :disabled="disabled" step="1" class="range-end"
      @input="update($event, 'end')" />
  </div>
</template>
