<template>
  <div class="abs-full overflow-hidden pointer-events-none" :style="{ zIndex }">
    <!-- vivid soft gradients -->
    <div class="abs-full vivid-gradients" :style="{ opacity }"></div>

    <!-- optional grain -->
    <div v-if="grain" class="abs-full noise-mask" :style="{ opacity: grainOpacity }"></div>

    <!-- vignette -->
    <div v-if="vignette" class="abs-full vignette"></div>
  </div>
</template>

<script setup>
const props = defineProps({
  zIndex: { type: [Number, String], default: 0 },
  opacity: { type: Number, default: 0.95 },
  vignette: { type: Boolean, default: true },
  grain: { type: Boolean, default: true },
  grainOpacity: { type: Number, default: 0.08 },
})
</script>

<style scoped>
.abs-full { position:absolute; inset:0 }

/* ===== Vivid gradients with your palette ===== */
.vivid-gradients {
  background:
    linear-gradient(120deg,
      #661313cc 0%,
      #661414cc 30%,
      #884554cc 60%,
      #e4ae87cc 85%,
      #daa8aecc 100%),
    linear-gradient(45deg,
      #e4ae87aa 0%,
      transparent 60%),
    linear-gradient(200deg,
      #884554aa 0%,
      transparent 70%),
    conic-gradient(from 160deg at 50% 50%,
      #66131399,
      #66141488,
      #88455488,
      #e4ae87aa,
      #daa8ae88,
      #66131399);

  filter: blur(18px) saturate(120%) contrast(108%);
  will-change: transform;
  animation: drift 55s ease-in-out infinite alternate;
}

@keyframes drift {
  0%   { transform: translate3d(0,0,0) scale(1) rotate(0deg); }
  25%  { transform: translate3d(-2%, 1.5%, 0) scale(1.025) rotate(2deg); }
  50%  { transform: translate3d(1.5%, -2%, 0) scale(1.03) rotate(-2deg); }
  75%  { transform: translate3d(-1%, 1.8%, 0) scale(1.02) rotate(1deg); }
  100% { transform: translate3d(2%, -1.5%, 0) scale(1.025) rotate(-1.5deg); }
}

/* vignette (linear only) */
.vignette {
  background:
    linear-gradient(to bottom, rgba(0,0,0,.45), transparent 25%, transparent 75%, rgba(0,0,0,.55)),
    linear-gradient(to right, rgba(0,0,0,.35), transparent 22%, transparent 78%, rgba(0,0,0,.4));
  mix-blend-mode: multiply;
}

/* grain */
.noise-mask {
  background:
    repeating-linear-gradient(0deg, rgba(255,255,255,.05) 0 1px, transparent 1px 2px),
    repeating-linear-gradient(90deg, rgba(255,255,255,.04) 0 1px, transparent 1px 2px);
  mix-blend-mode: overlay;
  animation: grain 2s steps(5) infinite;
}
@keyframes grain {
  0%,100% { transform: translate(0,0) }
  25%     { transform: translate(-0.6%, 0.4%) }
  50%     { transform: translate(0.6%, -0.4%) }
  75%     { transform: translate(0.4%, 0.6%) }
}

@media (prefers-reduced-motion: reduce) {
  .vivid-gradients, .noise-mask { animation: none; }
}
</style>
