<!-- src/pages/CreatePost.vue -->
<template>
  <main
    class="pb-mobile-nav relative mx-auto max-w-3xl px-4 py-8 text-neutral-200
           min-h-[100svh] pb-[calc(88px+env(safe-area-inset-bottom))] md:pb-10"
  >
    <!-- soft backdrop glow -->
    <div
      class="pointer-events-none absolute -inset-8 rounded-[32px]
             bg-[radial-gradient(60%_60%_at_0%_20%,rgba(102,20,20,.14),transparent_60%),radial-gradient(50%_60%_at_100%_80%,rgba(228,174,135,.12),transparent_60%)]">
    </div>

    <!-- Header -->
    <header
      class="relative mb-4 flex items-center justify-between rounded-2xl border border-white/10
             bg-neutral-900/50 px-4 py-3 shadow-[0_8px_28px_-14px_rgba(0,0,0,.65)]"
    >
      <h2 class="flex items-center gap-2 text-lg md:text-xl font-semibold tracking-tight">
        <span class="inline-block size-2 rounded-full bg-[#e4ae87] shadow-[0_0_12px_#e4ae87aa]"></span>
        Create Post
      </h2>

      <RouterLink
        to="/"
        class="group inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/5
               px-3 py-1.5 text-xs md:text-sm text-neutral-300 hover:text-white hover:bg-white/10 transition"
      >
        <svg viewBox="0 0 24 24" class="h-4 w-4" fill="currentColor"><path d="M15 18l-6-6 6-6"/></svg>
        Back to Feed
      </RouterLink>
    </header>

    <!-- Card -->
    <section class="relative overflow-hidden rounded-3xl border border-white/10 bg-black/40 shadow-[0_10px_36px_-14px_rgba(0,0,0,.75)]">
      <div class="pointer-events-none absolute -inset-[2px] rounded-[26px] neon-border"></div>
      <div class="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-[#e4ae87]/60 to-transparent"></div>

      <form class="relative z-10 grid gap-6 p-5 md:p-8" @submit.prevent="submit">
        <!-- vibe row (username removed; set from session) -->
        <div class="grid gap-4">
          <div class="space-y-1.5">
            <label class="text-sm text-neutral-300">Optional style prompt</label>
            <input
              v-model.trim="vibe"
              class="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2.5 text-neutral-100 placeholder:text-neutral-500 outline-none transition focus:ring-2 focus:ring-[#e4ae87]/30 focus:border-[#e4ae87]/60"
              placeholder='e.g., "cozy morning", "retro-future", "moody cinematic"'
            />
          </div>
        </div>

        <!-- image url -->
        <div class="space-y-1.5">
          <label class="text-sm text-neutral-300">Image URL</label>
          <input
            v-model.trim="imageUrl"
            class="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2.5 text-neutral-100 placeholder:text-neutral-500 outline-none transition focus:ring-2 focus:ring-[#e4ae87]/30 focus:border-[#e4ae87]/60"
            placeholder="https://..."
            inputmode="url"
            required
            @input="badImage=false"
          />
          <p class="text-xs text-neutral-400">Paste a direct image URL (upload is simulated for now).</p>
        </div>

        <!-- preview -->
        <div v-if="imageUrl" class="relative overflow-hidden rounded-2xl border border-white/10 bg-neutral-950/40">
          <img
            :src="imageUrl"
            alt=""
            class="w-full object-cover max-h-[60vh] transition-[transform,opacity] duration-300 ease-out"
            loading="lazy"
            decoding="async"
            @error="badImage=true"
            @load="badImage=false"
          />
          <div class="pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-black/50 to-transparent"></div>
          <p v-if="badImage" class="px-3 py-2 text-sm text-red-300 bg-red-500/10 border-t border-red-500/20">
            Couldn’t load that image URL.
          </p>
        </div>

        <!-- caption + actions -->
        <div class="space-y-2">
          <div class="flex items-center justify-between">
            <label class="text-sm text-neutral-300">Caption</label>
            <span class="text-xs text-neutral-400">{{ caption.length }}/2200</span>
          </div>

          <textarea
            v-model="caption"
            class="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2.5 text-neutral-100 placeholder:text-neutral-500 outline-none transition min-h-[120px] focus:ring-2 focus:ring-[#e4ae87]/30 focus:border-[#e4ae87]/60"
            maxlength="2200"
            placeholder="Write a caption or use 'Suggest'"
          />

          <div class="flex flex-wrap gap-2">
            <button
              class="inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-medium border border-white/15 text-neutral-100 bg-white/5 hover:bg-white/10 disabled:opacity-50 transition"
              type="button"
              @click="suggest"
              :disabled="!imageUrl || aiLoading"
              :title="aiLoading ? 'Generating…' : 'Generate caption & hashtags with AI'"
            >
              <span v-if="!aiLoading">Suggest</span>
              <span v-else class="inline-flex items-center gap-2"><span class="spinner"></span> Generating…</span>
            </button>

            <button
              class="inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-medium text-black bg-[#e4ae87] hover:bg-[#daa8ae] disabled:opacity-50 transition ml-auto"
              type="submit"
              :disabled="!canSubmit || submitting || badImage"
            >
              <span v-if="!submitting">Post</span>
              <span v-else class="inline-flex items-center gap-2"><span class="spinner"></span> Posting…</span>
            </button>
          </div>
        </div>

        <!-- hashtags -->
        <div v-if="hashtags.length" class="space-y-1">
          <label class="text-sm text-neutral-300">Hashtags</label>
          <div class="flex flex-wrap gap-2">
            <button
              v-for="t in hashtags"
              :key="t"
              type="button"
              class="inline-flex items-center gap-1 rounded-full border border-white/15 bg-white/5 px-3 py-1.5 text-xs text-neutral-100 hover:bg-white/10 transition"
              @click="removeTag(t)"
              :title="'Remove ' + t"
            >
              {{ t }} <span class="opacity-70">✕</span>
            </button>
          </div>
        </div>

        <!-- analysis chips -->
        <div v-if="analysisLabels.length" class="space-y-1">
          <label class="text-sm text-neutral-300">Detected</label>
          <div class="flex flex-wrap gap-2">
            <button
              v-for="l in analysisLabels"
              :key="l"
              type="button"
              class="inline-flex items-center gap-1 rounded-full border border-white/15 bg-transparent px-3 py-1.5 text-xs text-neutral-100 hover:bg-white/5 transition"
              @click="addHash('#' + kebab(l))"
              :title="'Add #' + kebab(l)"
            >
              {{ l }}
            </button>
          </div>
        </div>
      </form>
    </section>
  </main>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRouter, RouterLink } from 'vue-router'

const router = useRouter()

// Prefer VITE_API_BASE, fallback to older var name, then localhost
const API_BASE = (import.meta.env.VITE_API_BASE || import.meta.env.VITE_API_URL || '').replace(/\/+$/, '');

// -------- state --------
const imageUrl = ref('')
const caption  = ref('')
const vibe     = ref('')

const hashtags = ref([])         // '#...' strings
const analysisLabels = ref([])   // raw labels -> saved as tags

const badImage = ref(false)
const aiLoading = ref(false)
const submitting = ref(false)

// -------- helpers --------
const kebab = s => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
const addHash = (t) => { if (t?.startsWith('#') && !hashtags.value.includes(t)) hashtags.value.push(t) }
const removeTag = (t) => { hashtags.value = hashtags.value.filter(x => x !== t) }

const getCurrentUser = () => {
  try { return JSON.parse(localStorage.getItem('fg.user') || 'null')?.username || '' }
  catch { return '' }
}

const canSubmit = computed(() => Boolean(imageUrl.value && caption.value && !aiLoading.value && !badImage.value))

// -------- AI suggest --------
const suggest = async () => {
  if (!imageUrl.value || aiLoading.value) return
  aiLoading.value = true
  try {
    const r = await fetch(`${API_BASE}/api/ai/suggest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageUrl: imageUrl.value, prompt: vibe.value || null }),
    })
    const data = await r.json()
    if (!r.ok) throw new Error(data?.error?.message || 'AI error')

    if (Array.isArray(data.captions) && data.captions.length) caption.value = data.captions[0]
    else if (data.caption) caption.value = data.caption

    if (Array.isArray(data.hashtags)) hashtags.value = data.hashtags

    const objs = (data.analysis?.objects || []).map(o => o.label)
    const scns = (data.analysis?.scenes || []).map(s => s.label)
    analysisLabels.value = [...new Set([...objs.slice(0,3), ...scns.slice(0,3)])]
  } catch (e) {
    alert('AI suggestion failed. You can still post manually.')
  } finally {
    aiLoading.value = false
  }
}

// -------- submit --------
const submit = async () => {
  if (!canSubmit.value) return
  submitting.value = true
  try {
    // require login; fallback to demo user "bellaswan" if none (per your requirement)
    const username = getCurrentUser() || 'bellaswan'
    if (!getCurrentUser()) {
      // soft prompt to log in, but still honor "assume bellaswan" for now
      // router.push('/auth') // uncomment to force auth
    }

    // Detected -> tags (kebab), Hashtags stay as '#...'
    const tags = analysisLabels.value
      .map(l => kebab(l))
      .filter(Boolean)

    const payload = {
      username,
      imageUrl: imageUrl.value,
      caption: caption.value,        // keep clean; do not append hashtags here
      hashtags: [...hashtags.value], // '#...' as array
      tags,                          // plain tags for DB
      // also include labels for backward-compat if server only reads labels
      labels: tags
    }

    const r = await fetch(`${API_BASE}/api/posts`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-user': username
      },
      body: JSON.stringify(payload)
    })
    const post = await r.json()
    if (!r.ok) throw new Error(post?.error?.message || 'Create failed')

    router.push(`/p/${post.id}`)
  } catch (e) {
    alert(e.message || 'Failed to create post.')
  } finally {
    submitting.value = false
  }
}
</script>

<style scoped>
/* steps */
.step{
  display:inline-flex; align-items:center; gap:.5rem;
  padding:.35rem .6rem; border-radius:9999px;
  border:1px solid rgba(255,255,255,.12);
  background: rgba(255,255,255,.04);
}
.step > span{
  display:inline-grid; place-items:center;
  width:1.1rem; height:1.1rem; border-radius:9999px;
  background:#1b1b1b; color:#fff; font-weight:700; font-size:.7rem;
  border:1px solid rgba(255,255,255,.12);
}
.step.active{
  background: linear-gradient(180deg, rgba(228,174,135,.18), rgba(255,255,255,.06));
  border-color: rgba(228,174,135,.55);
}

/* neon frame */
.neon-border{
  background: linear-gradient(120deg,#661313,#661414 30%,#884554 55%,#e4ae87 80%,#daa8ae);
  filter: blur(6px);
  opacity:.22;
  animation: hue 18s linear infinite;
}

/* tiny spinner */
.spinner{
  width:12px;height:12px;border-radius:9999px;
  border:2px solid rgba(255,255,255,.35);
  border-top-color:#e4ae87;
  animation: spin 1s linear infinite;
}

/* keyframes */
@keyframes hue { from{ filter:hue-rotate(0deg) blur(6px)} to{ filter:hue-rotate(360deg) blur(6px)} }
@keyframes spin { from{ transform:rotate(0)} to{ transform:rotate(360deg)} }
</style>
