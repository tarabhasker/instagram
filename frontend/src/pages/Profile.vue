<!-- src/pages/Profile.vue -->
<template>
  <!-- Solid bg; BeamsBackground handles animated layers -->
  <main class="relative min-h-[100svh] bg-neutral-950 text-neutral-200 overflow-hidden">
    <BeamsBackground class="absolute inset-0 z-0" :speed="70" :opacity="0.7" :blur="7" />

    <!-- Content wrapper -->
    <div class="relative z-10 mx-auto max-w-6xl px-4 py-8 md:py-12">
      <!-- ===== HEADER ===== -->
      <section
        class="relative overflow-hidden rounded-3xl border border-white/10 bg-black/40 shadow-[0_8px_28px_-12px_rgba(0,0,0,.7)]"
      >
        <!-- neon gradient border glow -->
        <div class="absolute inset-0 pointer-events-none">
          <div class="absolute -inset-[2px] rounded-[26px] neon-border"></div>
        </div>
        <div class="pointer-events-none absolute inset-0 opacity-[.04] mix-blend-screen scanline"></div>

        <!-- subtle blobs -->
        <div class="pointer-events-none absolute -top-24 -left-24 size-72 opacity-25 rounded-full"
             style="background: radial-gradient(60% 60% at 50% 50%, #66131344, transparent 70%);"></div>
        <div class="pointer-events-none absolute -bottom-24 -right-24 size-80 opacity-25 rounded-full"
             style="background: radial-gradient(60% 60% at 50% 50%, #88455444, transparent 70%);"></div>

        <!-- faint circuit lines -->
        <svg class="pointer-events-none absolute inset-0 opacity-[.12]" aria-hidden="true">
          <defs>
            <linearGradient id="circuit" x1="0" x2="1">
              <stop offset="0" stop-color="#661313"/>
              <stop offset=".5" stop-color="#661414"/>
              <stop offset="1" stop-color="#e4ae87"/>
            </linearGradient>
          </defs>
          <g stroke="url(#circuit)" stroke-width="1" fill="none">
            <path d="M0,40 L220,40 L260,80" />
            <path d="M100,0 L100,120 L160,160" />
            <path d="M100,120 L20,200" />
            <path d="M100,120 L180,220" />
            <path d="M100,120 L120,260" />
          </g>
        </svg>

        <!-- CONTENT -->
        <div class="relative z-10 px-5 py-6 md:px-8 md:py-8">
          <div class="flex flex-col gap-6 md:flex-row md:items-center">
            <!-- Avatar + glow + change overlay -->
            <div class="mx-auto md:mx-0">
              <div class="relative h-32 w-32 md:h-36 md:w-36 group">
                <div class="pointer-events-none absolute -inset-[5px] rounded-full ring-2 ring-[#661414]/35"></div>
                <div class="pointer-events-none absolute -inset-[8px] rounded-full neon-ring"></div>

                <img
                  :src="(user?.avatarUrl) || fallbackAvatar"
                  :alt="`${displayName} avatar`"
                  class="relative h-full w-full rounded-full object-cover shadow-xl select-none"
                  loading="lazy"
                  decoding="async"
                />

                <!-- Only show change control on your own profile -->
                <button
                  v-if="isOwnProfile"
                  class="absolute inset-0 hidden place-items-center rounded-full bg-black/45 text-white text-sm font-semibold
                         border border-white/15 backdrop-blur-[2px] group-hover:grid transition"
                  title="Change avatar"
                  @click="openAvatarModal"
                >
                  Change
                </button>

                <span class="pointer-events-none absolute -right-2 bottom-4 size-3.5 rounded-full bg-[#e4ae87] shadow-[0_0_22px_3px_#e4ae87AA]"></span>
              </div>
            </div>

            <!-- Identity & actions -->
            <div class="flex-1">
              <div class="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
                <div>
                  <div class="flex items-center gap-3">
                    <h1 class="text-3xl md:text-4xl font-semibold tracking-tight text-white">
                      {{ displayName }}
                    </h1>
                    <span
                      class="inline-flex items-center gap-1 rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] uppercase tracking-wider text-[#daa8ae]"
                    >
                      <svg viewBox="0 0 24 24" class="h-3.5 w-3.5" fill="currentColor">
                        <path d="M9 16.2 4.8 12l1.4-1.4L9 13.4l8.8-8.8L19.2 6z"/>
                      </svg>
                      trusted
                    </span>
                  </div>
                  <p class="mt-1 text-sm text-neutral-300/90">@{{ route.params.username }}</p>
                  <p v-if="user?.bio" class="mt-2 max-w-prose text-neutral-200/90">{{ user.bio }}</p>
                  <p v-if="user?.dateRegistered" class="mt-2 text-xs text-neutral-400">
                    Joined {{ formatJoined(user.dateRegistered) }}
                  </p>
                </div>

                <!-- Actions (hide on your own profile) -->
                <div class="flex items-center gap-2" v-if="!isOwnProfile">
                  <button
                    class="group relative overflow-hidden rounded-xl px-4 py-2 text-sm font-medium text-black bg-[#e4ae87] hover:bg-[#daa8ae] transition active:scale-[.98] will-change-transform"
                  >
                    <span class="relative z-10">Follow</span>
                    <span class="absolute inset-0 -z-0 opacity-0 group-hover:opacity-100 transition"
                          style="background: radial-gradient(80% 80% at 50% 0%, #e4ae8722, transparent 60%);"></span>
                  </button>
                  <button
                    class="relative rounded-xl border border-white/15 bg-white/5 px-4 py-2 text-sm text-neutral-100 hover:bg-white/10 transition"
                  >
                    Message
                  </button>
                  <button
                    class="relative grid size-9 place-items-center rounded-xl border border-white/15 bg-white/5 hover:bg-white/10 transition"
                    title="More"
                  >
                    <svg viewBox="0 0 24 24" class="h-5 w-5" fill="currentColor">
                      <path d="M12 8a2 2 0 110-4 2 2 0 010 4zm0 6a2 2 0 110-4 2 2 0 010 4zm0 6a2 2 0 110-4 2 2 0 010 4z"/>
                    </svg>
                  </button>
                </div>
              </div>

              <!-- Stats -->
              <div class="mt-5 grid grid-cols-3 gap-3">
                <div class="stat-chip">
                  <p class="stat-val">{{ followersCount }}</p>
                  <p class="stat-label">Followers</p>
                </div>
                <div class="stat-chip">
                  <p class="stat-val">{{ followingCount }}</p>
                  <p class="stat-label">Following</p>
                </div>
                <div class="stat-chip">
                  <p class="stat-val">{{ totalLikesReceived }}</p>
                  <p class="stat-label">Likes</p>
                </div>
              </div>
            </div>
          </div>

          <!-- bottom divider -->
          <div class="relative mt-6 h-px w-full overflow-hidden">
            <div class="absolute inset-0 bg-gradient-to-r from-transparent via-[#e4ae87]/45 to-transparent"></div>
            <div class="absolute -inset-x-10 inset-y-0 pointer-events-none animate-[glide_5s_linear_infinite] bg-gradient-to-r from-transparent via-white/25 to-transparent"></div>
          </div>
        </div>
      </section>

      <!-- ===== TABS ===== -->
      <nav class="mt-8 relative">
        <div class="pointer-events-none absolute -inset-2 rounded-2xl tab-glow"></div>

        <div class="relative z-10 flex items-center gap-2 p-1.5 rounded-2xl border border-white/10 bg-black/35">
          <button
            v-for="t in tabs"
            :key="t.key"
            :ref="el => el && tabRefs.set(t.key, el)"
            @click="setTab(t.key)"
            class="tab-chip group will-change-transform"
            :class="activeTab === t.key ? 'tab-chip--active' : ''"
          >
            <span class="relative z-10 flex items-center gap-2">
              <span v-html="t.icon" class="h-4 w-4 opacity-80"></span>
              {{ t.label }}
              <span class="tab-count" v-if="t.key==='posts'">{{ posts.length }}</span>
              <span class="tab-count" v-else-if="t.key==='liked'">{{ liked.length }}</span>
              <span class="tab-count" v-else>{{ saved.length }}</span>
            </span>
            <span class="tab-wash"></span>
          </button>

          <div class="ml-auto hidden md:flex items-center gap-2 text-[11px] tracking-wider text-neutral-400 uppercase">
            <span class="h-2 w-2 rounded-full bg-[#e4ae87] shadow-[0_0_10px_#e4ae87AA]"></span>
            {{ activeTab }} view
          </div>
        </div>

        <transition name="snap">
          <div v-if="activeTab" class="tab-underline" :style="underlineStyle"></div>
        </transition>
      </nav>

      <!-- ===== CONTENT ===== -->
      <section class="mt-6 relative">
        <div class="pointer-events-none absolute -inset-4 rounded-3xl content-glow"></div>

        <!-- Loading -->
        <div v-if="isLoadingAny" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 md:gap-4">
          <div v-for="i in 8" :key="i" class="skeleton-card">
            <div class="skeleton-shimmer"></div>
          </div>
        </div>

        <!-- Error -->
        <div v-else-if="error" class="rounded-2xl border border-red-500/20 bg-red-500/10 p-4 text-red-200">
          {{ error }}
        </div>

        <!-- Empty -->
        <div v-else-if="currentList.length === 0" class="empty-state">
          <div class="empty-orb"></div>
          <p class="text-neutral-200">No {{ activeTab }} to show (yet).</p>
          <p class="text-neutral-500 text-sm mt-1">Come back later</p>
        </div>

        <!-- Grid -->
        <div v-else class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 md:gap-4">
          <article v-for="p in currentList" :key="p.id" class="neo-card group will-change-transform">
            <RouterLink :to="`/p/${p.id}`" class="absolute inset-0 z-20">
              <img
                :src="p.imageUrl"
                :alt="p.caption || 'post image'"
                class="card-img"
                loading="lazy"
                decoding="async"
              />
            </RouterLink>

            <div class="card-glow"></div>
            <div class="card-grid"></div>

            <div class="card-meta">
              <div class="flex items-center gap-3 text-xs text-neutral-200">
                <div class="flex items-center gap-1">
                  <svg class="h-4 w-4" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 6 4 4 6.5 4 8.28 4 9.91 4.97 10.74 6.36 11.57 4.97 13.2 4 14.98 4 17.48 4 19.48 6 19.48 8.5c0 3.78-3.4 6.86-8.05 11.54L12 21.35z"/>
                  </svg>
                  <span>{{ likeCount(p) }}</span>
                </div>
                <span class="opacity-75">{{ shortDate(p) }}</span>
              </div>
              <p v-if="p.caption" class="mt-1 line-clamp-1 text-sm text-neutral-100/90">
                {{ p.caption }}
              </p>
            </div>
          </article>
        </div>
      </section>
    </div>

    <!-- ===== Avatar Modal (teleported) ===== -->
    <teleport to="body">
      <transition name="fade">
        <div
          v-if="avatarModal.open"
          class="fixed inset-0 z-[9998] bg-black/60 backdrop-blur-[2px]"
          @click="closeAvatarModal"
        />
      </transition>

      <transition name="pop">
        <div
          v-if="avatarModal.open"
          class="fixed z-[9999] inset-0 grid place-items-center pointer-events-none"
        >
          <section
            class="pointer-events-auto w-[min(520px,92%)] rounded-2xl border border-white/12 bg-black/70 p-5 md:p-6 shadow-[0_20px_60px_-20px_rgba(0,0,0,.7)]"
            @click.stop
          >
            <header class="flex items-center justify-between mb-3">
              <h3 class="text-lg font-semibold text-white">Change avatar</h3>
              <button class="x-btn" @click="closeAvatarModal" title="Close">✕</button>
            </header>

            <div class="grid gap-4">
              <div class="text-sm text-neutral-300">
                Paste a direct image URL. We’ll save it to your profile.
              </div>

              <div class="grid gap-2">
                <input
                  v-model.trim="avatarModal.url"
                  class="w-full rounded-xl border border-white/12 bg-white/5 px-3 py-2.5 text-neutral-100 placeholder:text-neutral-500 outline-none focus:ring-2 focus:ring-[#e4ae87]/30 focus:border-[#e4ae87]/60"
                  placeholder="https://example.com/me.jpg"
                  @keydown.enter.prevent="submitAvatar"
                  autofocus
                />
                <p v-if="avatarModal.error" class="text-sm text-red-300">
                  {{ avatarModal.error }}
                </p>
              </div>

              <div class="flex items-center gap-3">
                <img
                  :src="avatarModal.url || user?.avatarUrl || fallbackAvatar"
                  alt="preview"
                  class="size-16 rounded-full object-cover border border-white/10"
                />
                <span class="text-xs text-neutral-400">Preview</span>
              </div>

              <div class="mt-1 flex items-center justify-end gap-2">
                <button class="rounded-xl px-3 h-9 text-sm font-medium border border-white/15 text-neutral-100 bg-white/5 hover:bg-white/10 transition" @click="closeAvatarModal">
                  Cancel
                </button>
                <button
                  class="inline-flex items-center gap-2 rounded-xl px-4 h-9 text-sm font-semibold text-black bg-[#e4ae87] hover:bg-[#d6a17c] disabled:opacity-60 transition"
                  :disabled="avatarModal.saving || !avatarModal.url"
                  @click="submitAvatar"
                >
                  <span v-if="!avatarModal.saving">Save</span>
                  <span v-else class="inline-flex items-center gap-2"><span class="spinner"></span> Saving…</span>
                </button>
              </div>
            </div>
          </section>
        </div>
      </transition>
    </teleport>
  </main>
</template>

<script setup>
import { ref, computed, watch, nextTick, onMounted, onBeforeUnmount } from 'vue'
import { useRoute, useRouter, RouterLink } from 'vue-router'
import BeamsBackground from '@/components/BeamsBackground.vue'

const route = useRoute()
const router = useRouter()

const API_BASE = (
  import.meta.env.VITE_API_BASE ?? (import.meta.env.DEV ? 'http://localhost:5050' : '')
).replace(/\/+$/, '')

const fallbackAvatar = 'https://img.freepik.com/premium-vector/user-icon-round-grey-icon_1076610-44912.jpg?w=360'

/* ----- auth / self ----- */
const sessionUser = ref(null)
const isOwnProfile = computed(() => {
  const me = sessionUser.value?.username
  return !!me && me === String(route.params.username || '').trim()
})

/* ----- state ----- */
const user = ref(null)
const posts = ref([])
const liked = ref([])
const saved = ref([])
const error = ref(null)

const loadingUser = ref(true)
const loadingPosts = ref(true)
const loadingLiked = ref(true)
const loadingSaved = ref(true)

/* ----- tabs ----- */
const tabs = [
  { key: 'posts', label: 'Posts', icon: '<svg viewBox="0 0 24 24" class="h-4 w-4" fill="currentColor"><path d="M3 3h8v8H3zM13 3h8v8h-8zM3 13h8v8H3zM13 13h8v8h-8z"/></svg>' },
  { key: 'liked', label: 'Liked', icon: '<svg viewBox="0 0 24 24" class="h-4 w-4" fill="currentColor"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 6 4 4 6.5 4 8.28 4 9.91 4.97 10.74 6.36 11.57 4.97 13.2 4 14.98 4 17.48 4 19.48 6 19.48 8.5c0 3.78-3.4 6.86-8.05 11.54L12 21.35z"/></svg>' },
  { key: 'saved', label: 'Saved', icon: '<svg viewBox="0 0 24 24" class="h-4 w-4" fill="currentColor"><path d="M6 2h12a2 2 0 0 1 2 2v18l-8-4-8 4V4a2 2 0 0 1 2-2z"/></svg>' },
]
const activeTab = ref(route.query.tab === 'liked' || route.query.tab === 'saved' ? route.query.tab : 'posts')

/* ---- Tab underline tracker ---- */
const underline = ref({ left: 0, width: 0 })
const tabRefs = new Map()
function updateUnderline () {
  const el = tabRefs.get(activeTab.value)
  if (!el) return
  const rect = el.getBoundingClientRect()
  const parent = el.parentElement.getBoundingClientRect()
  underline.value = { left: rect.left - parent.left + 8, width: rect.width - 16 }
}
const underlineStyle = computed(() => ({ transform: `translateX(${underline.value.left}px)`, width: `${underline.value.width}px` }))
function setTab (key) {
  if (activeTab.value === key) return
  activeTab.value = key
  router.replace({ query: { ...route.query, tab: key } })
  nextTick(updateUnderline)
}

/* ---- Loading helpers ---- */
const loadingList = computed(() =>
  activeTab.value === 'posts' ? loadingPosts.value
  : activeTab.value === 'liked' ? loadingLiked.value
  : loadingSaved.value
)
const isLoadingAny = computed(() => loadingUser.value || loadingList.value)

const currentList = computed(() =>
  activeTab.value === 'posts' ? posts.value
  : activeTab.value === 'liked' ? liked.value
  : saved.value
)

/* ---- Derived data ---- */
const displayName = computed(() => {
  const u = user.value
  if (!u) return route.params.username
  const full = [u.firstName, u.lastName].filter(Boolean).join(' ')
  return full || u.username || route.params.username
})
const followersCount = computed(() => user.value?.followers?.length || 0)
const followingCount = computed(() => user.value?.following?.length || 0)
const totalLikesReceived = computed(() =>
  posts.value.reduce((acc, p) => acc + (Array.isArray(p.likes) ? p.likes.length : (typeof p.likes === 'number' ? p.likes : 0)), 0)
)
function likeCount (p) { return Array.isArray(p.likes) ? p.likes.length : (typeof p.likes === 'number' ? p.likes : 0) }
function shortDate (p) {
  const d = new Date(p.createdAt || p.date || '')
  return isNaN(d) ? '' : d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}
function formatJoined (iso) { try { return new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'long' }) } catch { return '' } }

/* ---- Data fetching ---- */
async function fetchJSON (path, opts = {}) {
  const url = API_BASE ? `${API_BASE}${path}` : path   // use relative when no base
  const r = await fetch(url, {
    ...opts,
    headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) }
  })
  const data = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(data?.error?.message || `${r.status} ${r.statusText}`)
  return data
}
async function loadUser (username) {
  error.value = null
  loadingUser.value = true
  try {
    if (!username) throw new Error('No username in route')
    user.value = await fetchJSON(`/api/users/${encodeURIComponent(username)}`)
  } catch (e) {
    error.value = 'User not found.'
    user.value = null
  } finally { loadingUser.value = false }
}
async function loadCollections (username) {
  loadingPosts.value = true; loadingLiked.value = true; loadingSaved.value = true
  try {
    const [p, l, s] = await Promise.all([
      fetchJSON(`/api/users/${encodeURIComponent(username)}/posts`),
      fetchJSON(`/api/users/${encodeURIComponent(username)}/liked`),
      fetchJSON(`/api/users/${encodeURIComponent(username)}/saved`)
    ])
    const sortByDateDesc = (a, b) => new Date(b.createdAt || b.date || 0) - new Date(a.createdAt || a.date || 0)
    posts.value = Array.isArray(p) ? [...p].sort(sortByDateDesc) : []
    liked.value = Array.isArray(l) ? [...l].sort(sortByDateDesc) : []
    saved.value = Array.isArray(s) ? [...s].sort(sortByDateDesc) : []
  } catch (e) {
    error.value = 'Failed to load posts.'
    posts.value = []; liked.value = []; saved.value = []
  } finally {
    loadingPosts.value = false; loadingLiked.value = false; loadingSaved.value = false
  }
}

/* ---- Avatar modal / save ---- */
const avatarModal = ref({ open: false, url: '', saving: false, error: null })

function openAvatarModal () {
  avatarModal.value = {
    open: true,
    url: user.value?.avatarUrl || '',
    saving: false,
    error: null
  }
}
function closeAvatarModal () {
  avatarModal.value.open = false
  avatarModal.value.error = null
}

function validUrl (u) {
  try { new URL(u); return true } catch { return false }
}

async function submitAvatar () {
  avatarModal.value.error = null
  const url = avatarModal.value.url.trim()
  if (!validUrl(url)) {
    avatarModal.value.error = 'Please enter a valid image URL.'
    return
  }
  if (!sessionUser.value?.username) {
    avatarModal.value.error = 'You must be logged in.'
    return
  }

  avatarModal.value.saving = true
  try {
    const updated = await fetchJSON(`/api/users/${encodeURIComponent(sessionUser.value.username)}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
        'x-user': sessionUser.value.username
      },
      body: JSON.stringify({ avatarUrl: url })
    })
    user.value = updated
    // sync mini session
    const stored = JSON.parse(localStorage.getItem('fg.user') || '{}')
    localStorage.setItem('fg.user', JSON.stringify({ ...stored, avatarUrl: updated.avatarUrl }))
    closeAvatarModal()
  } catch (e) {
    avatarModal.value.error = e.message || 'Failed to update avatar'
  } finally {
    avatarModal.value.saving = false
  }
}

/* ---- Mount / watchers ---- */
function handleResize () { updateUnderline() }
onMounted(() => {
  // hydrate mini session
  try { sessionUser.value = JSON.parse(localStorage.getItem('fg.user') || '{}') } catch { sessionUser.value = null }

  const username = String(route.params.username || '').trim()
  loadUser(username)
  loadCollections(username)
  nextTick(updateUnderline)
  window.addEventListener('resize', handleResize, { passive: true })
})
onBeforeUnmount(() => window.removeEventListener('resize', handleResize))
watch(() => route.params.username, (nu, old) => {
  if (nu && nu !== old) {
    const username = String(nu).trim()
    activeTab.value = 'posts'
    router.replace({ query: { tab: 'posts' } })
    loadUser(username)
    loadCollections(username)
    nextTick(updateUnderline)
  }
})
</script>

<style scoped>
/* ---------- Header Accents ---------- */
.neon-border{
  background: linear-gradient(120deg, #661313, #661414 30%, #884554 55%, #e4ae87 80%, #daa8ae);
  filter: blur(6px);
  opacity: .22;
  animation: hue 18s linear infinite;
}
.neon-ring{
  background: conic-gradient(from 0deg, #661313, #661414, #884554, #e4ae87, #daa8ae, #661313);
  filter: blur(6px);
  opacity: .45;
  animation: spin 10s linear infinite;
}
.scanline{
  background: repeating-linear-gradient(to bottom, rgba(255,255,255,0.45) 0, rgba(255,255,255,0.45) 1px, transparent 1px, transparent 3px);
}
@media (prefers-reduced-motion: reduce){
  .neon-border,.neon-ring{ animation: none }
  .scanline{ display:none }
}

/* ---------- Stat chips ---------- */
.stat-chip{
  position: relative; padding: .85rem 1rem; border-radius: 16px;
  border: 1px solid rgba(255,255,255,.12);
  background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
  box-shadow: 0 8px 22px -14px rgba(0,0,0,.6), inset 0 0 0 1px rgba(255,255,255,.04);
}
.stat-chip::before{
  content:''; position:absolute; inset:-2px;
  background: radial-gradient(40% 50% at 10% 0%, rgba(228,174,135,.22), transparent 60%);
  opacity:.35;
}
.stat-val{ font-size: clamp(1.1rem, 1.8vw, 1.35rem); font-weight: 700; color:#fff; text-shadow: 0 0 10px rgba(228,174,135,.22) }
.stat-label{ margin-top:2px; font-size:11px; letter-spacing:.6px; text-transform:uppercase; color: rgba(234,234,234,.6) }

/* ---------- Tabs ---------- */
.tab-glow{
  background:
    radial-gradient(60% 60% at 10% 10%, rgba(102,20,20,.14), transparent 60%),
    radial-gradient(40% 60% at 90% 0%, rgba(136,69,84,.14), transparent 60%);
}
.tab-chip{
  position: relative; display:inline-flex; align-items:center; gap:.5rem;
  padding:.6rem .9rem; border-radius: 14px;
  border: 1px solid rgba(255,255,255,.12);
  background: linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.02));
  color:#eaeaea; transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease;
}
.tab-chip:hover{ transform: translateY(-1px); box-shadow: 0 10px 20px -16px rgba(228,174,135,.35); border-color: rgba(228,174,135,.3) }
.tab-chip--active{ background: linear-gradient(180deg, rgba(228,174,135,.16), rgba(255,255,255,.05)); border-color: rgba(228,174,135,.5); color:#fff }
.tab-wash{ content:''; position:absolute; inset:0; border-radius:inherit; background: radial-gradient(80% 80% at 50% 0%, rgba(228,174,135,.22), transparent 60%); opacity:0; transition:opacity .2s ease }
.tab-chip:hover .tab-wash{ opacity:1 }
.tab-count{ font-size:11px; padding:2px 6px; border-radius:999px; background: rgba(0,0,0,.45); border:1px solid rgba(255,255,255,.16); color:#e4ae87 }
.tab-underline{
  position:absolute; bottom:-6px; height:3px; border-radius:999px;
  background: linear-gradient(90deg, #661313, #661414 30%, #884554 60%, #e4ae87 100%);
  box-shadow: 0 0 12px #e4ae87AA, 0 0 3px #daa8aeAA inset;
  transition: transform .25s cubic-bezier(.2,.8,.25,1), width .25s cubic-bezier(.2,.8,.25,1);
}
.snap-enter-active,.snap-leave-active{ transition: all .25s ease }
.snap-enter-from{ opacity:0 } .snap-enter-to{ opacity:1 }
.snap-leave-from{ opacity:1 } .snap-leave-to{ opacity:0 }

/* ---------- Content / Cards ---------- */
.content-glow{
  background:
    radial-gradient(60% 60% at 0% 20%, rgba(102,20,20,.1), transparent 60%),
    radial-gradient(50% 60% at 100% 80%, rgba(228,174,135,.1), transparent 60%),
    linear-gradient(180deg, rgba(255,255,255,.02), transparent);
}
.skeleton-card{ position:relative; aspect-ratio:1; border-radius:18px; overflow:hidden; border:1px solid rgba(255,255,255,.12); background: rgba(255,255,255,.04) }
.skeleton-shimmer{ position:absolute; inset:0; background: linear-gradient(110deg, transparent 0%, rgba(255,255,255,.06) 45%, transparent 90%); animation: shimmer 1.8s infinite }
.empty-state{ position:relative; border-radius:18px; border:1px solid rgba(255,255,255,.12); background: linear-gradient(180deg, rgba(255,255,255,.035), rgba(255,255,255,.02)); padding:3rem 1.5rem; text-align:center; overflow:hidden }
.empty-orb{ position:absolute; top:-40px; left:50%; transform: translateX(-50%); width:240px; height:240px; border-radius:50%; background: radial-gradient(circle at 50% 50%, #e4ae87, rgba(136,69,84,.5) 60%, transparent 70%); opacity:.14 }
.neo-card{
  position:relative; aspect-ratio:1; overflow:hidden; border-radius:18px;
  border:1px solid rgba(255,255,255,.12); background: rgba(0,0,0,.55);
  transition: transform .2s ease, box-shadow .25s ease, border-color .2s ease;
}
.neo-card:hover{ transform: translateY(-2px); box-shadow: 0 16px 34px -18px rgba(228,174,135,.3); border-color: rgba(228,174,135,.4) }
.card-img{ position:absolute; inset:0; width:100%; height:100%; object-fit:cover; transition: transform .45s ease, opacity .45s ease }
.neo-card:hover .card-img{ transform: scale(1.035); opacity:.97 }
.card-glow{
  position:absolute; inset:-18%;
  background:
    radial-gradient(40% 40% at 20% 15%, rgba(228,174,135,.14), transparent 60%),
    radial-gradient(30% 30% at 80% 85%, rgba(136,69,84,.14), transparent 60%);
  opacity:.5; pointer-events:none;
}
.card-grid{
  position:absolute; inset:0; pointer-events:none; opacity:.12;
  background:
    linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px) 0 0/ 14px 14px,
    linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px) 0 0/ 14px 14px;
  mask-image: radial-gradient(120% 80% at 50% 120%, black, transparent);
}
.card-meta{
  position:absolute; left:0; right:0; bottom:0; padding:.75rem .85rem;
  background: linear-gradient(180deg, transparent, rgba(0,0,0,.6));
  border-top: 1px solid rgba(255,255,255,.08);
}

/* ---------- Modal bits ---------- */
.x-btn{
  width:2rem; height:2rem; border-radius:.6rem;
  border:1px solid rgba(255,255,255,.12);
  background:rgba(255,255,255,.06);
}
.spinner{
  width:12px;height:12px;border-radius:9999px;
  border:2px solid rgba(255,255,255,.35); border-top-color:#e4ae87;
  animation: spin 1s linear infinite;
}

/* ---------- Keyframes ---------- */
@keyframes hue { from{ filter:hue-rotate(0deg) blur(6px) } to{ filter:hue-rotate(360deg) blur(6px) } }
@keyframes spin { from{ transform: rotate(0deg) } to{ transform: rotate(360deg) } }
@keyframes glide { 0%{ transform: translateX(-20%) } 100%{ transform: translateX(120%) } }

/* Transitions for modal */
.fade-enter-active,.fade-leave-active{ transition: opacity .18s ease }
.fade-enter-from,.fade-leave-to{ opacity: 0 }
.pop-enter-active,.pop-leave-active{ transition: transform .18s ease, opacity .18s ease }
.pop-enter-from,.pop-leave-to{ transform: scale(.98); opacity: 0 }
</style>
