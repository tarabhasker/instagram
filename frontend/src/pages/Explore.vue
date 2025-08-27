<!-- src/pages/Explore.vue -->
<template>
  <main class="relative mx-auto max-w-5xl px-4 pt-6 md:pt-8 pb-[calc(88px+env(safe-area-inset-bottom))] md:pb-10 text-neutral-200">
    <!-- Soft themed backdrop -->
    <div class="pointer-events-none absolute -inset-10 -z-10 rounded-[36px] feed-bg"></div>

    <!-- Header / search -->
    <section
      class="relative overflow-hidden rounded-3xl border border-white/10 bg-black/40 shadow-[0_8px_28px_-12px_rgba(0,0,0,.7)]"
    >
      <div class="pointer-events-none absolute -inset-[2px] rounded-[28px] neon-border"></div>
      <div class="relative z-10 p-4 md:p-5">
        <div class="flex items-center gap-3">
          <div class="relative flex-1">
            <input
              v-model="q"
              @focus="searchFocused = true"
              @blur="onBlur"
              type="search"
              class="w-full rounded-xl border border-white/10 bg-white/5 pl-10 pr-3 py-2.5 text-sm text-neutral-100 placeholder:text-neutral-500 outline-none transition focus:ring-2 focus:ring-[#e4ae87]/30 focus:border-[#e4ae87]/60"
              placeholder="Search accounts…"
            />
            <svg class="absolute left-3 top-1/2 -translate-y-1/2 h-4.5 w-4.5 text-neutral-400" viewBox="0 0 24 24" fill="currentColor">
              <path d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0016 9.5 6.5 6.5 0 109.5 16a6.471 6.471 0 004.23-1.57l.27.28v.79L20 21.49 21.49 20 15.5 14zM10 14a4 4 0 110-8 4 4 0 010 8z"/>
            </svg>
          </div>

          <button
            v-if="searchFocused || q"
            class="icon-btn"
            @click="clearSearch"
            title="Clear"
          >
            ✕
          </button>
        </div>

        <!-- SIMILAR BANNER -->
        <transition name="fade">
          <div
            v-if="hasSimilarActive && !searchFocused"
            class="mt-4 flex flex-wrap items-center gap-2 rounded-xl border border-[#e4ae87]/30 bg-white/[0.04] px-3 py-2"
          >
            <span class="text-sm text-neutral-300">Finding similar to</span>
            <RouterLink
              :to="`/p/${similarFrom?.id}`"
              class="text-sm font-medium text-neutral-100 underline underline-offset-2"
            >
              {{ similarFrom?.caption || 'this post' }}
            </RouterLink>
            <span class="text-sm text-neutral-500">· tags:</span>
            <span
              v-for="t in similarTags"
              :key="t"
              class="chip"
              >#{{ t }}</span
            >
            <button
              class="ml-auto inline-flex items-center gap-2 rounded-lg border border-white/15 text-neutral-100 bg-white/5 hover:bg-white/10 px-2.5 h-8 text-xs"
              @click="clearSimilar"
              title="Clear similar filter"
            >
              ✕ Clear
            </button>
          </div>
        </transition>
      </div>
    </section>

    <!-- Accounts list (when searching) -->
    <section v-if="searchFocused || q" class="mt-6">
      <!-- Loading -->
      <div v-if="loadingUsers" class="grid gap-2">
        <div v-for="i in 6" :key="'ul'+i" class="rounded-xl border border-white/10 bg-white/[0.04] p-3 skeleton">
          <div class="h-6 w-40 rounded bg-white/10"></div>
        </div>
      </div>

      <!-- Results -->
      <ul v-else class="grid gap-2">
        <li
          v-for="u in filteredUsers"
          :key="u.username"
          class="flex items-center gap-3 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2 hover:bg-white/[0.06] transition"
        >
          <img :src="u.avatarUrl || fallbackAvatar" :alt="u.username" class="size-9 rounded-full object-cover border border-white/10" />
          <div class="min-w-0">
            <RouterLink :to="`/@${u.username}`" class="text-sm font-medium text-neutral-100 hover:underline">{{ displayName(u) }}</RouterLink>
            <div class="text-xs text-neutral-400 truncate">@{{ u.username }}</div>
          </div>
          <RouterLink :to="`/@${u.username}`" class="ml-auto text-xs rounded-md border border-white/10 px-2.5 h-8 inline-flex items-center hover:bg-white/[0.06]">
            View
          </RouterLink>
        </li>
      </ul>

      <!-- Empty -->
      <div v-if="!filteredUsers.length && !loadingUsers" class="empty-card mt-6">
        <div class="empty-orb"></div>
        <p class="text-neutral-200">No accounts match “{{ q }}”.</p>
      </div>
    </section>

    <!-- Grid (when NOT searching) -->
    <section v-else class="mt-6">
      <!-- Loading -->
      <div v-if="loading" class="grid grid-cols-2 md:grid-cols-3 gap-2 md:gap-3">
        <div v-for="i in 9" :key="'g'+i" class="aspect-square rounded-xl border border-white/10 bg-white/[0.04] skeleton"></div>
      </div>

      <!-- Empty (e.g., no similar tag matches) -->
      <div v-else-if="!grid.length" class="empty-card">
        <div class="empty-orb"></div>
        <p class="text-neutral-200">
          {{ hasSimilarActive ? 'No similar posts found from non-followed accounts.' : 'Nothing here yet.' }}
        </p>
        <p class="text-neutral-500 text-sm mt-1" v-if="hasSimilarActive">Try clearing the filter or exploring later.</p>
      </div>

      <!-- Posts grid -->
      <div v-else class="grid grid-cols-2 md:grid-cols-3 gap-2 md:gap-3">
        <RouterLink
          v-for="p in grid"
          :key="p.id"
          :to="`/p/${p.id}`"
          class="group relative block overflow-hidden rounded-xl border border-white/10 bg-black/40"
          :title="p.caption || 'post'"
        >
          <img
            :src="p.imageUrl"
            :alt="p.caption || 'post image'"
            class="w-full h-full object-cover aspect-square select-none"
            loading="lazy"
            decoding="async"
          />
          <div class="pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-black/60 via-black/20 to-transparent opacity-0 group-hover:opacity-100 transition"></div>
          <div class="absolute left-2 bottom-2 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition">
            <img
              :src="(profileCache.get(p.user?.username)?.avatarUrl) || fallbackAvatar"
              :alt="p.user?.username"
              class="size-6 rounded-full object-cover border border-white/10"
            />
            <span class="text-xs text-neutral-200">@{{ p.user?.username }}</span>
          </div>
        </RouterLink>
      </div>
    </section>
  </main>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5050'
const fallbackAvatar = 'https://i.pravatar.cc/300?img=47'

/* route */
const route = useRoute()
const router = useRouter()

/* session (logged-in user) */
const sessionUser = ref(null)
const currentUsername = computed(() => sessionUser.value?.username || 'guest')

/* state */
const loading = ref(true)
const loadingUsers = ref(true)
const posts = ref([])
const me = ref(null)
const profileCache = ref(new Map())
const allUsers = ref([])

const q = ref('')
const searchFocused = ref(false)

/* similar filter state */
const similarTags = ref([]) // lowercase strings
const similarFrom = ref(null) // { id, caption }
const hasSimilarActive = computed(() => similarTags.value.length > 0)

/* helpers */
const fetchJSON = async (path, options) => {
  const r = await fetch(`${API_BASE}${path}`, options)
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
  return r.json()
}
const fetchUser = async (username) => {
  if (!username) return null
  if (profileCache.value.has(username)) return profileCache.value.get(username)
  try {
    const u = await fetchJSON(`/api/users/${encodeURIComponent(username)}`)
    profileCache.value.set(username, u)
    return u
  } catch { return null }
}

/* loaders */
const loadCore = async () => {
  loading.value = true
  try {
    // hydrate session
    try { sessionUser.value = JSON.parse(localStorage.getItem('fg.user') || '{}') } catch { sessionUser.value = null }
    if (currentUsername.value !== 'guest') {
      me.value = await fetchUser(currentUsername.value)
    } else {
      me.value = null
    }

    const all = await fetchJSON('/api/posts')
    posts.value = all.map(p => ({
      ...p,
      tags: Array.isArray(p.tags) ? p.tags : [],
      user: p.user || { username: p.username }
    }))
    // warm author profiles present in grid
    const usernames = [...new Set(posts.value.map(p => p.user?.username).filter(Boolean))]
    await Promise.all(usernames.map(fetchUser))
  } finally {
    loading.value = false
  }
}

const loadUsers = async () => {
  loadingUsers.value = true
  try {
    // Try to fetch all users; if endpoint isn't present, derive from posts + social graph
    let list = []
    try {
      list = await fetchJSON('/api/users')
    } catch {
      const names = new Set([
        ...posts.value.map(p => p.user?.username).filter(Boolean),
        ...(me.value?.following || []),
        ...(me.value?.followers || []),
        currentUsername.value,
      ])
      const arr = await Promise.all([...names].map(fetchUser))
      list = arr.filter(Boolean)
    }
    // De-dup by username
    const seen = new Set()
    allUsers.value = list.filter(u => {
      if (!u?.username || seen.has(u.username)) return false
      seen.add(u.username); return true
    })
    // cache them
    allUsers.value.forEach(u => profileCache.value.set(u.username, u))
  } finally {
    loadingUsers.value = false
  }
}

const loadSimilarIfAny = async () => {
  const pid = route.query?.similar
  if (!pid) {
    similarTags.value = []
    similarFrom.value = null
    return
  }
  // try to find in current posts; otherwise fetch directly
  let item = posts.value.find(p => p.id === pid)
  if (!item) {
    try { item = await fetchJSON(`/api/posts/${encodeURIComponent(pid)}`) } catch { item = null }
  }
  if (item) {
    const tags = Array.isArray(item.tags) ? item.tags : []
    similarTags.value = tags.map(t => String(t).toLowerCase()).filter(Boolean)
    similarFrom.value = { id: item.id, caption: item.caption || '' }
  } else {
    similarTags.value = []
    similarFrom.value = null
  }
}

/* search behavior */
const clearSearch = () => { q.value = ''; searchFocused.value = false }
const onBlur = (e) => {
  setTimeout(() => {
    if (!q.value) searchFocused.value = false
  }, 120)
}

/* following set (hide followed + me from Explore grid) */
const followingSet = computed(() => new Set([currentUsername.value, ...(me.value?.following || [])]))

/* base grid: non-followed only */
const baseGrid = computed(() =>
  posts.value.filter(p => !followingSet.value.has(p.user?.username))
)

/* final grid: maybe tag-filtered by ?similar */
const grid = computed(() => {
  let list = baseGrid.value
  if (hasSimilarActive.value) {
    list = list.filter(p => {
      const ptags = (p.tags || []).map(t => String(t).toLowerCase())
      return ptags.some(t => similarTags.value.includes(t))
    })
  }
  // recency first
  return list.sort((a,b) => new Date(b.createdAt || b.date) - new Date(a.createdAt || a.date))
})

/* accounts filtering */
const filteredUsers = computed(() => {
  const ql = (q.value || '').trim().toLowerCase()
  if (!ql) return allUsers.value
  return allUsers.value.filter(u => {
    const full = [u.firstName, u.lastName].filter(Boolean).join(' ').toLowerCase()
    return u.username.toLowerCase().includes(ql) || full.includes(ql)
  })
})

/* utils */
const displayName = (u) => {
  if (!u) return ''
  const full = [u.firstName, u.lastName].filter(Boolean).join(' ')
  return full || u.username
}
const clearSimilar = () => { router.push({ path: '/explore' }) }

onMounted(async () => {
  await loadCore()
  await loadUsers()
  await loadSimilarIfAny()
})
watch(() => route.query.similar, loadSimilarIfAny)
</script>

<style scoped>
/* Background (palette) */
.feed-bg{
  background:
    linear-gradient(120deg,#66131322,#66141422 30%,#88455422 60%,#e4ae8722 85%,#daa8ae22),
    radial-gradient(50% 60% at 0% 20%, rgba(102,20,20,.18), transparent 60%),
    radial-gradient(50% 60% at 100% 80%, rgba(228,174,135,.14), transparent 60%);
  filter: blur(20px);
}

/* Neon rim */
.neon-border{
  background: linear-gradient(120deg,#661313,#661414 30%,#884554 55%,#e4ae87 80%,#daa8ae);
  filter: blur(7px);
  opacity:.22;
  animation: hue 18s linear infinite;
}

/* Buttons */
.icon-btn{
  display:grid; place-items:center;
  width:2.25rem; height:2.25rem; border-radius:.75rem;
  color:rgba(255,255,255,.85);
  border:1px solid rgba(255,255,255,.1);
  background: transparent;
  transition: background .15s ease, color .15s ease, border-color .15s ease;
}
.icon-btn:hover{ background: rgba(255,255,255,.06); border-color: rgba(255,255,255,.16); color:#fff; }

/* Chips */
.chip{
  display:inline-flex; align-items:center; gap:.25rem;
  padding:.25rem .5rem; border-radius:999px;
  border:1px solid rgba(255,255,255,.14);
  background: rgba(255,255,255,.05);
  font-size:.75rem; color:#fff;
}

/* Empty card */
.empty-card{
  position: relative;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,.12);
  background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
  padding: 3rem 1.5rem; text-align: center; overflow: hidden;
}
.empty-orb{
  position:absolute; top:-40px; left:50%; transform: translateX(-50%);
  width: 240px; height: 240px; border-radius: 50%;
  background: radial-gradient(circle at 50% 50%, #e4ae87, rgba(136,69,84,.6) 60%, transparent 70%);
  filter: blur(24px); opacity: .16;
}

/* Skeleton shimmer */
.skeleton{ position: relative; overflow: hidden; }
.skeleton::after{
  content:""; position:absolute; inset:0;
  background: linear-gradient(110deg, transparent 0%, rgba(255,255,255,.08) 45%, transparent 90%);
  animation: shimmer 1.8s infinite;
}

/* Animations */
@keyframes hue { from{ filter:hue-rotate(0deg) blur(7px)} to{ filter:hue-rotate(360deg) blur(7px)} }
@keyframes shimmer { 0% { transform: translateX(-100%) } 60%, 100% { transform: translateX(100%) } }

.fade-enter-active,.fade-leave-active{ transition: opacity .2s }
.fade-enter-from,.fade-leave-to{ opacity: 0 }
</style>
