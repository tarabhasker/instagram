<!-- src/pages/Activity.vue -->
<template>
  <main
    class="relative mx-auto max-w-3xl px-4 pt-6 md:pt-8 pb-[calc(88px+env(safe-area-inset-bottom))] md:pb-10 text-neutral-200"
  >
    <!-- Soft themed backdrop -->
    <div class="pointer-events-none absolute -inset-10 -z-10 rounded-[36px] feed-bg"></div>

    <!-- Header -->
    <section
      class="relative overflow-hidden rounded-3xl border border-white/10 bg-black/40 shadow-[0_8px_28px_-12px_rgba(0,0,0,.7)]"
    >
      <div class="pointer-events-none absolute -inset-[2px] rounded-[28px] neon-border"></div>
      <div class="relative z-10 p-5 md:p-7">
        <div class="flex items-center justify-between">
          <h2 class="text-lg md:text-xl font-semibold tracking-tight flex items-center gap-2">
            <span class="inline-block size-2 rounded-full bg-[#e4ae87] shadow-[0_0_12px_#e4ae87aa]"></span>
            Activity
          </h2>
          <div class="text-xs text-neutral-400">{{ filtered.length }} new</div>
        </div>

        <!-- Filters -->
        <div class="mt-4 flex flex-wrap gap-2">
          <button
            v-for="t in filters"
            :key="t.key"
            class="tab-chip"
            :class="activeFilter === t.key ? 'tab-chip--active' : ''"
            @click="activeFilter = t.key"
          >
            <span class="text-xs">{{ t.label }}</span>
            <span class="tab-count">{{ countByType(t.key) }}</span>
          </button>
        </div>
      </div>
    </section>

    <!-- Loading -->
    <div v-if="loading" class="grid gap-3 mt-6">
      <div v-for="i in 6" :key="i" class="rounded-2xl border border-white/10 bg-white/[0.04] p-4 skeleton">
        <div class="h-6 w-40 rounded bg-white/10"></div>
        <div class="mt-3 h-4 w-20 rounded bg-white/10"></div>
      </div>
    </div>

    <!-- Empty -->
    <div v-else-if="filtered.length === 0" class="empty-card mt-6">
      <div class="empty-orb"></div>
      <p class="text-neutral-200">No notifications yet.</p>
      <p class="text-neutral-500 text-sm mt-1">Likes, comments, shares, saves, and new followers will appear here.</p>
    </div>

    <!-- List -->
    <section v-else class="mt-6 grid gap-3">
      <article
        v-for="n in filtered"
        :key="n.id"
        class="relative overflow-hidden rounded-2xl border border-white/10 bg-black/40 p-4 hover:bg-white/[0.04] transition"
      >
        <div class="flex items-start gap-3">
          <img
            :src="(profileCache.get(n.actor)?.avatarUrl) || fallbackAvatar"
            :alt="n.actor"
            class="size-10 rounded-full object-cover border border-white/10"
          />
          <div class="min-w-0 flex-1">
            <p class="text-sm text-neutral-100">
              <RouterLink :to="`/@${n.actor}`" class="font-medium hover:underline">
                {{ displayName(profileCache.get(n.actor)) || '@' + n.actor }}
              </RouterLink>

              <template v-if="n.type==='like'">
                liked your <RouterLink v-if="n.postId" :to="`/p/${n.postId}`" class="underline underline-offset-2">post</RouterLink>.
              </template>
              <template v-else-if="n.type==='comment'">
                commented on your <RouterLink v-if="n.postId" :to="`/p/${n.postId}`" class="underline underline-offset-2">post</RouterLink>:
                <span class="text-neutral-300">“{{ n.meta?.content || '…' }}”</span>
              </template>
              <template v-else-if="n.type==='share'">
                shared your <RouterLink v-if="n.postId" :to="`/p/${n.postId}`" class="underline underline-offset-2">post</RouterLink>
                <span v-if="n.meta?.to?.length"> with {{ n.meta.to.length }} friend{{ n.meta.to.length>1 ? 's':'' }}</span>.
              </template>
              <template v-else-if="n.type==='save'">
                saved your <RouterLink v-if="n.postId" :to="`/p/${n.postId}`" class="underline underline-offset-2">post</RouterLink>.
              </template>
              <template v-else-if="n.type==='follow'">
                started following you.
              </template>
            </p>

            <div class="mt-1 text-xs text-neutral-400">{{ timeAgo(n.date) }}</div>
          </div>

          <RouterLink
            v-if="n.postId && postThumb(n.postId)"
            :to="`/p/${n.postId}`"
            class="shrink-0"
            :title="postById(n.postId)?.caption || 'post'"
          >
            <img
              :src="postThumb(n.postId)"
              class="size-12 rounded-lg object-cover border border-white/10"
              :alt="postById(n.postId)?.caption || 'post image'"
              loading="lazy"
              decoding="async"
            />
          </RouterLink>
        </div>
      </article>
    </section>
  </main>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter, useRoute, RouterLink } from 'vue-router'
import { useAuth } from '@/composables/useAuth'

const router = useRouter()
const route = useRoute()
const auth = useAuth()

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5050'
const fallbackAvatar = 'https://i.pravatar.cc/300?img=47'

/* who am I? */
const ME = computed(() => auth.user.value?.username || '')
const isLoggedIn = computed(() => !!ME.value)
if (!isLoggedIn.value) {
  router.replace({ path: '/auth', query: { next: route.fullPath } })
}

/* state */
const loading = ref(true)
const events = ref([])          // normalized notifications
const posts = ref([])
const me = ref(null)
const profileCache = ref(new Map())

/* filters */
const filters = [
  { key: 'all',     label: 'All' },
  { key: 'like',    label: 'Likes' },
  { key: 'comment', label: 'Comments' },
  { key: 'share',   label: 'Shares' },
  { key: 'save',    label: 'Saves' },
  { key: 'follow',  label: 'Follows' },
]
const activeFilter = ref('all')

/* fetch helpers */
const authHeaders = computed(() => {
  const t = localStorage.getItem('fg.token') || ''
  return t ? { Authorization: `Bearer ${t}` } : {}
})

const fetchJSON = async (path, options = {}) => {
  const r = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}), ...authHeaders.value }
  })
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

/* loader */
const load = async () => {
  if (!ME.value) return
  loading.value = true
  try {
    // me + my posts (for thumbs)
    me.value = await fetchUser(ME.value)
    const allPosts = await fetchJSON('/api/posts')
    posts.value = allPosts.map(p => ({ ...p, user: p.user || { username: p.username } }))

    // Try server-provided activity first
    let raw = null
    try {
      raw = await fetchJSON(`/api/activity/${encodeURIComponent(ME.value)}`)
    } catch {
      try {
        raw = await fetchJSON(`/api/activity?username=${encodeURIComponent(ME.value)}`)
      } catch {
        raw = null
      }
    }

    if (Array.isArray(raw) && raw.length) {
      events.value = normalizeEvents(raw)
    } else {
      // Fallback: derive from DB snapshot
      events.value = deriveFromSnapshot()
    }

    // Warm actor profiles
    const actors = [...new Set(events.value.map(e => e.actor).filter(Boolean))]
    await Promise.all(actors.map(fetchUser))
  } finally {
    loading.value = false
  }
}

/* normalization */
const normalizeEvents = (arr) => {
  // Expecting items like: { id,type,actor,targetUser,postId,date,meta }
  return arr
    .filter(e => e && e.targetUser === ME.value)
    .map(e => ({
      id: e.id || `${e.type}_${e.actor}_${e.postId || ''}_${e.date || ''}_${Math.random().toString(36).slice(2,7)}`,
      type: e.type,               // 'like' | 'comment' | 'share' | 'save' | 'follow'
      actor: e.actor,
      postId: e.postId || null,
      date: e.date || new Date().toISOString(),
      meta: e.meta || null
    }))
    .sort((a,b)=>new Date(b.date)-new Date(a.date))
}

/* fallback builder (when API /activity is not available) */
const deriveFromSnapshot = () => {
  const mine = posts.value.filter(p => (p.user?.username || p.username) === ME.value)
  const out = []

  // comments
  for (const p of mine) {
    const pid = p.id
    const comments = Array.isArray(p.comments) ? p.comments : []
    for (const c of comments) {
      if (!c?.username || c.username === ME.value) continue
      out.push({
        id: `c_${pid}_${c.id || Math.random().toString(36).slice(2,9)}`,
        type: 'comment',
        actor: c.username,
        postId: pid,
        date: c.date || new Date().toISOString(),
        meta: { content: c.content }
      })
    }
  }

  // likes
  for (const p of mine) {
    const pid = p.id
    const likes = Array.isArray(p.likes) ? p.likes : []
    for (const who of likes) {
      if (!who || typeof who !== 'string' || who === ME.value) continue
      out.push({
        id: `l_${pid}_${who}`,
        type: 'like',
        actor: who,
        postId: pid,
        date: p.updatedAt || p.createdAt || new Date(Date.now() - 1000 * 60 * 60).toISOString(),
        meta: null
      })
    }
  }

  // saves
  for (const p of mine) {
    const pid = p.id
    const saves = Array.isArray(p.saves) ? p.saves : []
    for (const who of saves) {
      if (!who || typeof who !== 'string' || who === ME.value) continue
      out.push({
        id: `s_${pid}_${who}`,
        type: 'save',
        actor: who,
        postId: pid,
        date: p.updatedAt || p.createdAt || new Date(Date.now() - 1000 * 60 * 30).toISOString(),
        meta: null
      })
    }
  }

  // shares (if your server wrote p.shares[])
  for (const p of mine) {
    const pid = p.id
    const shares = Array.isArray(p.shares) ? p.shares : [] // [{username,to[],date}]
    for (const sh of shares) {
      if (!sh?.username || sh.username === ME.value) continue
      out.push({
        id: `sh_${pid}_${sh.username}_${sh.date || Math.random().toString(36).slice(2,7)}`,
        type: 'share',
        actor: sh.username,
        postId: pid,
        date: sh.date || new Date().toISOString(),
        meta: { to: sh.to || [] }
      })
    }
  }

  // follows (requires /api/users to prefill profileCache['__all__'])
  out.push(...derivedFollows())

  return out.sort((a,b)=>new Date(b.date)-new Date(a.date))
}

const derivedFollows = () => {
  const arr = []
  const allUsers = profileCache.value.get('__all__') || []
  for (const u of allUsers) {
    if (u?.username === ME.value) continue
    if (Array.isArray(u.following) && u.following.includes(ME.value)) {
      arr.push({
        id: `f_${u.username}`,
        type: 'follow',
        actor: u.username,
        postId: null,
        date: u.updatedAt || new Date(Date.now() - 1000 * 60 * 20).toISOString(),
        meta: null
      })
    }
  }
  return arr
}

/* helpers for UI */
const countByType = (key) => {
  if (key === 'all') return events.value.length
  return events.value.filter(e => e.type === key).length
}
const filtered = computed(() => {
  if (activeFilter.value === 'all') return events.value
  return events.value.filter(e => e.type === activeFilter.value)
})

const postById = (id) => posts.value.find(p => p.id === id)
const postThumb = (id) => postById(id)?.imageUrl || null
const displayName = (u) => {
  if (!u) return ''
  const full = [u.firstName, u.lastName].filter(Boolean).join(' ')
  return full || u.username
}
const timeAgo = (iso) => {
  if (!iso) return ''
  const d = new Date(iso)
  const now = new Date()
  const sec = Math.max(1, Math.floor((now - d) / 1000))
  if (sec < 60) return `${sec}s`
  const min = Math.floor(sec / 60)
  if (min < 60) return `${min}m`
  const hr = Math.floor(min / 60)
  if (hr < 24) return `${hr}h`
  const day = Math.floor(hr / 24)
  if (day < 7) return `${day}d`
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

/* prefetch all users to power follow derivation fallback */
const preloadAllUsers = async () => {
  try {
    const list = await fetchJSON('/api/users')
    if (Array.isArray(list)) {
      profileCache.value.set('__all__', list)
      list.forEach(u => profileCache.value.set(u.username, u))
    }
  } catch {
    // ignore; endpoint might not exist
  }
}

onMounted(async () => {
  if (!isLoggedIn.value) return
  await preloadAllUsers()
  await load()
})
</script>

<style scoped>
/* Background (palette, organic) */
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

/* Tabs (reuse style language) */
.tab-chip{
  position: relative; display:inline-flex; align-items:center; gap:.4rem;
  padding:.45rem .65rem; border-radius: 12px;
  border: 1px solid rgba(255,255,255,.12);
  background: linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.02));
  color:#eaeaea; transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease;
  font-size:.85rem;
}
.tab-chip--active{ background: linear-gradient(180deg, rgba(228,174,135,.16), rgba(255,255,255,.05)); border-color: rgba(228,174,135,.5); color:#fff }
.tab-count{ font-size:10px; padding:1px 6px; border-radius:999px; background: rgba(0,0,0,.45); border:1px solid rgba(255,255,255,.16); color:#e4ae87 }

/* Empty card */
.empty-card{
  position: relative; border-radius: 18px;
  border: 1px solid rgba(255,255,255,.12);
  background: linear-gradient(180deg, rgba(255,255,255,.035), rgba(255,255,255,.02));
  padding: 3rem 1.5rem; text-align: center; overflow: hidden;
}
.empty-orb{
  position:absolute; top:-40px; left:50%; transform: translateX(-50%);
  width: 240px; height: 240px; border-radius: 50%;
  background: radial-gradient(circle at 50% 50%, #e4ae87, rgba(136,69,84,.5) 60%, transparent 70%);
  opacity:.14;
}

/* Skeleton shimmer */
.skeleton{
  position: relative; overflow: hidden;
}
.skeleton::after{
  content:""; position:absolute; inset:0;
  background: linear-gradient(110deg, transparent 0%, rgba(255,255,255,.08) 45%, transparent 90%);
  animation: shimmer 1.8s infinite;
}

/* Keyframes */
@keyframes hue { from{ filter:hue-rotate(0deg) blur(7px)} to{ filter:hue-rotate(360deg) blur(7px)} }
@keyframes shimmer { 0% { transform: translateX(-100%) } 60%, 100% { transform: translateX(100%) } }
</style>
