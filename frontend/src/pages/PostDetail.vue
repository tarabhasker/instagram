<!-- src/pages/PostDetail.vue -->
<template>
  <main class="relative mx-auto max-w-3xl px-4 pt-6 md:pt-8 pb-[calc(88px+env(safe-area-inset-bottom))] md:pb-10 text-neutral-200">
    <!-- Soft themed backdrop -->
    <div class="pointer-events-none absolute -inset-10 -z-10 rounded-[36px] feed-bg"></div>

    <!-- Loading -->
    <div v-if="loading" class="grid gap-4">
      <div class="rounded-3xl border border-white/10 bg-white/[0.04] overflow-hidden">
        <div class="h-14 flex items-center gap-3 px-4">
          <div class="size-9 rounded-full bg-white/10 skeleton"></div>
          <div class="h-3 w-28 rounded bg-white/10 skeleton"></div>
        </div>
        <div class="aspect-[4/5] bg-white/5 skeleton"></div>
        <div class="p-4 flex items-center gap-3">
          <div class="h-6 w-16 rounded bg-white/10 skeleton"></div>
          <div class="h-6 w-16 rounded bg-white/10 skeleton"></div>
        </div>
      </div>
    </div>

    <!-- Not found -->
    <div v-else-if="!post" class="empty-card">
      <div class="empty-orb"></div>
      <p class="text-neutral-200">Post not found.</p>
      <p class="text-neutral-500 text-sm mt-1">It may have been removed or the link is broken.</p>
    </div>

    <!-- Post -->
    <article v-else class="group relative overflow-hidden rounded-3xl border border-white/10 bg-black/40 shadow-[0_10px_36px_-14px_rgba(0,0,0,.75)]">
      <!-- faint neon rim -->
      <div class="pointer-events-none absolute -inset-[2px] rounded-[28px] neon-border"></div>

      <!-- Header -->
      <div class="relative z-10 flex items-center justify-between px-4 py-3">
        <div class="flex items-center gap-3">
          <img
            :src="author?.avatarUrl || fallbackAvatar"
            :alt="author?.username || 'user'"
            class="size-9 rounded-full object-cover border border-white/10"
            loading="lazy"
            decoding="async"
          />
          <div>
            <RouterLink :to="`/@${author?.username || post.user?.username}`" class="text-sm font-medium text-white hover:underline">
              {{ displayName }}
            </RouterLink>
            <div class="text-xs text-neutral-400">{{ shortTime(post.createdAt) }}</div>
          </div>
        </div>

        <button class="icon-btn" title="More" @click="openOptions">
          <svg viewBox="0 0 24 24" class="h-5 w-5" fill="currentColor">
            <path d="M12 8a2 2 0 110-4 2 2 0 010 4zm0 6a2 2 0 110-4 2 2 0 010 4zm0 6a2 2 0 110-4 2 2 0 010 4z"/>
          </svg>
        </button>
      </div>

      <!-- Image -->
      <div class="relative">
        <img
          :src="post.imageUrl"
          :alt="post.caption || 'post image'"
          class="w-full h-auto object-cover max-h-[78vh] md:max-h-[72vh] select-none"
          loading="lazy"
          decoding="async"
        />
        <div class="pointer-events-none absolute inset-x-0 top-0 h-10 bg-gradient-to-b from-black/30 to-transparent"></div>
        <div class="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-black/30 to-transparent"></div>

        <button class="find-similar-btn" @click="findSimilar" title="Find similar">
          <svg viewBox="0 0 24 24" class="h-4 w-4" fill="currentColor"><path d="M10 3H3v7h2V6.41l5.29 5.3 1.41-1.42L6.41 5H10V3zm4 18h7v-7h-2v3.59l-5.29-5.3-1.41 1.42L17.59 19H14v2z"/></svg>
          <span class="hidden sm:inline">Find similar</span>
        </button>
      </div>

      <!-- Actions -->
      <div class="relative z-10 px-4 py-3">
        <div class="flex items-center gap-2">
          <!-- like -->
          <button
            class="action-btn"
            :class="isLiked && 'action-btn--active'"
            @click="toggleLike"
            :title="isLiked ? 'Unlike' : 'Like'"
          >
            <svg viewBox="0 0 24 24" class="h-5 w-5" fill="currentColor">
              <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 6 4 4 6.5 4c1.78 0 3.41.97 4.24 2.36C11.57 4.97 13.2 4 14.98 4 17.48 4 19.48 6 19.48 8.5c0 3.78-3.4 6.86-8.05 11.54z"/>
            </svg>
            <span class="text-sm tabular-nums">{{ likeCount }}</span>
          </button>

          <!-- comments -->
          <button class="action-btn" @click="openComments" title="Comments">
            <svg viewBox="0 0 24 24" class="h-5 w-5" fill="currentColor">
              <path d="M20 2H4a2 2 0 0 0-2 2v18l4-4h14a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2z"/>
            </svg>
            <span class="text-sm tabular-nums">{{ commentCount }}</span>
          </button>

          <!-- share -->
          <button class="action-btn" @click="openShare" title="Share">
            <svg viewBox="0 0 24 24" class="h-5 w-5" fill="currentColor">
              <path d="M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7a3.27 3.27 0 0 0 0-1.39l7-4.11A2.99 2.99 0 1 0 15 4.91l-7.04 4.12a3 3 0 1 0 0 5.94l7.04 4.12A3 3 0 1 0 18 16.08z"/>
            </svg>
            <span class="text-sm">{{ shareCount }}</span>
          </button>

          <!-- save -->
          <button
            class="action-btn ml-auto"
            :class="isSaved && 'action-btn--active'"
            @click="toggleSave"
            :title="isSaved ? 'Unsave' : 'Save'"
          >
            <svg viewBox="0 0 24 24" class="h-5 w-5" fill="currentColor">
              <path d="M6 2h12a2 2 0 0 1 2 2v18l-8-4-8 4V4a2 2 0 0 1 2-2z"/>
            </svg>
            <span class="text-sm">{{ isSaved ? 'Saved' : 'Save' }}</span>
          </button>
        </div>

        <!-- Stats / liked by -->
        <div class="mt-3 flex items-center gap-3 text-sm">
          <div class="flex -space-x-2">
            <img
              v-for="u in likeFaces"
              :key="u.username"
              :src="u.avatarUrl || fallbackAvatar"
              :alt="u.username"
              class="size-6 rounded-full object-cover border border-white/20"
              loading="lazy"
            />
          </div>
          <div class="text-neutral-300">
            <span class="font-medium text-neutral-100">{{ likeCount }}</span> likes
            <span class="mx-2 opacity-40">·</span>
            <span class="font-medium text-neutral-100">{{ commentCount }}</span> comments
            <span class="mx-2 opacity-40">·</span>
            <span class="font-medium text-neutral-100">{{ shareCount }}</span> shares
            <span v-if="saveCount" class="mx-2 opacity-40">·</span>
            <span v-if="saveCount" class="font-medium text-neutral-100">{{ saveCount }}</span><span v-if="saveCount"> saves</span>
          </div>
        </div>

        <!-- Caption -->
        <p v-if="post.caption" class="mt-3 text-[15px] leading-snug">
          <RouterLink :to="`/@${author?.username || post.user?.username}`" class="font-medium text-neutral-100 hover:underline">
            {{ displayName }}
          </RouterLink>
          <span class="text-neutral-300"> {{ ' ' + post.caption }}</span>
        </p>

        <!-- Inline comments (last 4) -->
        <ol v-if="commentsPreview.length" class="mt-3 space-y-2">
          <li v-for="c in commentsPreview" :key="c.id" class="text-sm">
            <RouterLink :to="`/@${c.username}`" class="font-medium hover:underline">{{ c.username }}</RouterLink>
            <span class="text-neutral-300"> {{ ' ' + c.content }}</span>
            <span class="text-[11px] text-neutral-500 ml-2">{{ shortTime(c.date) }}</span>
          </li>
        </ol>

        <!-- View all comments -->
        <button
          v-if="commentCount > 0"
          class="mt-2 text-sm text-neutral-400 hover:text-neutral-200 underline underline-offset-2"
          @click="openComments"
        >
          View all {{ commentCount }} comments
        </button>

        <!-- Composer -->
        <div class="mt-3 flex items-center gap-2">
          <input
            v-model="inlineDraft"
            class="flex-1 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-neutral-100 placeholder:text-neutral-500 outline-none transition focus:ring-2 focus:ring-[#e4ae87]/30 focus:border-[#e4ae87]/60"
            :placeholder="`Comment as ${displayMe}`"
            @keydown.enter.prevent="submitInlineComment"
          />
          <button
            class="inline-flex items-center gap-2 rounded-xl px-3 h-9 text-sm font-medium border border-white/15 text-neutral-100 bg-white/5 hover:bg-white/10 disabled:opacity-60 transition"
            :disabled="!inlineDraft"
            @click="submitInlineComment"
          >
            Send
          </button>
        </div>
      </div>
    </article>

    <!-- ===== Modals (teleported) ===== -->
    <teleport to="body">
      <!-- Backdrop -->
      <transition name="fade">
        <div v-if="sheet.open" class="fixed inset-0 z-[9998] bg-black/50 backdrop-blur-[2px]" @click="closeSheet" />
      </transition>

      <!-- Comments Sheet -->
      <transition name="slide-up">
        <div v-if="sheet.type==='comments' && sheet.open" class="sheet" role="dialog" aria-modal="true">
          <header class="sheet-hd">
            <div class="font-semibold">Comments</div>
            <button class="x-btn" @click="closeSheet" title="Close">✕</button>
          </header>
          <div class="sheet-body">
            <div v-if="commentsLoading" class="text-sm text-neutral-400">Loading…</div>
            <ol v-else class="space-y-3">
              <li v-for="c in comments" :key="c.id" class="flex items-start gap-3">
                <img
                  :src="(profileCache.get(c.username)?.avatarUrl) || fallbackAvatar"
                  :alt="c.username"
                  class="size-8 rounded-full object-cover border border-white/10"
                  loading="lazy"
                />
                <div class="flex-1">
                  <div class="text-sm">
                    <RouterLink :to="`/@${c.username}`" class="font-medium hover:underline">{{ c.username }}</RouterLink>
                    <span class="text-neutral-300"> {{ ' ' + c.content }}</span>
                  </div>
                  <div class="text-[11px] text-neutral-500 mt-0.5">{{ shortTime(c.date) }}</div>
                </div>
              </li>
            </ol>
          </div>
          <footer class="sheet-ft">
            <input
              v-model="commentDraft"
              class="flex-1 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-neutral-100 placeholder:text-neutral-500 outline-none"
              :placeholder="`Comment as ${displayMe}`"
              @keydown.enter.prevent="submitModalComment"
            />
            <button
              class="ml-2 rounded-xl px-3 h-9 text-sm font-medium border border-white/15 text-neutral-100 bg-white/5 hover:bg-white/10 disabled:opacity-60 transition"
              :disabled="!commentDraft"
              @click="submitModalComment"
            >
              Send
            </button>
          </footer>
        </div>
      </transition>

      <!-- Share Sheet -->
      <transition name="slide-up">
        <div v-if="sheet.type==='share' && sheet.open" class="sheet" role="dialog" aria-modal="true">
          <header class="sheet-hd">
            <div class="font-semibold">Share</div>
            <button class="x-btn" @click="closeSheet" title="Close">✕</button>
          </header>
          <div class="sheet-body">
            <p class="text-sm text-neutral-400 mb-3">Share to people you follow:</p>
            <ul class="grid gap-2 max-h-[46vh] overflow-y-auto pr-1">
              <li
                v-for="f in followingProfiles"
                :key="f.username"
                class="flex items-center gap-3 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2"
                @click="toggleRecipient(f.username)"
              >
                <img :src="f.avatarUrl || fallbackAvatar" :alt="f.username" class="size-8 rounded-full object-cover border border-white/10" />
                <div class="flex-1">
                  <div class="text-sm font-medium">{{ f.username }}</div>
                  <div class="text-xs text-neutral-500">{{ [f.firstName, f.lastName].filter(Boolean).join(' ') }}</div>
                </div>
                <input type="checkbox" class="accent-[#e4ae87]" :checked="recipients.has(f.username)" @change.stop="toggleRecipient(f.username)" />
              </li>
            </ul>
          </div>
          <footer class="sheet-ft">
            <button class="rounded-xl px-3 h-9 text-sm font-medium border border-white/15 text-neutral-100 bg-white/5 hover:bg-white/10 transition" @click="shareNative">
              System Share
            </button>
            <button
              class="ml-auto rounded-xl px-3 h-9 text-sm font-semibold text-black bg-[#e4ae87] hover:bg-[#d6a17c] transition"
              @click="sendShare"
            >
              Send
            </button>
          </footer>
        </div>
      </transition>

      <!-- Options Sheet -->
      <transition name="slide-up">
        <div v-if="sheet.type==='options' && sheet.open" class="sheet" role="dialog" aria-modal="true">
          <header class="sheet-hd">
            <div class="font-semibold">Options</div>
            <button class="x-btn" @click="closeSheet" title="Close">✕</button>
          </header>
          <div class="sheet-body">
            <ul class="grid gap-2">
              <li><button class="opt-row" @click="copyLink"><span>Copy link</span></button></li>
              <li><button class="opt-row" @click="openAuthor"><span>View profile</span></button></li>
              <li><button class="opt-row"><span>Mute this user</span></button></li>
              <li><button class="opt-row text-red-300"><span>Report</span></button></li>
            </ul>
          </div>
          <footer class="sheet-ft">
            <button class="rounded-xl px-3 h-9 text-sm font-medium border border-white/15 text-neutral-100 bg-white/5 hover:bg-white/10 transition" @click="closeSheet">Close</button>
          </footer>
        </div>
      </transition>
    </teleport>
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
const postId = computed(() => route.params.id)

/* session (logged-in user) */
const sessionUser = ref(null)
const currentUsername = computed(() => sessionUser.value?.username || 'guest')
const displayMe = computed(() => currentUsername.value)

/* state */
const loading = ref(true)
const post = ref(null)
const comments = ref([])
const commentsLoading = ref(false)
const inlineDraft = ref('')
const commentDraft = ref('')
const me = ref(null)                      // full profile of logged-in user (optional)
const profileCache = ref(new Map())

/* sheets */
const sheet = ref({ open: false, type: null })
const recipients = ref(new Set())

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

/* loader */
const load = async () => {
  loading.value = true
  post.value = null
  try {
    // hydrate session
    try { sessionUser.value = JSON.parse(localStorage.getItem('fg.user') || '{}') } catch { sessionUser.value = null }
    if (sessionUser.value?.username) {
      me.value = await fetchUser(sessionUser.value.username)
    } else {
      me.value = null
    }

    const item = await fetchJSON(`/api/posts/${encodeURIComponent(postId.value)}`)
    // coerce shapes
    item.likes = Array.isArray(item.likes) ? item.likes : []
    item.saves = Array.isArray(item.saves) ? item.saves : []
    item.comments = Array.isArray(item.comments) ? item.comments : []
    item.shareEvents = Array.isArray(item.shareEvents) ? item.shareEvents : []
    post.value = item

    await fetchUser(item.user?.username)

    // warm liker faces (first 6)
    const likerUsernames = (item.likes || []).slice(0, 6)
    await Promise.all(likerUsernames.map(fetchUser))

    // preview comments: fetch full list then slice in UI
    try {
      comments.value = await fetchJSON(`/api/posts/${encodeURIComponent(postId.value)}/comments`)
    } catch {
      comments.value = item.comments || []
    }
    // warm commenters
    await Promise.all([...new Set(comments.value.map(c => c.username))].map(fetchUser))
  } finally {
    loading.value = false
  }
}

/* computed */
const author = computed(() => profileCache.value.get(post.value?.user?.username))
const displayName = computed(() => {
  const a = author.value
  if (!a) return post.value?.user?.username || 'user'
  const full = [a.firstName, a.lastName].filter(Boolean).join(' ')
  return full || a.username
})
const likeCount = computed(() => (Array.isArray(post.value?.likes) ? post.value.likes.length : 0))
const commentCount = computed(() => (Array.isArray(comments.value) ? comments.value.length : 0))
const shareCount = computed(() => (Array.isArray(post.value?.shareEvents) ? post.value.shareEvents.length : 0))
const saveCount = computed(() => (Array.isArray(post.value?.saves) ? post.value.saves.length : 0))

const isLiked = computed(() => Array.isArray(post.value?.likes) && post.value.likes.includes(currentUsername.value))
const isSaved = computed(() => Array.isArray(post.value?.saves) && post.value.saves.includes(currentUsername.value))

const likeFaces = computed(() => {
  const list = (post.value?.likes || []).slice(0, 6)
  return list.map(u => profileCache.value.get(u) || { username: u, avatarUrl: fallbackAvatar })
})

const commentsPreview = computed(() => (comments.value || []).slice(-4))

const followingProfiles = computed(() =>
  (me.value?.following || []).map(u => profileCache.value.get(u)).filter(Boolean)
)

/* actions */
const toggleLike = async () => {
  if (!post.value) return
  const user = currentUsername.value
  const had = isLiked.value
  // optimistic
  post.value.likes = had ? post.value.likes.filter(u => u !== user) : [...post.value.likes, user]
  try {
    const updated = await fetchJSON(`/api/posts/${encodeURIComponent(post.value.id)}/like`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: user })
    })
    post.value.likes = Array.isArray(updated.likes) ? updated.likes : post.value.likes
  } catch {
    // revert
    post.value.likes = had ? [...post.value.likes, user] : post.value.likes.filter(u => u !== user)
  }
}

const toggleSave = async () => {
  if (!post.value) return
  const user = currentUsername.value
  const had = isSaved.value
  post.value.saves = had ? post.value.saves.filter(u => u !== user) : [...post.value.saves, user]
  try {
    const updated = await fetchJSON(`/api/posts/${encodeURIComponent(post.value.id)}/save`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: user })
    })
    post.value.saves = Array.isArray(updated.saves) ? updated.saves : post.value.saves
  } catch {
    post.value.saves = had ? [...post.value.saves, user] : post.value.saves.filter(u => u !== user)
  }
}

/* comments */
const openComments = async () => {
  if (!post.value) return
  sheet.value = { open: true, type: 'comments' }
  commentsLoading.value = true
  commentDraft.value = ''
  try {
    comments.value = await fetchJSON(`/api/posts/${encodeURIComponent(post.value.id)}/comments`)
    await Promise.all([...new Set(comments.value.map(c => c.username))].map(fetchUser))
  } finally {
    commentsLoading.value = false
  }
}
const submitInlineComment = async () => {
  if (!inlineDraft.value.trim() || !post.value) return
  const user = currentUsername.value
  try {
    const item = await fetchJSON(`/api/posts/${encodeURIComponent(post.value.id)}/comments`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: user, content: inlineDraft.value.trim() })
    })
    comments.value.push(item)
    post.value.comments = Array.isArray(post.value.comments) ? post.value.comments : []
    post.value.comments.push(item)
    inlineDraft.value = ''
  } catch {}
}
const submitModalComment = async () => {
  if (!commentDraft.value.trim() || !post.value) return
  const user = currentUsername.value
  try {
    const item = await fetchJSON(`/api/posts/${encodeURIComponent(post.value.id)}/comments`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: user, content: commentDraft.value.trim() })
    })
    comments.value.push(item)
    post.value.comments = Array.isArray(post.value.comments) ? post.value.comments : []
    post.value.comments.push(item)
    commentDraft.value = ''
  } catch {}
}

/* share */
const openShare = async () => {
  if (!post.value) return
  sheet.value = { open: true, type: 'share' }
  recipients.value = new Set()
  // warm following profiles
  await Promise.all((me.value?.following || []).map(fetchUser))
}
const toggleRecipient = (u) => {
  const s = new Set(recipients.value)
  if (s.has(u)) s.delete(u); else s.add(u)
  recipients.value = s
}
const shareNative = async () => {
  if (!post.value) return
  const url = `${location.origin}/p/${post.value.id}`
  const text = post.value.caption || 'Check this out'
  const user = currentUsername.value
  try {
    if (navigator.share) await navigator.share({ title: 'Flashgram', text, url })
    // record even if system share
    await fetchJSON(`/api/posts/${encodeURIComponent(post.value.id)}/share`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: user, recipients: [], method: 'native' })
    })
    post.value.shareEvents = Array.isArray(post.value.shareEvents) ? post.value.shareEvents : []
    post.value.shareEvents.push({ id: 'e_'+Math.random().toString(36).slice(2,9), type:'share', from: user, to: null, method:'native', date: new Date().toISOString() })
  } catch {
    // best-effort; ignore
  }
}
const sendShare = async () => {
  if (!post.value) return
  const user = currentUsername.value
  try {
    const payload = { username: user, recipients: [...recipients.value], method: 'dm' }
    const created = await fetchJSON(`/api/posts/${encodeURIComponent(post.value.id)}/share`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    post.value.shareEvents = Array.isArray(post.value.shareEvents) ? post.value.shareEvents : []
    post.value.shareEvents.push(...created)
  } catch {}
  closeSheet()
}

/* options */
const openOptions = () => { sheet.value = { open: true, type: 'options' } }
const copyLink = async () => {
  if (!post.value) return
  const url = `${location.origin}/p/${post.value.id}`
  try { await navigator.clipboard.writeText(url); alert('Link copied!') } catch {}
}
const openAuthor = () => {
  if (!post.value) return
  router.push(`/@${post.value.user?.username}`)
}
const closeSheet = () => { sheet.value = { open: false, type: null } }

/* misc */
const findSimilar = () => {
  if (!post.value) return
  window.location.href = `/explore?similar=${encodeURIComponent(post.value.id)}`
}
const shortTime = (iso) => {
  if (!iso) return ''
  const d = new Date(iso)
  const now = new Date()
  const diff = Math.floor((now - d) / 1000)
  if (diff < 60) return `${diff}s`
  if (diff < 3600) return `${Math.floor(diff / 60)}m`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h`
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

/* effects */
onMounted(load)
watch(postId, load)
</script>

<style scoped>
/* Background (palette, organic, cheap) */
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

.action-btn{
  display:inline-flex; align-items:center; gap:.5rem;
  padding:.45rem .65rem; border-radius:.75rem;
  color:rgba(255,255,255,.85);
  border:1px solid rgba(255,255,255,.08);
  background: rgba(255,255,255,.04);
  transition: background .15s ease, color .15s ease, border-color .15s ease, transform .15s ease;
}
.action-btn:hover{ background: rgba(255,255,255,.08); border-color: rgba(255,255,255,.16); color:#fff; }
.action-btn--active{
  background: linear-gradient(180deg, rgba(228,174,135,.18), rgba(255,255,255,.06));
  border-color: rgba(228,174,135,.55);
  box-shadow: 0 10px 24px -16px rgba(228,174,135,.45);
  color:#fff;
}

/* Find similar overlay */
.find-similar-btn{
  position:absolute; right:.75rem; bottom:.75rem;
  display:inline-flex; align-items:center; gap:.5rem;
  padding:.5rem .7rem; border-radius:.75rem;
  font-size:.8rem; font-weight:600;
  color:#0b0b0b; background:#e4ae87;
  transition: transform .15s ease, filter .15s ease, opacity .2s ease;
  box-shadow: 0 12px 24px -12px rgba(228,174,135,.55);
  opacity: 1; /* visible on mobile */
}
@media (hover:hover){
  .group .find-similar-btn { opacity: 0; }
  .group:hover .find-similar-btn { opacity: 1; }
}
.find-similar-btn:hover{ filter:brightness(0.95); transform:translateY(-1px); }

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

/* --- Sheets / Modals --- */
.sheet{
  position: fixed; inset-inline: 0; bottom: 0; z-index: 9999;
  margin-inline: auto; width: min(720px, 94%); max-height: 80vh;
  border-radius: 20px 20px 0 0; overflow: hidden;
  border: 1px solid rgba(255,255,255,.12);
  background: rgba(0,0,0,.7); backdrop-filter: blur(8px);
  display: grid; grid-template-rows: auto 1fr auto;
}
.sheet-hd{ display:flex; align-items:center; justify-content:space-between; padding: .85rem 1rem; border-bottom: 1px solid rgba(255,255,255,.08); }
.sheet-body{ padding: .9rem 1rem; overflow: auto; }
.sheet-ft{ display:flex; align-items:center; gap:.5rem; padding: .75rem 1rem; border-top: 1px solid rgba(255,255,255,.08); }
.x-btn{ width:2rem; height:2rem; border-radius:.6rem; border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.06); }
.opt-row{ width:100%; display:flex; align-items:center; gap:.75rem; justify-content:space-between; padding:.7rem .9rem; border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.04); border-radius:.8rem; }

.fade-enter-active,.fade-leave-active{ transition: opacity .2s }
.fade-enter-from,.fade-leave-to{ opacity: 0 }
.slide-up-enter-active,.slide-up-leave-active{ transition: transform .22s ease, opacity .22s ease }
.slide-up-enter-from,.slide-up-leave-to{ transform: translateY(8px); opacity: 0 }

/* Animations */
@keyframes hue { from{ filter:hue-rotate(0deg) blur(7px)} to{ filter:hue-rotate(360deg) blur(7px)} }
@keyframes shimmer { 0% { transform: translateX(-100%) } 60%, 100% { transform: translateX(100%) } }
</style>
