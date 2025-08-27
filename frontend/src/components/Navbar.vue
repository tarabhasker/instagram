<!-- src/components/Navbar.vue -->
<template>
  <div>
    <!-- ===== Mobile: floating bottom bar (icons) ===== -->
    <teleport to="body">
      <nav
        class="md:hidden fixed z-[9999]
            bottom-[max(0.75rem,env(safe-area-inset-bottom))]
            left-[max(0.75rem,env(safe-area-inset-left))]
            right-[max(0.75rem,env(safe-area-inset-right))]
            mx-auto max-w-[720px]
            rounded-2xl border border-white/10 bg-black/60 backdrop-blur-md
            px-2 py-2 shadow-[0_20px_40px_-20px_rgba(0,0,0,.7)] pointer-events-auto"
        role="navigation" aria-label="Primary">
        <ul class="flex items-center justify-between">
          <li>
            <RouterLink
              to="/"
              :aria-current="isActive('/') ? 'page' : undefined"
              class="nav-ico"
              :class="isActive('/') && 'nav-ico--active'"
              title="Feed"
            >
              <svg viewBox="0 0 24 24" class="h-6 w-6" fill="currentColor"><path d="M12 3 2 12h3v9h6v-6h2v6h6v-9h3z"/></svg>
            </RouterLink>
          </li>

          <li>
            <RouterLink
              to="/explore"
              :aria-current="isActive('/explore') ? 'page' : undefined"
              class="nav-ico"
              :class="isActive('/explore') && 'nav-ico--active'"
              title="Explore"
            >
              <svg viewBox="0 0 24 24" class="h-6 w-6" fill="currentColor">
                <path d="M9.5 3a6.5 6.5 0 1 1 0 13A6.5 6.5 0 0 1 9.5 3zm0 2a4.5 4.5 0 1 0 0 9A4.5 4.5 0 0 0 9.5 5zm8.3 10.9 4 4-1.8 1.8-4-4 1.8-1.8z"/>
              </svg>
            </RouterLink>
          </li>

          <!-- Big center "Create" button -->
          <li>
            <RouterLink
              to="/create"
              :aria-current="isActive('/create') ? 'page' : undefined"
              class="grid size-12 place-items-center rounded-xl bg-[#e4ae87] text-black
                     shadow-[0_10px_30px_-8px_rgba(228,174,135,.6)]
                     transition hover:brightness-95 active:scale-95"
              title="Create"
            >
              <svg viewBox="0 0 24 24" class="h-6 w-6" fill="currentColor"><path d="M11 11V5h2v6h6v2h-6v6h-2v-6H5v-2z"/></svg>
            </RouterLink>
          </li>

          <li>
            <RouterLink
              to="/activity"
              :aria-current="isActive('/activity') ? 'page' : undefined"
              class="nav-ico"
              :class="isActive('/activity') && 'nav-ico--active'"
              title="Activity"
            >
              <svg viewBox="0 0 24 24" class="h-6 w-6" fill="currentColor">
                <path d="M12 22a2 2 0 0 0 2-2h-4a2 2 0 0 0 2 2zm6-6V9a6 6 0 1 0-12 0v7l-2 2v1h16v-1z"/>
              </svg>
            </RouterLink>
          </li>

          <li>
            <RouterLink
              :to="profileHref"
              :aria-current="isOnProfile ? 'page' : undefined"
              class="nav-ico"
              :class="isOnProfile && 'nav-ico--active'"
              title="Profile"
            >
              <svg viewBox="0 0 24 24" class="h-6 w-6" fill="currentColor">
                <path d="M12 12a5 5 0 1 0-5-5 5 5 0 0 0 5 5zm0 2c-5 0-9 2.5-9 5v1h18v-1c0-2.5-4-5-9-5z"/>
              </svg>
            </RouterLink>
          </li>
        </ul>
      </nav>
    </teleport>

    <!-- ===== Desktop: sticky top bar (labels) ===== -->
    <nav class="hidden md:block fixed top-0 left-0 right-0 z-50 w-full border-b border-white/10 bg-black/60 backdrop-blur-md">
      <div class="mx-auto flex h-14 w-[min(1100px,94%)] items-center gap-4">
        <!-- brand -->
        <RouterLink to="/" class="flex items-center gap-2 text-white font-semibold tracking-wide">
          <span class="inline-block size-2 rounded-full bg-[#e4ae87] shadow-[0_0_10px_#e4ae87aa]"></span>
          Flashgram
        </RouterLink>

        <div class="ml-auto flex items-center gap-1.5">
          <RouterLink
            to="/"
            :aria-current="isActive('/') ? 'page' : undefined"
            class="nav-tab"
            :class="isActive('/') && 'nav-tab--active'"
          >Feed</RouterLink>

          <RouterLink
            to="/explore"
            :aria-current="isActive('/explore') ? 'page' : undefined"
            class="nav-tab"
            :class="isActive('/explore') && 'nav-tab--active'"
          >Explore</RouterLink>

          <RouterLink
            to="/create"
            :aria-current="isActive('/create') ? 'page' : undefined"
            class="nav-tab"
            :class="isActive('/create') && 'nav-tab--active'"
          >Create</RouterLink>

          <RouterLink
            to="/activity"
            :aria-current="isActive('/activity') ? 'page' : undefined"
            class="nav-tab"
            :class="isActive('/activity') && 'nav-tab--active'"
          >Activity</RouterLink>

          <RouterLink
            :to="profileHref"
            :aria-current="isOnProfile ? 'page' : undefined"
            class="nav-tab"
            :class="isOnProfile && 'nav-tab--active'"
          >Profile</RouterLink>

          <!-- Logout button (desktop header) -->
          <button
            v-if="isAuthed"
            @click="doLogout"
            class="nav-tab"
            title="Log out"
          >
            Logout
          </button>
        </div>
      </div>
    </nav>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute, useRouter, RouterLink } from 'vue-router'
import { useAuth } from '@/composables/useAuth' // uses your simple localStorage auth

const route = useRoute()
const router = useRouter()

const { user, isAuthed, logout } = useAuth()

const me = computed(() => user.value?.username || '')
const profileHref = computed(() => (me.value ? `/@${me.value}` : '/auth'))

const isActive = (path) => {
  if (!path) return false
  if (path === '/') return route.path === '/'
  return route.path.startsWith(path)
}

const isOnProfile = computed(() =>
  route.name === 'profile'
    ? route.params?.username === me.value || route.params?.username === `@${me.value}`
    : me.value && route.path === `/@${me.value}`
)

const doLogout = async () => {
  try {
    await logout()
  } finally {
    router.push('/auth')
  }
}
</script>

<style scoped>
/* --- Mobile icon buttons --- */
.nav-ico{
  display:grid; place-items:center;
  width:2.5rem; height:2.5rem; /* 40px */
  border-radius:0.75rem;
  color:rgba(255,255,255,.8);
  border:1px solid rgba(255,255,255,.08);
  background: transparent;
  transition: transform .15s ease, background .15s ease, color .15s ease, border-color .15s ease, box-shadow .2s ease;
}
.nav-ico:hover{ background: rgba(255,255,255,.08); color:#fff; }
.nav-ico--active{
  background: rgba(255,255,255,.12);
  color:#fff;
  border-color: rgba(228,174,135,.5);
  box-shadow: 0 8px 24px -12px rgba(228,174,135,.45) inset;
}

/* --- Desktop tabs --- */
.nav-tab{
  display:inline-flex; align-items:center; gap:.5rem;
  padding:.5rem .75rem;
  border-radius:.75rem;
  font-size:.875rem; /* text-sm */
  color:rgba(255,255,255,.8);
  border:1px solid transparent;
  transition: color .15s ease, background .15s ease, border-color .15s ease, transform .15s ease;
}
.nav-tab:hover{
  color:#fff;
  background: rgba(255,255,255,.06);
  border-color: rgba(255,255,255,.12);
}
.nav-tab--active{
  color:#fff;
  background: linear-gradient(180deg, rgba(228,174,135,.18), rgba(255,255,255,.06));
  border-color: rgba(228,174,135,.55);
  box-shadow: 0 10px 24px -16px rgba(228,174,135,.45);
}
</style>
