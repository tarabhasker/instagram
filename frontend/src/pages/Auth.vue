<!-- src/pages/Auth.vue -->
<template>
    <main class="relative min-h-[100svh] bg-neutral-950 text-neutral-200 overflow-hidden grid place-items-center px-4 py-10">
      <!-- Ambient background layers -->
      <div class="pointer-events-none absolute inset-0 z-0">
        <!-- slow beams / blobs -->
        <div class="absolute -top-40 -left-32 size-[28rem] rounded-full opacity-25"
             style="background: radial-gradient(60% 60% at 50% 50%, #66131344, transparent 70%);"></div>
        <div class="absolute -bottom-40 -right-32 size-[30rem] rounded-full opacity-25"
             style="background: radial-gradient(60% 60% at 50% 50%, #88455444, transparent 70%);"></div>
  
        <!-- soft scanline + grid mask -->
        <div class="absolute inset-0 opacity-[.06] scanline"></div>
        <div class="absolute inset-0 pointer-events-none opacity-[.12] grid-mask"></div>
      </div>
  
      <!-- Card -->
      <section
        class="relative z-10 w-[min(520px,96vw)]
                   overflow-hidden rounded-3xl border border-white/10
                   bg-black/40 backdrop-blur-md
                   shadow-[0_18px_48px_-18px_rgba(0,0,0,.8)]"
        role="dialog" aria-labelledby="authTitle"
      >
        <!-- neon perimeter glow -->
        <div class="pointer-events-none absolute inset-0">
          <div class="absolute -inset-[2px] rounded-[26px] neon-border"></div>
        </div>
  
        <!-- top accent bar -->
        <div class="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-[#e4ae87]/55 to-transparent"></div>
  
        <!-- content -->
        <div class="relative z-10 p-6 md:p-8">
          <!-- brand + title (centered) -->
          <div class="mx-auto grid place-items-center text-center">
            <div class="relative mb-3">
              <div class="pointer-events-none absolute -inset-[10px] rounded-full neon-ring"></div>
              <div class="grid size-12 place-items-center rounded-2xl bg-black/60 border border-white/10">
                <span class="inline-block size-2 rounded-full bg-[#e4ae87] shadow-[0_0_14px_#e4ae87aa]"></span>
              </div>
            </div>
            <h1 id="authTitle" class="text-2xl md:text-3xl font-semibold tracking-tight text-white">
              {{ mode === 'login' ? 'Welcome back' : 'Create your account' }}
            </h1>
            <p class="mt-1 text-sm text-neutral-400">
              Flashgram
            </p>
          </div>
  
          <!-- Tabs -->
          <div class="relative mt-6 mb-6 flex items-center justify-center">
            <div class="relative inline-flex items-center gap-2 p-1.5 rounded-2xl border border-white/10 bg-black/35">
              <button class="tab-chip" :class="mode==='login' ? 'tab-chip--active' : ''" @click="mode='login'">
                <span class="relative z-10 flex items-center gap-2">
                  <svg viewBox="0 0 24 24" class="h-4 w-4" fill="currentColor"><path d="M10 17l5-5-5-5v10zM4 4h2v16H4z"/></svg>
                  Login
                </span>
                <span class="tab-wash"></span>
              </button>
              <button class="tab-chip" :class="mode==='signup' ? 'tab-chip--active' : ''" @click="mode='signup'">
                <span class="relative z-10 flex items-center gap-2">
                  <svg viewBox="0 0 24 24" class="h-4 w-4" fill="currentColor"><path d="M12 12a4.8 4.8 0 1 0 0-9.6 4.8 4.8 0 0 0 0 9.6zM12 14.4c-3.2 0-9.6 1.6-9.6 4.8V22h19.2v-2.8c0-3.2-6.4-4.8-9.6-4.8z"/></svg>
                  Sign up
                </span>
                <span class="tab-wash"></span>
              </button>
            </div>
          </div>
  
          <!-- Form -->
          <form @submit.prevent="onSubmit" class="grid gap-4">
            <div class="grid md:grid-cols-2 gap-3" v-if="mode==='signup'">
              <div class="space-y-1">
                <label class="text-sm text-neutral-300">First name</label>
                <input v-model.trim="firstName" class="input" placeholder="Enter your first name" autocomplete="given-name" />
              </div>
              <div class="space-y-1">
                <label class="text-sm text-neutral-300">Last name</label>
                <input v-model.trim="lastName" class="input" placeholder="Enter your last name" autocomplete="family-name" />
              </div>
            </div>
  
            <div class="space-y-1">
              <label class="text-sm text-neutral-300">Username</label>
              <input v-model.trim="username" class="input" placeholder="Enter your username" autocomplete="username" required />
            </div>
  
            <div class="space-y-1" v-if="mode==='signup'">
              <label class="text-sm text-neutral-300">Email</label>
              <input v-model.trim="email" class="input" placeholder="Enter your email" type="email" autocomplete="email" required />
            </div>
  
            <div class="space-y-1">
              <label class="text-sm text-neutral-300">Password</label>
              <input v-model="password" class="input" type="password" placeholder="Enter your password" :autocomplete="mode==='login' ? 'current-password' : 'new-password'" required />
              <p v-if="mode==='login'" class="text-xs text-neutral-400">
                Tip: Seeded users use password <span class="font-semibold text-neutral-200">demo</span>
              </p>
            </div>
  
            <p v-if="error" class="rounded-lg border border-red-500/20 bg-red-500/10 p-2 text-red-200 text-sm">{{ error }}</p>
  
            <button
              type="submit"
              class="group relative overflow-hidden inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-medium text-black bg-[#e4ae87] hover:bg-[#daa8ae] active:scale-[.99] transition"
              :disabled="submitting"
            >
              <span class="relative z-10" v-if="!submitting">{{ mode==='login' ? 'Login' : 'Create account' }}</span>
              <span class="relative z-10 inline-flex items-center gap-2" v-else><span class="spinner"></span> {{ mode==='login' ? 'Signing in…' : 'Creating…' }}</span>
              <span class="absolute inset-0 -z-0 opacity-0 group-hover:opacity-100 transition"
                    style="background: radial-gradient(80% 80% at 50% 0%, #e4ae8722, transparent 60%);"></span>
            </button>
  
            <!-- mini footnote -->
            <p class="mt-2 text-xs text-neutral-500 text-center">
              By continuing, you agree to our imaginary Terms & Privacy.
            </p>
          </form>
        </div>
      </section>
    </main>
  </template>
  
  <script setup>
  import { ref } from 'vue'
  import { useRouter } from 'vue-router'
  import { useAuth } from '@/composables/useAuth'
  
  const router = useRouter()
  const { login: setAuthUser } = useAuth()
  
  const API = (import.meta.env.VITE_API_BASE || import.meta.env.VITE_API_URL || '').replace(/\/+$/, '')
  
  const mode = ref('login') // or 'signup'
  const username = ref('')
  const password = ref('')
  const email    = ref('')
  const firstName = ref('')
  const lastName  = ref('')
  
  const submitting = ref(false)
  const error = ref(null)
  
  async function onSubmit () {
    error.value = null
    submitting.value = true
    try {
      const path = mode.value === 'login' ? '/api/auth/login' : '/api/auth/signup'
      const body = mode.value === 'login'
        ? { username: username.value, password: password.value }
        : { username: username.value, email: email.value, password: password.value, firstName: firstName.value, lastName: lastName.value }
  
      const r = await fetch(`${API}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })
      const data = await r.json()
      if (!r.ok) throw new Error(data?.error?.message || 'Auth failed')
  
      // minimal "session"
      localStorage.setItem('fg.token', data.token || '')
      localStorage.setItem('fg.user', JSON.stringify({ username: data.user.username, avatarUrl: data.user.avatarUrl }))
  
      // sync global auth state (fetches the full user via /api/users/:username)
      await setAuthUser(data.user.username)
  
      // go to profile
      router.push(`/u/${encodeURIComponent(data.user.username)}`)
    } catch (e) {
      error.value = e.message || 'Something went wrong'
    } finally {
      submitting.value = false
    }
  }
  </script>
  
  <style scoped>
  @reference "tailwindcss";
  
  /* Inputs */
  .input{
    @apply w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2.5
           text-neutral-100 placeholder:text-neutral-500 outline-none transition
           focus:ring-2 focus:ring-[#e4ae87]/30 focus:border-[#e4ae87]/60;
  }
  
  /* Futuristic accents */
  .neon-border{
    background: linear-gradient(120deg, #661313, #661414 30%, #884554 55%, #e4ae87 80%, #daa8ae);
    filter: blur(6px);
    opacity:.22;
    animation: hue 18s linear infinite;
  }
  .neon-ring{
    background: conic-gradient(from 0deg, #661313, #661414, #884554, #e4ae87, #daa8ae, #661313);
    filter: blur(6px);
    opacity:.45;
    animation: spin 10s linear infinite;
  }
  .scanline{
    background: repeating-linear-gradient(
      to bottom,
      rgba(255,255,255,0.45) 0,
      rgba(255,255,255,0.45) 1px,
      transparent 1px,
      transparent 3px
    );
  }
  .grid-mask{
    background:
      linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px) 0 0/ 16px 16px,
      linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px) 0 0/ 16px 16px;
    mask-image: radial-gradient(120% 80% at 50% 120%, black, transparent);
  }
  
  /* Tabs */
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
  
  /* Spinner */
  .spinner{
    width:12px;height:12px;border-radius:9999px;
    border:2px solid rgba(255,255,255,.35);
    border-top-color:#e4ae87;
    animation: spin 1s linear infinite;
  }
  
  /* Keyframes */
  @keyframes hue { from{ filter:hue-rotate(0deg) blur(6px) } to{ filter:hue-rotate(360deg) blur(6px) } }
  @keyframes spin { from{ transform:rotate(0) } to{ transform:rotate(360deg) } }
  </style>
  