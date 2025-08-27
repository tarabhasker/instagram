// src/composables/useAuth.js
import { ref, computed } from 'vue'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5050'

// ---- state ----
const user = ref(loadUser())

function loadUser () {
  try { return JSON.parse(localStorage.getItem('fg.user') || 'null') }
  catch { return null }
}
function persistUser (u) {
  if (u) localStorage.setItem('fg.user', JSON.stringify(u))
  else localStorage.removeItem('fg.user')
}

// ---- derived ----
const isAuthed = computed(() => !!user.value)

// ---- actions ----
async function login (username, _password) {
  if (!username) throw new Error('Username required')
  // Demo backend: verify the user exists via public endpoint
  const r = await fetch(`${API_BASE}/api/users/${encodeURIComponent(username)}`)
  if (!r.ok) throw new Error('User not found')
  const u = await r.json()
  user.value = u
  persistUser(u)
  return u
}

async function logout () {
  user.value = null
  persistUser(null)
}

function getAuthHeader () {
  return user.value?.username ? { 'x-user': user.value.username } : {}
}

export function useAuth () {
  return { user, isAuthed, login, logout, getAuthHeader }
}
