import { defineStore } from 'pinia'
const BASE = import.meta.env.VITE_API_URL || 'http://localhost:5050'

export const usePostsStore = defineStore('posts', {
  state: () => ({ items: [], loading: false }),
  actions: {
    async fetchPosts() {
      this.loading = true
      const res = await fetch(`${BASE}/api/posts`)
      this.items = await res.json()
      this.loading = false
    },
    async likePost(id) {
      const i = this.items.findIndex(p => p.id === id)
      if (i > -1) this.items[i].likes++
      try {
        const r = await fetch(`${BASE}/api/posts/${id}/like`, { method: 'PATCH' })
        const updated = await r.json()
        if (i > -1) this.items[i] = updated
      } catch { if (i > -1) this.items[i].likes-- }
    },
  },
})
