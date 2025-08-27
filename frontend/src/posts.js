// src/stores/posts.js
import { defineStore } from 'pinia'

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:5050'

export const usePostsStore = defineStore('posts', {
  state: () => ({
    items: [],
    loading: false,
  }),
  actions: {
    async fetchAll () {
      this.loading = true
      try {
        const r = await fetch(`${BASE}/api/posts`)
        const data = await r.json()
        this.items = Array.isArray(data) ? data : []
      } finally {
        this.loading = false
      }
    },

    async createPost (payload) {
      const r = await fetch(`${BASE}/api/posts`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-user': payload.username,   // mirrors API’s username fallback
        },
        body: JSON.stringify(payload),
      })
      const data = await r.json()
      if (!r.ok) throw new Error(data?.error?.message || 'Create failed')

      // optimistic update
      this.items = [data, ...this.items]
      return data
    },
  },
})
