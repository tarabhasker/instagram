// src/router/index.js
import { createRouter, createWebHistory } from 'vue-router'

/* Lazy pages (using @ alias) */
const Feed       = () => import('@/pages/Feed.vue')
const Explore    = () => import('@/pages/Explore.vue')
const PostDetail = () => import('@/pages/PostDetail.vue')
const CreatePost = () => import('@/pages/CreatePost.vue')
const Activity   = () => import('@/pages/Activity.vue')
const Profile    = () => import('@/pages/Profile.vue')
const Auth       = () => import('@/pages/Auth.vue') // login / signup UI

function getUser () {
  try { return JSON.parse(localStorage.getItem('fg.user') || 'null') || null }
  catch { return null }
}

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/',            name: 'feed',        component: Feed },
    { path: '/explore',     name: 'explore',     component: Explore },
    { path: '/p/:id',       name: 'post',        component: PostDetail, props: true },

    // Auth
    { path: '/auth',        name: 'auth',        component: Auth, meta: { guestOnly: true } },
    {
      path: '/logout',
      name: 'logout',
      beforeEnter: (to, from, next) => {
        try { localStorage.removeItem('fg.user') } catch {}
        const back = from?.fullPath && from.fullPath !== '/logout' ? from.fullPath : '/'
        next({ path: '/auth', query: { mode: 'login', next: back } })
      }
    },

    // Protected
    { path: '/create',      name: 'create',      component: CreatePost, meta: { requiresAuth: true } },
    { path: '/activity',    name: 'activity',    component: Activity,   meta: { requiresAuth: true } },

    // Profile
    { path: '/u/:username', name: 'profile',     component: Profile, props: true },
    {
      path: '/@:username',
      name: 'profileAt',
      beforeEnter: (to, _from, next) =>
        next({ name: 'profile', params: { username: to.params.username } }),
    },

    { path: '/:pathMatch(.*)*', redirect: '/' },
  ],
  scrollBehavior: () => ({ top: 0 }),
})

/* -------- Global auth guard -------- */
router.beforeEach((to, _from, next) => {
  const user = getUser()

  // Block authed users from /auth (send to next or profile/home)
  if (to.meta?.guestOnly && user) {
    const fallback = `/u/${user.username || ''}`.replace(/\/u\/$/, '/')
    return next(String(to.query?.next || fallback))
  }

  // Protect routes
  if (to.meta?.requiresAuth && !user) {
    return next({ path: '/auth', query: { next: to.fullPath } })
  }

  next()
})

export default router
