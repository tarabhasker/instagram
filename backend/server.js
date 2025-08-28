// server.js
// Flashgram demo API (ESM). Node 18+ recommended.

import express from 'express'
import cors from 'cors'
import morgan from 'morgan'
import fs from 'fs'
import path from 'path'
import fetch from 'node-fetch' // if Node >=18 you can use global fetch
import crypto from 'crypto';

const hash = (s) => crypto.createHash('sha256').update(String(s)).digest('hex');
const toPublicUser = (u) => { const { password, ...safe } = u; return safe; };
/* ----------------------------- Setup ----------------------------- */

const app = express()
app.use(cors({
  origin: [
    "http://localhost:5173",           
    "https://tara-takehome-ui.vercel.app"  
  ],
  credentials: true
}));
app.use(express.json())
app.use(morgan('dev'))

const DB_PATH = path.resolve('./posts.json')

// Ensure DB file exists
if (!fs.existsSync(DB_PATH)) {
  fs.writeFileSync(
    DB_PATH,
    JSON.stringify({ users: [], posts: [] }, null, 2),
    'utf8'
  )
}

const load = () => JSON.parse(fs.readFileSync(DB_PATH, 'utf8'))
const save = (data) => fs.writeFileSync(DB_PATH, JSON.stringify(data, null, 2))

// server.js (ADD these routes just above "/* ----------------------------- AI Proxy -------------------------- */")

/* ----------------------------- Auth -------------------------------- */

// Login (demo): accept { username, password }
app.post('/api/auth/login', (req, res) => {
  const { username, password } = req.body || {};
  if (!username || !password) {
    return res.status(400).json({ error: { code: 'BAD_REQUEST', message: 'username and password required' } });
  }
  const db = load();
  const u = db.users?.find(x => x.username === username);
  if (!u) return res.status(401).json({ error: { code: 'UNAUTHORIZED', message: 'Invalid credentials' } });

  // Seeded users have "hashed-demo" — accept "demo" for convenience
  const ok = u.password === 'hashed-demo'
    ? (password === 'demo')
    : (u.password === hash(password));

  if (!ok) return res.status(401).json({ error: { code: 'UNAUTHORIZED', message: 'Invalid credentials' } });

  // silly demo token
  const token = 't_' + Math.random().toString(36).slice(2, 10);
  res.json({ token, user: toPublicUser(u) });
});

// Sign up (demo): { username, email, password, firstName?, lastName? }
app.post('/api/auth/signup', (req, res) => {
  let { username, email, password, firstName = '', lastName = '' } = req.body || {};
  username = (username || '').trim();
  email    = (email || '').trim();

  if (!username || !email || !password) {
    return res.status(400).json({ error: { code: 'BAD_REQUEST', message: 'username, email, password required' } });
  }

  const db = load();
  if (!db.users) db.users = [];
  if (db.users.some(u => u.username === username)) {
    return res.status(409).json({ error: { code: 'CONFLICT', message: 'username already exists' } });
  }

  const u = {
    id: 'u_' + Math.random().toString(36).slice(2, 9),
    firstName, lastName, username,
    dob: '',
    email,
    password: hash(password),               // store hash for new users
    dateRegistered: new Date().toISOString(),
    followers: [], following: [],
    posts: [], liked: [], saved: [],
    bio: '',
    avatarUrl: `https://i.pravatar.cc/300?u=${encodeURIComponent(username)}`,
  };

  db.users.push(u);
  save(db);
  const token = 't_' + Math.random().toString(36).slice(2, 10);
  res.status(201).json({ token, user: toPublicUser(u) });
});

// Current user (by header)
app.get('/api/auth/me', (req, res) => {
  const username = req.get('x-user');
  if (!username) return res.status(401).json({ error: { code: 'UNAUTHORIZED', message: 'missing x-user' } });
  const db = load();
  const u = db.users?.find(x => x.username === username);
  if (!u) return res.status(404).json({ error: { code: 'NOT_FOUND', message: 'user not found' } });
  res.json({ user: toPublicUser(u) });
});


/* --------------------------- Utilities --------------------------- */

// Normalize a post object so the app can rely on arrays.
const normalizePost = (p) => {
  const base = {
    id: p.id,
    imageUrl: p.imageUrl,
    caption: p.caption ?? '',
    user: p.user || (p.username ? { username: p.username } : { username: 'unknown' }),
    createdAt: p.createdAt || p.date || new Date().toISOString(),
    tags: Array.isArray(p.tags) ? p.tags : [],
    hashtags: Array.isArray(p.hashtags) ? p.hashtags : [],
    labels: Array.isArray(p.labels) ? p.labels : [],
  }
  return {
    ...base,
    likes: Array.isArray(p.likes) ? p.likes : [],
    saves: Array.isArray(p.saves) ? p.saves : [],
    comments: Array.isArray(p.comments) ? p.comments : [],
    likeEvents: Array.isArray(p.likeEvents) ? p.likeEvents : [],
    saveEvents: Array.isArray(p.saveEvents) ? p.saveEvents : [],
    shareEvents: Array.isArray(p.shareEvents) ? p.shareEvents : [],
    link: p.link || '',
  }
}

const ensureUsers = () => {
  const db = load()
  if (!db.users) db.users = []
  if (db.users.length === 0) {
    db.users.push({
      id: 'u_' + Math.random().toString(36).slice(2, 9),
      firstName: 'Bella',
      lastName: '',
      username: 'b3llaaaxx',
      dob: '1996-10-09',
      email: 'bella@example.com',
      password: 'hashed-demo', // never expose in API responses
      dateRegistered: new Date().toISOString(),
      followers: ['alice89', 'tarajaneee'],
      following: ['alice89', 'tarajaneee', 'daryldixon1', 'therickgrimes'],
      posts: [],
      liked: [],
      saved: [],
      bio: 'Matcha fanatic',
      avatarUrl: 'https://i.pravatar.cc/300?img=47',
    })
    save(db)
  }
}
ensureUsers()

const ensureSeedPost = () => {
  const db = load()
  if (!db.posts) db.posts = []
  if (db.posts.length === 0) {
    const id = 'p_' + Math.random().toString(36).slice(2, 9)
    db.posts.push(
      normalizePost({
        id,
        user: { username: 'alice89' },
        imageUrl: 'https://picsum.photos/seed/first/900/900',
        caption: 'First post!',
        tags: ['welcome'],
        createdAt: new Date().toISOString(),
        likes: [],
        saves: [],
        comments: [],
        link: '',
      })
    )
    // if alice exists, attach
    const alice = db.users.find((u) => u.username === 'alice89')
    if (alice) {
      alice.posts = alice.posts || []
      alice.posts.unshift(id)
    }
    save(db)
  }
}
ensureSeedPost()

const getUser = (username) => {
  const db = load()
  return db.users.find((u) => u.username === username)
}

/* ----------------------------- Posts ----------------------------- */

app.get('/api/posts', (req, res) => {
  const db = load()
  const items = (db.posts || [])
    .map(normalizePost)
    .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
  res.json(items)
})

app.get('/api/posts/:id', (req, res) => {
  const db = load()
  const p = (db.posts || []).find((x) => x.id === req.params.id)
  if (!p) return res.status(404).json({ error: { code: 'NOT_FOUND', message: 'Post not found' } })
  res.json(normalizePost(p))
})

app.post('/api/posts', (req, res) => {
  let { username, imageUrl, caption, hashtags = [], labels = [], tags = [] } = req.body || {}
  username = username || req.get('x-user') || 'bellaswan'
  if (!imageUrl || !caption) {
    return res.status(400).json({
      error: { code: 'BAD_REQUEST', message: 'imageUrl and caption required' }
    })
  }

  const db = load()
  const id = 'p_' + Math.random().toString(36).slice(2, 9)
  const post = normalizePost({
    id,
    imageUrl,
    user: { username },
    createdAt: new Date().toISOString(),
    caption,
    likes: [],
    saves: [],
    comments: [],
    hashtags,     // e.g. ["#forest-whispers", "#couple"]
    tags,         // e.g. ["forest","park","cloud"]  <-- from Detected
    labels,       // optional, if you keep it for other UI
    link: '',
  })

  db.posts.push(post)

  const u = db.users.find((x) => x.username === username)
  if (u) {
    u.posts = Array.isArray(u.posts) ? u.posts : []
    u.posts.unshift(id)
  }

  save(db)
  res.status(201).json(post)
})



/* --------- Likes / Saves (toggle + mirror to user doc) ---------- */

app.patch('/api/posts/:id/like', (req, res) => {
    const { username = 'guest' } = req.body || {}
    const db = load()
    const p = (db.posts || []).find(x => x.id === req.params.id)
    if (!p) return res.status(404).json({ error:{ code:'NOT_FOUND', message:'Post not found' } })
  
    p.likes = Array.isArray(p.likes) ? p.likes : []
    p.likeEvents = Array.isArray(p.likeEvents) ? p.likeEvents : []
  
    const had = p.likes.includes(username)
    if (had) p.likes = p.likes.filter(u => u !== username)
    else p.likes.push(username)
  
    p.likeEvents.push({
      id: 'e_' + Math.random().toString(36).slice(2,9),
      type: had ? 'unlike' : 'like',
      username,
      date: new Date().toISOString(),
    })
  
    const u = db.users?.find(x => x.username === username)
    if (u) {
      u.liked = Array.isArray(u.liked) ? u.liked : []
      const j = u.liked.indexOf(p.id)
      if (had && j >= 0) u.liked.splice(j, 1)
      if (!had && j < 0) u.liked.unshift(p.id)
    }
  
    save(db)
    res.json(normalizePost(p))
  })
  

  app.patch('/api/posts/:id/save', (req, res) => {
    const { username = 'guest' } = req.body || {}
    const db = load()
    const p = (db.posts || []).find(x => x.id === req.params.id)
    if (!p) return res.status(404).json({ error:{ code:'NOT_FOUND', message:'Post not found' } })
  
    p.saves = Array.isArray(p.saves) ? p.saves : []
    p.saveEvents = Array.isArray(p.saveEvents) ? p.saveEvents : []
  
    const had = p.saves.includes(username)
    if (had) p.saves = p.saves.filter(u => u !== username)
    else p.saves.push(username)
  
    p.saveEvents.push({
      id: 'e_' + Math.random().toString(36).slice(2,9),
      type: had ? 'unsave' : 'save',
      username,
      date: new Date().toISOString(),
    })
  
    const u = db.users?.find(x => x.username === username)
    if (u) {
      u.saved = Array.isArray(u.saved) ? u.saved : []
      const j = u.saved.indexOf(p.id)
      if (had && j >= 0) u.saved.splice(j, 1)
      if (!had && j < 0) u.saved.unshift(p.id)
    }
  
    save(db)
    res.json(normalizePost(p))
  })


// Create share events (DM-style or native share)
app.post('/api/posts/:id/share', (req, res) => {
    const { username, recipients = [], method = 'dm' } = req.body || {}
    if (!username) return res.status(400).json({ error:{ code:'BAD_REQUEST', message:'username required' } })
  
    const db = load()
    const p = (db.posts || []).find(x => x.id === req.params.id)
    if (!p) return res.status(404).json({ error:{ code:'NOT_FOUND', message:'Post not found' } })
  
    p.shareEvents = Array.isArray(p.shareEvents) ? p.shareEvents : []
    const now = new Date().toISOString()
  
    // If no recipients (system share), still record one event
    const tos = recipients.length ? recipients : [null]
    const created = tos.map(to => ({
      id: 'e_' + Math.random().toString(36).slice(2,9),
      type: 'share',
      from: username,
      to,
      method,          // 'dm' | 'native'
      date: now,
    }))
  
    p.shareEvents.push(...created)
    save(db)
    res.status(201).json(created)
  })
  
  

/* ---------------------------- Comments --------------------------- */

// List comments (chronological)
app.get('/api/posts/:id/comments', (req, res) => {
  const db = load()
  const p = (db.posts || []).find((x) => x.id === req.params.id)
  if (!p) return res.status(404).json({ error: { code: 'NOT_FOUND', message: 'Post not found' } })
  const list = Array.isArray(p.comments) ? p.comments : []
  list.sort((a, b) => new Date(a.date) - new Date(b.date))
  res.json(list)
})

// Add a comment
app.post('/api/posts/:id/comments', (req, res) => {
  const { username, content } = req.body || {}
  if (!username || !content) {
    return res
      .status(400)
      .json({ error: { code: 'BAD_REQUEST', message: 'username and content required' } })
  }
  const db = load()
  const p = (db.posts || []).find((x) => x.id === req.params.id)
  if (!p) return res.status(404).json({ error: { code: 'NOT_FOUND', message: 'Post not found' } })

  p.comments = Array.isArray(p.comments) ? p.comments : []
  const item = {
    id: 'c_' + Math.random().toString(36).slice(2, 9),
    username,
    content,
    date: new Date().toISOString(),
  }
  p.comments.push(item)

  save(db)
  res.status(201).json(item)
})

/* ----------------------------- Users ----------------------------- */

// Public user (safe shape)
app.get('/api/users/:username', (req, res) => {
  const db = load()
  const u = db.users.find((x) => x.username === req.params.username)
  if (!u) return res.status(404).json({ error: { code: 'NOT_FOUND', message: 'User not found' } })
  const { password, ...safe } = u
  res.json(safe)
})

// --- Update user (avatar etc.) ---
function isValidUrl(u) {
  try { new URL(u); return true } catch { return false }
}

app.patch('/api/users/:username', (req, res) => {
  const paramUser = String(req.params.username || '');
  const actingUser = req.get('x-user') || paramUser; // simple demo "auth"
  if (!actingUser || actingUser !== paramUser) {
    return res.status(403).json({ error: { code: 'FORBIDDEN', message: 'not your profile' } });
  }

  const { avatarUrl } = req.body || {};
  if (!avatarUrl || !isValidUrl(avatarUrl)) {
    return res.status(400).json({ error: { code: 'BAD_REQUEST', message: 'avatarUrl must be a valid URL' } });
  }

  const db = load();
  const u = db.users?.find(x => x.username === paramUser);
  if (!u) return res.status(404).json({ error: { code: 'NOT_FOUND', message: 'User not found' } });

  u.avatarUrl = avatarUrl;
  u.updatedAt = new Date().toISOString();
  save(db);

  const { password, ...safe } = u;
  res.json(safe);
});

// User's own posts
app.get('/api/users/:username/posts', (req, res) => {
  const db = load()
  const u = db.users.find((x) => x.username === req.params.username)
  if (!u) return res.json([])
  const ids = new Set(Array.isArray(u.posts) ? u.posts : [])
  const items = (db.posts || [])
    .filter((p) => ids.has(p.id) || p.user?.username === u.username)
    .map(normalizePost)
  res.json(items)
})

// Posts the user liked
app.get('/api/users/:username/liked', (req, res) => {
  const db = load()
  const items = (db.posts || [])
    .map(normalizePost)
    .filter((p) => p.likes.includes(req.params.username))
  res.json(items)
})

// Posts the user saved
app.get('/api/users/:username/saved', (req, res) => {
  const db = load()
  const items = (db.posts || [])
    .map(normalizePost)
    .filter((p) => p.saves.includes(req.params.username))
  res.json(items)
})

/* ----------------------------- AI Proxy -------------------------- */

app.post('/api/ai/suggest', async (req, res) => {
  const { imageUrl, prompt } = req.body || {}
  if (!imageUrl) {
    return res.status(400).json({ error: { code: 'BAD_REQUEST', message: 'imageUrl required' } })
  }
  try {
    const AI_BASE = process.env.AI_URL || 'http://localhost:8001'
    const r = await fetch(`${AI_BASE}/ai/suggest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageUrl, prompt: prompt || null }),
    })
    const data = await r.json()
    res.status(r.ok ? 200 : r.status).json(data)
  } catch (e) {
    res.status(502).json({ error: { code: 'AI_UNAVAILABLE', message: 'AI service unreachable' } })
  }
})

app.get('/api/activity/:username', (req, res) => {
    const db = load()
    const me = req.params.username
    const myPosts = (db.posts || []).filter(p => (p.user?.username) === me)
  
    const out = []
    for (const p of myPosts) {
      // likes / unlikes
      for (const e of (p.likeEvents || [])) {
        out.push({
          id: e.id,
          type: e.type === 'unlike' ? 'unlike' : 'like',
          actor: e.username,
          targetUser: me,
          postId: p.id,
          date: e.date,
          meta: null,
        })
      }
      // saves / unsaves
      for (const e of (p.saveEvents || [])) {
        out.push({
          id: e.id,
          type: e.type === 'unsave' ? 'unsave' : 'save',
          actor: e.username,
          targetUser: me,
          postId: p.id,
          date: e.date,
          meta: null,
        })
      }
      // shares
      for (const e of (p.shareEvents || [])) {
        out.push({
          id: e.id,
          type: 'share',
          actor: e.from,
          targetUser: me,
          postId: p.id,
          date: e.date,
          meta: {
            to: e.to ? (Array.isArray(e.to) ? e.to : [e.to]).filter(Boolean) : [],
            method: e.method || 'dm',
          },
        })
      }
      // comments
      for (const c of (p.comments || [])) {
        out.push({
          id: c.id,
          type: 'comment',
          actor: c.username,
          targetUser: me,
          postId: p.id,
          date: c.date,
          meta: { content: c.content },
        })
      }
    }
  
    out.sort((a, b) => new Date(b.date) - new Date(a.date))
    res.json(out)
  })

  app.get('/api/activity', (req, res) => {
    const username = req.query.username
    if (!username) {
      return res.status(400).json({ error: { code: 'BAD_REQUEST', message: 'username required' } })
    }
    // delegate to the param route
    req.params.username = username
    app._router.handle(req, res, () => {}) // reuse handler above
  })
  

/* ----------------------------- Server ---------------------------- */

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`API listening on ${PORT}`));

