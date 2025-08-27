<!-- src/App.vue -->
<template>
  <div class="min-h-screen bg-black text-white overflow-x-hidden">
    <Navbar />
    <div class="pt-16 md:pt-16 pb-24 md:pb-0">
      <router-view />
    </div>
  </div>
</template>

<script setup>
import Navbar from '@/components/Navbar.vue'
</script>

<style>
/* optional utility if some pages need extra bottom padding for the mobile bar */
.page-has-bottom-nav {
  padding-bottom: calc(84px + env(safe-area-inset-bottom));
}
</style>
