// SAGA app shell: network-first so updates show up, cache as the offline fallback.
// API calls and photos are never cached.
const CACHE = "saga-m-v9";
const SHELL = ["/m/", "/m/manifest.webmanifest", "/m/icons/apple-touch-icon.png", "/m/icons/icon-192.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin || !url.pathname.startsWith("/m/")) return;
  e.respondWith(fetch(e.request).then((res) => {
    if (res.redirected || !res.ok) return res; // e.g. sent to the login gate: never cache that
    const copy = res.clone();
    caches.open(CACHE).then((c) => c.put(e.request, copy));
    return res;
  }).catch(() => caches.match(e.request).then((hit) => hit || caches.match("/m/"))));
});
