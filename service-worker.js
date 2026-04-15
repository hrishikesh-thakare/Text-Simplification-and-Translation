const CACHE_NAME = "text-simplifier-v1";
const ASSETS_TO_CACHE = ["/", "/index.html", "/manifest.json"];

// Install: cache core assets
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log("Service Worker: caching core assets");
      return cache.addAll(ASSETS_TO_CACHE).catch(() => {
        console.log(
          "Service Worker: some assets not cached (expected for first load)",
        );
      });
    }),
  );
  self.skipWaiting();
});

// Activate: clean up old caches
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log("Service Worker: deleting old cache", cacheName);
            return caches.delete(cacheName);
          }
        }),
      );
    }),
  );
  self.clients.claim();
});

// Fetch: network-first for API calls, cache-first for assets
self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // API calls: network-first with offline fallback
  if (request.method === "POST" || url.pathname.includes("api")) {
    event.respondWith(
      fetch(request)
        .then((response) => {
          if (response.ok) {
            const cache = caches.open(CACHE_NAME);
            cache.then((c) => c.put(request, response.clone()));
          }
          return response;
        })
        .catch(() => {
          return caches.match(request).then((cachedResponse) => {
            if (cachedResponse) {
              return cachedResponse;
            }
            return new Response(
              JSON.stringify({
                error: "Offline. Please check your connection.",
              }),
              {
                status: 503,
                headers: { "Content-Type": "application/json" },
              },
            );
          });
        }),
    );
  } else {
    // Static assets: cache-first
    event.respondWith(
      caches.match(request).then((cachedResponse) => {
        if (cachedResponse) {
          return cachedResponse;
        }
        return fetch(request)
          .then((response) => {
            if (
              !response ||
              response.status !== 200 ||
              response.type === "error"
            ) {
              return response;
            }
            const responseClone = response.clone();
            caches.open(CACHE_NAME).then((cache) => {
              cache.put(request, responseClone);
            });
            return response;
          })
          .catch(() => {
            return new Response("Offline - asset not available", {
              status: 503,
              headers: { "Content-Type": "text/plain" },
            });
          });
      }),
    );
  }
});
