// /* Open-Meteo overlay for Mapbox GL
//    - Radar-style precip circles (no heatmap fog)
//    - Bigger dots with zoom scaling
//    - Optional darken-basemap layer (under labels, over basemap)
//    - Time slider + play/pause
//    - Self-contained: does not touch your existing code
// */
// (function () {
//   if (window.__openMeteoRadarOverlay) return;
//   window.__openMeteoRadarOverlay = true;

//   // ---------- utils ----------
//   const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
//   const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
//   const fmtHour = (iso) => {
//     try {
//       const d = new Date(iso);
//       return d.toLocaleString(undefined, {
//         year: 'numeric',
//         month: 'short',
//         day: '2-digit',
//         hour: '2-digit',
//       });
//     } catch {
//       return iso;
//     }
//   };

//   // ---------- styles ----------
//   (function injectCSS() {
//     const css = `
//     .omctl{font:13px/1.35 system-ui,-apple-system,Segoe UI,Roboto,Arial;color:#0f172a}
//     .omctl .panel{background:#fff;border-radius:12px;box-shadow:0 10px 28px rgba(0,0,0,.16);padding:10px 12px;min-width:230px;border:1px solid #e5e7eb}
//     .omctl .row{display:flex;align-items:center;justify-content:space-between;gap:10px}
//     .omctl .tog{appearance:none;width:38px;height:22px;border-radius:999px;position:relative;background:#e5e7eb;outline:none;cursor:pointer;transition:background .2s}
//     .omctl .tog:checked{background:#3b82f6}
//     .omctl .tog:before{content:"";position:absolute;top:3px;left:3px;width:16px;height:16px;border-radius:999px;background:#fff;transition:left .2s}
//     .omctl .tog:checked:before{left:19px}
//     .omctl .legend{margin-top:10px}
//     .omctl .bar{height:10px;border-radius:6px;background:linear-gradient(90deg, rgba(0,0,0,0), #b3e5fc, #4fc3f7, #22c55e, #facc15, #fb923c, #ef4444)}
//     .omctl .scale{display:flex;justify-content:space-between;color:#475569;margin-top:4px}

//     .omtime.mapboxgl-ctrl{padding:8px}
//     .omtime .box{background:#fff;border:1px solid #e5e7eb;border-radius:12px;box-shadow:0 10px 28px rgba(0,0,0,.16);padding:10px 12px;min-width:300px}
//     .omtime .top{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:8px}
//     .omtime .label{font-weight:600;color:#111827}
//     .omtime .play{border:0;background:#2563eb;color:#fff;padding:6px 10px;border-radius:8px;cursor:pointer}
//     .omtime input[type=range]{width:100%
//      .mapboxgl-ctrl-bottom-right .omctl { margin: 0 8px 8px 0; }
//     .mapboxgl-ctrl-bottom-right .omtime { margin: 0 8px 8px 0; }}
//     `;
//     const s = document.createElement('style');
//     s.textContent = css;
//     document.head.appendChild(s);
//   })();

//   // ---------- get Mapbox map ----------
//   async function getMapInstance() {
//     for (let i = 0; i < 200; i++) {
//       const map =
//         window.__legacyMapInstance ||
//         window.map ||
//         (window.mapboxgl && window.mapboxgl.__mapInstance);
//       if (map && typeof map.getBounds === 'function') return map;
//       await sleep(100);
//     }
//     return null;
//   }

//   // ---------- grid of sample points over current view ----------
//   function buildGridCoords(map) {
//     const b = map.getBounds();
//     const sw = b.getSouthWest(),
//       ne = b.getNorthEast();
//     const west = sw.lng,
//       south = sw.lat,
//       east = ne.lng,
//       north = ne.lat;

//     const z = map.getZoom();
//     let cols = 18,
//       rows = 12;
//     if (z < 5) {
//       cols = 8;
//       rows = 6;
//     } else if (z < 7) {
//       cols = 12;
//       rows = 8;
//     } else if (z > 9) {
//       cols = 24;
//       rows = 16;
//     }
//     cols = clamp(cols, 4, 30);
//     rows = clamp(rows, 4, 30);

//     const lats = [],
//       lons = [];
//     for (let r = 0; r < rows; r++) {
//       const lat = south + ((r + 0.5) * (north - south)) / rows;
//       lats.push(lat.toFixed(5));
//     }
//     for (let c = 0; c < cols; c++) {
//       const lon = west + ((c + 0.5) * (east - west)) / cols;
//       lons.push(lon.toFixed(5));
//     }
//     const latList = [],
//       lonList = [];
//     for (const la of lats)
//       for (const lo of lons) {
//         latList.push(la);
//         lonList.push(lo);
//       }
//     return { latList, lonList };
//   }

//   // ---------- Open-Meteo fetch (multi-coords) ----------
//   async function fetchOpenMeteo(latList, lonList) {
//     const params = new URLSearchParams({
//       latitude: latList.join(','),
//       longitude: lonList.join(','),
//       hourly:
//         'precipitation,temperature_2m,relative_humidity_2m,wind_speed_10m',
//       timezone: 'auto',
//       forecast_days: '2',
//     });
//     const res = await fetch(`https://api.open-meteo.com/v1/forecast?${params}`);
//     if (!res.ok) throw new Error('Open-Meteo request failed');
//     return res.json();
//   }

//   // ---------- normalize payload ----------
//   function buildState(payload, latList, lonList) {
//     const arr = Array.isArray(payload)
//       ? payload
//       : payload.results
//       ? payload.results
//       : [payload];
//     const points = [];
//     let times = null;

//     for (let i = 0; i < arr.length; i++) {
//       const r = arr[i];
//       if (!times) times = r.hourly.time;
//       points.push({
//         lon: parseFloat(lonList[i]),
//         lat: parseFloat(latList[i]),
//         series: {
//           p: r.hourly.precipitation, // mm
//           tc: r.hourly.temperature_2m, // °C
//           rh: r.hourly.relative_humidity_2m, // %
//           ws: r.hourly.wind_speed_10m, // m/s
//         },
//       });
//     }
//     // hour closest to now
//     const now = Date.now();
//     let hourIndex = 0,
//       best = Infinity;
//     times.forEach((iso, i) => {
//       const d = Math.abs(new Date(iso).getTime() - now);
//       if (d < best) {
//         best = d;
//         hourIndex = i;
//       }
//     });
//     return { points, times, hourIndex };
//   }

//   // ---------- GeoJSON for given hour ----------
//   function geojsonForHour(state, idx) {
//     return {
//       type: 'FeatureCollection',
//       features: state.points.map((pt) => ({
//         type: 'Feature',
//         geometry: { type: 'Point', coordinates: [pt.lon, pt.lat] },
//         properties: {
//           precipitation: pt.series.p[idx] ?? 0,
//           temperature: pt.series.tc[idx] ?? null,
//           humidity: pt.series.rh[idx] ?? null,
//           wind: pt.series.ws[idx] ?? null,
//         },
//       })),
//     };
//   }

//   // ---------- helper: insert darken layer (over basemap, under labels) ----------
//   function ensureDarkener(map, beforeId) {
//     if (!map.getSource('om-world')) {
//       map.addSource('om-world', {
//         type: 'geojson',
//         data: {
//           type: 'FeatureCollection',
//           features: [
//             {
//               type: 'Feature',
//               geometry: {
//                 type: 'Polygon',
//                 coordinates: [
//                   [
//                     [-179, -85],
//                     [179, -85],
//                     [179, 85],
//                     [-179, 85],
//                     [-179, -85],
//                   ],
//                 ],
//               },
//             },
//           ],
//         },
//       });
//     }
//     if (!map.getLayer('om-darken')) {
//       map.addLayer(
//         {
//           id: 'om-darken',
//           type: 'fill',
//           source: 'om-world',
//           paint: { 'fill-color': 'rgba(10,15,20,0.55)' }, // darken base map
//         },
//         beforeId
//       ); // below labels
//     }
//   }

//   // ---------- layers ----------
//   function ensureLayers(map) {
//     if (map.getSource('openmeteo')) return;

//     map.addSource('openmeteo', {
//       type: 'geojson',
//       data: { type: 'FeatureCollection', features: [] },
//     });

//     const layers = map.getStyle().layers || [];
//     const firstSymbolId = (
//       layers.find(
//         (l) => l.type === 'symbol' && l.layout && l.layout['text-field']
//       ) || {}
//     ).id;

//     // darken base map below labels but above basemap
//     ensureDarkener(map, firstSymbolId);

//     // RADAR-STYLE PRECIP CIRCLES (no fog)
//     const radiusByZoom = [
//       'interpolate',
//       ['linear'],
//       ['zoom'],
//       3,
//       4, // low zoom
//       5,
//       6,
//       7,
//       9,
//       9,
//       12,
//       11,
//       16,
//       13,
//       20,
//     ];
//     map.addLayer(
//       {
//         id: 'om-precip-circles',
//         type: 'circle',
//         source: 'openmeteo',
//         filter: ['>', ['get', 'precipitation'], 0.01],
//         paint: {
//           'circle-radius': radiusByZoom,
//           'circle-blur': 0.35, // soft speckle
//           'circle-color': [
//             'interpolate',
//             ['linear'],
//             ['get', 'precipitation'],
//             0.0,
//             'rgba(0,0,0,0)',
//             0.1,
//             '#b3e5fc', // light cyan
//             0.5,
//             '#4fc3f7', // blue-cyan
//             1.5,
//             '#22c55e', // green
//             3.0,
//             '#facc15', // yellow
//             6.0,
//             '#fb923c', // orange
//             12.0,
//             '#ef4444', // red
//           ],
//           'circle-opacity': 0.9,
//         },
//       },
//       firstSymbolId
//     );

//     // Bigger data dots (start hidden; user toggles)
//     const dotRadius = [
//       'interpolate',
//       ['linear'],
//       ['zoom'],
//       3,
//       4,
//       5,
//       6,
//       7,
//       8,
//       9,
//       10,
//       11,
//       12,
//       13,
//       14,
//     ];

//     map.addLayer(
//       {
//         id: 'om-temp-dots',
//         type: 'circle',
//         source: 'openmeteo',
//         paint: {
//           'circle-radius': dotRadius,
//           'circle-color': [
//             'interpolate',
//             ['linear'],
//             ['get', 'temperature'],
//             -10,
//             '#4575b4',
//             0,
//             '#91bfdb',
//             10,
//             '#fee090',
//             20,
//             '#fc8d59',
//             30,
//             '#d73027',
//           ],
//           'circle-opacity': 0.95,
//           'circle-stroke-color': '#ffffff',
//           'circle-stroke-width': 1.2,
//         },
//         layout: { visibility: 'none' },
//       },
//       firstSymbolId
//     );

//     map.addLayer(
//       {
//         id: 'om-humidity-dots',
//         type: 'circle',
//         source: 'openmeteo',
//         paint: {
//           'circle-radius': dotRadius,
//           'circle-color': [
//             'interpolate',
//             ['linear'],
//             ['get', 'humidity'],
//             0,
//             '#fdf2f8',
//             30,
//             '#fbcfe8',
//             60,
//             '#a78bfa',
//             80,
//             '#6366f1',
//             100,
//             '#312e81',
//           ],
//           'circle-opacity': 0.95,
//           'circle-stroke-color': '#ffffff',
//           'circle-stroke-width': 1.2,
//         },
//         layout: { visibility: 'none' },
//       },
//       firstSymbolId
//     );

//     map.addLayer(
//       {
//         id: 'om-wind-dots',
//         type: 'circle',
//         source: 'openmeteo',
//         paint: {
//           'circle-radius': dotRadius,
//           'circle-color': [
//             'interpolate',
//             ['linear'],
//             ['get', 'wind'],
//             0,
//             '#e0f2fe',
//             3,
//             '#93c5fd',
//             6,
//             '#60a5fa',
//             10,
//             '#3b82f6',
//             15,
//             '#1d4ed8',
//           ],
//           'circle-opacity': 0.95,
//           'circle-stroke-color': '#ffffff',
//           'circle-stroke-width': 1.2,
//         },
//         layout: { visibility: 'none' },
//       },
//       firstSymbolId
//     );
//   }

//   function updateSource(map, geojson) {
//     const src = map.getSource('openmeteo');
//     src && src.setData(geojson);
//   }

//   // ---------- controls ----------
//   function addControls(map, state) {
//     // Layer toggles + darkener + legend
//     if (!document.querySelector('.omctl')) {
//       const el = document.createElement('div');
//       el.className = 'mapboxgl-ctrl omctl';
//       el.innerHTML = `
//         <div class="panel">
//           <div class="row"><div>Precipitation</div><input class="tog" id="om_precip" type="checkbox" checked></div>
//           <div class="row" style="margin-top:6px"><div>Temperature</div><input class="tog" id="om_temp" type="checkbox"></div>
//           <div class="row" style="margin-top:6px"><div>Humidity</div><input class="tog" id="om_hum" type="checkbox"></div>
//           <div class="row" style="margin-top:6px"><div>Wind</div><input class="tog" id="om_wind" type="checkbox"></div>
//           <div class="row" style="margin-top:6px"><div>Darken basemap</div><input class="tog" id="om_dark" type="checkbox" checked></div>
//           <div class="legend">
//             <div class="bar"></div>
//             <div class="scale"><span>Dry</span><span>Wet</span></div>
//           </div>
//         </div>`;
//       const ctrl = {
//         onAdd() {
//           return el;
//         },
//         onRemove() {
//           el.remove();
//         },
//       };
//       map.addControl(ctrl, 'top-right');

//       const setVis = (id, on) =>
//         map.setLayoutProperty(id, 'visibility', on ? 'visible' : 'none');
//       el.querySelector('#om_precip').addEventListener('change', (e) =>
//         setVis('om-precip-circles', e.target.checked)
//       );
//       el.querySelector('#om_temp').addEventListener('change', (e) =>
//         setVis('om-temp-dots', e.target.checked)
//       );
//       el.querySelector('#om_hum').addEventListener('change', (e) =>
//         setVis('om-humidity-dots', e.target.checked)
//       );
//       el.querySelector('#om_wind').addEventListener('change', (e) =>
//         setVis('om-wind-dots', e.target.checked)
//       );
//       el.querySelector('#om_dark').addEventListener('change', (e) =>
//         map.setLayoutProperty(
//           'om-darken',
//           'visibility',
//           e.target.checked ? 'visible' : 'none'
//         )
//       );
//     }

//     // Time slider
//     if (!document.querySelector('.omtime')) {
//       const timeCtrl = document.createElement('div');
//       timeCtrl.className = 'mapboxgl-ctrl omtime';
//       timeCtrl.innerHTML = `
//         <div class="box">
//           <div class="top">
//             <div class="label" id="om_time_label"></div>
//             <button class="play" id="om_play">▶</button>
//           </div>
//           <input id="om_slider" type="range" min="0" value="0" step="1" />
//         </div>`;
//       const ctrl = {
//         onAdd() {
//           return timeCtrl;
//         },
//         onRemove() {
//           timeCtrl.remove();
//         },
//       };
//       map.addControl(ctrl, 'bottom-right');

//       const label = timeCtrl.querySelector('#om_time_label');
//       const slider = timeCtrl.querySelector('#om_slider');
//       const btn = timeCtrl.querySelector('#om_play');

//       slider.max = String(state.times.length - 1);
//       slider.value = String(state.hourIndex);
//       label.textContent = fmtHour(state.times[state.hourIndex]);

//       let playing = false,
//         handle = null;
//       const applyHour = (idx) => {
//         state.hourIndex = idx;
//         label.textContent = fmtHour(state.times[idx]);
//         updateSource(map, geojsonForHour(state, idx));
//       };

//       slider.addEventListener('input', (e) =>
//         applyHour(parseInt(e.target.value, 10))
//       );

//       btn.addEventListener('click', () => {
//         playing = !playing;
//         btn.textContent = playing ? '❚❚' : '▶';
//         if (playing) {
//           handle = setInterval(() => {
//             let n = state.hourIndex + 1;
//             if (n > state.times.length - 1) n = 0;
//             slider.value = String(n);
//             applyHour(n);
//           }, 1200);
//         } else {
//           clearInterval(handle);
//         }
//       });
//     }
//   }

//   // ---------- refresh ----------
//   let inflight = false,
//     queued = false,
//     state = null;

//   async function refresh(map) {
//     if (inflight) {
//       queued = true;
//       return;
//     }
//     inflight = true;
//     try {
//       const { latList, lonList } = buildGridCoords(map);
//       const payload = await fetchOpenMeteo(latList, lonList);
//       state = buildState(payload, latList, lonList);

//       ensureLayers(map);
//       addControls(map, state);

//       updateSource(map, geojsonForHour(state, state.hourIndex));
//     } catch (e) {
//       // silent
//     } finally {
//       inflight = false;
//       if (queued) {
//         queued = false;
//         refresh(map);
//       }
//     }
//   }

//   // ---------- init ----------
//   (async function init() {
//     const map = await getMapInstance();
//     if (!map) return;
//     if (window.mapboxgl && !window.mapboxgl.__mapInstance)
//       window.mapboxgl.__mapInstance = map;

//     const start = () => refresh(map);
//     map.loaded() ? start() : map.once('load', start);

//     let timer = null;
//     map.on('moveend', () => {
//       clearTimeout(timer);
//       timer = setTimeout(() => refresh(map), 250);
//     });
//   })();
// })();
