// ---- Globals ----
const mapboxgl = window.mapboxgl;

mapboxgl.accessToken =
  'pk.eyJ1IjoiaXNhYWtoYW4iLCJhIjoiY21iejFzMjUzMWlsczJqcXcwa2N4NHdtZSJ9.M7oGg-7sSbbNc0rl3596Jg';

let map,
  currentProvince = 'all';

function toggleSidebar() {
  const el = document.getElementById('container');
  if (el) el.classList.toggle('expanded');
}

// --- Entry point ---
function initializeMap(styleName = 'satellite-streets-v12') {
  map = new mapboxgl.Map({
    container: 'map',
    style: `mapbox://styles/mapbox/${styleName}`,
    center: [30, 39],
    zoom: 6,
    pitch: 45,
    bearing: -17.6,
    antialias: true,
    projection: 'globe',
  });

  map.on('load', () => {
    addLayersAndSources();
    initStyleSwitcher();
  });

  map.once('style.load', initAtmosControls);

  const origSetStyle = map.setStyle.bind(map);
  map.setStyle = function (...args) {
    const ret = origSetStyle(...args);
    this.once('style.load', () => {
      addLayersAndSources();
      rehydrateAtmosphereIfNeeded();
      initAtmosControls();
    });
    return ret;
  };
}

function initStyleSwitcher() {
  const styleSel = document.getElementById('layer-style');
  if (!styleSel) return;

  if (styleSel._wired) return;
  styleSel._wired = true;

  styleSel.addEventListener('change', (e) => {
    const styleName = e.target.value;
    const center = map.getCenter();
    const zoom = map.getZoom();
    const pitch = map.getPitch();
    const bearing = map.getBearing();

    console.log(`Switching map style to: ${styleName}`);

    map.setStyle(`mapbox://styles/mapbox/${styleName}`);

    map.once('style.load', () => {
      addLayersAndSources();
      rehydrateAtmosphereIfNeeded();
      initAtmosControls();
      map.flyTo({ center, zoom, pitch, bearing });
    });
  });
}

// ---------------- Sources & Layers ----------------
function addLayersAndSources() {
  // DEM
  if (!map.getSource('mapbox-dem')) {
    map.addSource('mapbox-dem', {
      type: 'raster-dem',
      url: 'mapbox://mapbox.terrain-rgb',
      tileSize: 512,
      maxzoom: 14,
    });
    map.setTerrain({ source: 'mapbox-dem', exaggeration: 1.5 });
  }

  // Wind farms
  {
    const WIND_ICON_DATA =
      'data:image/svg+xml;utf8,' +
      encodeURIComponent(`
        <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24">
          <g fill="#0b7">
            <circle cx="12" cy="12" r="2"/>
            <path d="M12 2l1.2 7.5L20 9l-6.5 3.2L15 20l-3-5-3 5 1.5-7.8L4 9l6.8-.5L12 2z"/>
          </g>
        </svg>`);

    const addWindIcon = (img) => {
      if (!map.hasImage('windmill-icon'))
        map.addImage('windmill-icon', img, { pixelRatio: 2 });
      if (!map.getSource('windmills')) {
        map.addSource('windmills', {
          type: 'geojson',
          data: 'windmills_turkey.geojson',
        });
        map.addLayer({
          id: 'windmill-points',
          type: 'symbol',
          source: 'windmills',
          layout: {
            'icon-image': 'windmill-icon',
            'icon-size': 0.07,
            'icon-allow-overlap': true,
            visibility: 'visible',
          },
        });
        bindClicks('windmill-points', 'wind');
      }
    };

    map.loadImage('/wind-power.png', (e, img) => {
      if (e || !img)
        map.loadImage(WIND_ICON_DATA, (e2, img2) => addWindIcon(img2));
      else addWindIcon(img);
    });
  }

  // Solar farms
  {
    const SOLAR_ICON_DATA =
      'data:image/svg+xml;utf8,' +
      encodeURIComponent(`
        <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24">
          <g fill="#e6a400">
            <rect x="4" y="10" width="16" height="6" rx="1.2"/>
            <path d="M6 18h12l-1 2H7z"/>
            <circle cx="18.5" cy="5.5" r="3.5"/>
          </g>
        </svg>`);

    const addSolarIcon = (img) => {
      if (!map.hasImage('solar-icon'))
        map.addImage('solar-icon', img, { pixelRatio: 2 });
      if (!map.getSource('solarfarms')) {
        map.addSource('solarfarms', {
          type: 'geojson',
          data: 'solar.geojson',
        });
        map.addLayer({
          id: 'solar-points',
          type: 'symbol',
          source: 'solarfarms',
          layout: {
            'icon-image': 'solar-icon',
            'icon-size': 0.07,
            'icon-allow-overlap': true,
            visibility: 'visible',
          },
        });
        bindClicks('solar-points', 'solar');
      }
    };

    map.loadImage('/solar.png', (e, img) => {
      if (e || !img)
        map.loadImage(SOLAR_ICON_DATA, (e2, img2) => addSolarIcon(img2));
      else addSolarIcon(img);
    });
  }

  // Provinces
  if (!map.getSource('provinces')) {
    map.addSource('provinces', { type: 'geojson', data: 'tr-cities.json' });
    map.addLayer({
      id: 'province-fill',
      type: 'fill',
      source: 'provinces',
      paint: { 'fill-color': '#088', 'fill-opacity': 0 },
    });
    map.addLayer({
      id: 'province-outline',
      type: 'line',
      source: 'provinces',
      paint: { 'line-color': '#000', 'line-width': 1 },
    });
  }

  populateProvinces();
  applyProvinceFilter();
  applyDataTypeFilter();
}

// ---------------- Popup binding ----------------
function bindClicks(layerId, type) {
  map.on('click', layerId, (e) => {
    const f = e.features && e.features[0];
    if (!f) return;

    const c = f.geometry?.coordinates
      ? f.geometry.coordinates.slice()
      : [e.lngLat.lng, e.lngLat.lat];
    const p = f.properties || {};
    const esc = (v) =>
      String(v ?? '').replace(
        /[&<>"']/g,
        (s) =>
          ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;',
          }[s])
      );

    const lat = p.Latitude || p.lat || c[1] || e.lngLat.lat;
    const lon = p.Longitude || p.lon || p.lng || c[0] || e.lngLat.lng;

    let html = '';
    if (type === 'wind') {
      html = `
        <strong>Name:</strong> ${esc(p.Name)}<br>
        <strong>Status:</strong> ${esc(p.Status)}<br>
        <strong>Year:</strong> ${esc(p.Year)}<br>
        <strong>Province:</strong> ${esc(p.Province)}<br>
        <button id="btn-weather">View Weather</button>
        <button id="btn-report">Download Report</button>
      `;
    } else {
      html = `
        <strong>Project:</strong> ${esc(p.Project)}<br>
        <strong>Operator:</strong> ${esc(p.Operator)}<br>
        <strong>Method:</strong> ${esc(p.Method)}<br>
        <strong>Capacity:</strong> ${esc(p.Capacity)}<br>
        <strong>Province:</strong> ${esc(p.Province)}<br>
        <strong>Plant Source:</strong> ${esc(p['plant:source'])}<br>
        <button id="btn-weather">View Weather</button>
        <button id="btn-report">Download Report</button>
      `;
    }

    new mapboxgl.Popup().setLngLat(c).setHTML(html).addTo(map);

    setTimeout(() => {
      const btnWeather = document.getElementById('btn-weather');
      const btnReport = document.getElementById('btn-report');
      if (btnWeather)
        btnWeather.addEventListener('click', () => openWeatherModal(lat, lon));
      if (btnReport) {
        if (type === 'wind')
          btnReport.addEventListener('click', () =>
            downloadWindReport(lat, lon, p)
          );
        else
          btnReport.addEventListener('click', () =>
            downloadSolarReport(lat, lon, p)
          );
      }
    }, 100);

    map.flyTo({ center: c, zoom: 17 });
  });
}

function downloadWindReport(lat, lon, props = {}) {
  const qs = new URLSearchParams({
    lat: lat.toFixed(5),
    lon: lon.toFixed(5),
    days: 7,
    site_type: 'wind',
    rated_power_mw: props['Rated Power'] || props['Capacity'] || 3.6,
    name: props['Name'] || '',
    status: props['Status'] || '',
    year: props['Year'] || props['Start_year'] || '',
    province: props['Province'] || '',
  }).toString();
  window.open(`http://localhost:8000/report.pdf?${qs}`, '_blank');
}

function downloadSolarReport(lat, lon, props = {}) {
  const qs = new URLSearchParams({
    lat: lat.toFixed(5),
    lon: lon.toFixed(5),
    days: 7,
    site_type: 'solar',
    capacity: props['Capacity'] || '',
    project: props['Project'] || props['Name'] || '',
    operator: props['Operator'] || '',
    province: props['Province'] || '',
    method: props['Method'] || '',
    plant_source: props['plant:source'] || '',
    power: props['Power'] || '',
  }).toString();
  window.open(`http://localhost:8000/report.pdf?${qs}`, '_blank');
}
function openModal(type) {
  const M = document.getElementById('graphModal'),
    F = document.getElementById('graphFrame'),
    T = document.getElementById('modalTitle'),
    C = M.querySelector('.modal-content'),
    H = M.querySelector('.modal-header');
  let url, w, h;
  if (type === 'scatter') {
    T.textContent = 'Static Scatter';
    url = 'http://localhost:8000/scatter';
    w = 600;
    h = 600;
  } else if (type === 'bar') {
    T.textContent = 'Static Histogram';
    url = 'http://localhost:8000/bar';
    w = 600;
    h = 500;
  } else {
    T.textContent = 'Interactive Scatter';
    url = 'http://localhost:8000/scatter_int';
    w = window.innerWidth * 0.8;
    h = window.innerHeight * 0.8;
  }
  F.src = url;
  F.width = w;
  F.height = h;
  const hh = H.getBoundingClientRect().height;
  C.style.width = `${w}px`;
  C.style.height = `${hh + h}px`;
  M.style.display = 'flex';
}
function closeModal() {
  const M = document.getElementById('graphModal');
  const F = document.getElementById('graphFrame');
  if (M) M.style.display = 'none';
  if (F) F.src = '';
}
function openInfoModal() {
  const M = document.getElementById('infoModal');
  if (M) M.style.display = 'flex';
}
function closeInfoModal() {
  const M = document.getElementById('infoModal');
  if (M) M.style.display = 'none';
}
// ---------------- Province filter ----------------
function populateProvinces() {
  if (populateProvinces.done) return;
  populateProvinces.done = true;
  Promise.all([
    fetch('windmills_turkey.geojson').then((r) => r.json()),
    fetch('solar.geojson').then((r) => r.json()),
  ]).then(([W, S]) => {
    const set = new Set();
    [...W.features, ...S.features].forEach((f) => {
      const props = f.properties || {};
      const p = props.Province || props.province;
      if (p) set.add(p);
    });
    const sel = document.getElementById('province-select');
    if (!sel) return;
    Array.from(set)
      .sort()
      .forEach((p) => sel.appendChild(new Option(p, p)));
    sel.addEventListener('change', (e) => {
      currentProvince = e.target.value;
      applyProvinceFilter();
    });
  });
}

function applyProvinceFilter() {
  const f =
    currentProvince === 'all'
      ? null
      : ['==', ['get', 'Province'], currentProvince];
  if (map.getLayer('windmill-points')) map.setFilter('windmill-points', f);
  if (map.getLayer('solar-points')) map.setFilter('solar-points', f);
}

// ---------------- Weather Modal ----------------
function openWeatherModal(lat, lon) {
  const M = document.getElementById('weatherModal'),
    F = document.getElementById('weatherFrame');
  if (F) F.src = `/weather.html?lat=${lat}&lon=${lon}`;
  if (M) M.style.display = 'flex';
}
function closeWeatherModal() {
  const M = document.getElementById('weatherModal'),
    F = document.getElementById('weatherFrame');
  if (M) M.style.display = 'none';
  if (F) F.src = '';
}

// ---------------- Atmosphere Add-On ----------------
const TEMP_SOURCE_ID = 'gfs-temperature';
const TEMP_LAYER_ID = 'gfs-temperature-layer';
const WIND_SOURCE_ID = 'raster-array-source';
const WIND_LAYER_ID = 'wind-layer';
const RAIN_SOURCE_ID = 'rainviewer';
const RAIN_LAYER_ID = 'rainviewer-layer';

let _rainFrames = [],
  _rainIndex = 0,
  _rainTimer = null,
  _rainSpeed = 400;

// --- Temperature layer ---
function addTemperatureLayer() {
  if (!map.getSource(TEMP_SOURCE_ID)) {
    map.addSource(TEMP_SOURCE_ID, {
      type: 'raster-array',
      url: 'mapbox://mapbox.gfs-temperature',
      tileSize: 256,
    });
  }
  if (!map.getLayer(TEMP_LAYER_ID)) {
    const under = map.getStyle().layers.find((l) => /-label$/.test(l.id))?.id;
    map.addLayer(
      {
        id: TEMP_LAYER_ID,
        type: 'raster',
        source: TEMP_SOURCE_ID,
        'source-layer': '2t',
        paint: {
          'raster-color-range': [253, 313],
          'raster-array-band': '3',
          'raster-color': [
            'interpolate',
            ['linear'],
            ['raster-value'],
            263,
            '#313695',
            268,
            '#4575b4',
            273,
            '#74add1',
            278,
            '#abd9e9',
            283,
            '#e0f3f8',
            288,
            '#ffffbf',
            293,
            '#fee090',
            298,
            '#fdae61',
            303,
            '#f46d43',
            308,
            '#d73027',
            313,
            '#a50026',
            318,
            '#67001f',
            323,
            '#40001a',
          ],
          'raster-resampling': 'nearest',
          'raster-opacity': 0.75,
        },
      },
      under
    );
  }
}
function removeTemperatureLayer() {
  if (map.getLayer(TEMP_LAYER_ID)) map.removeLayer(TEMP_LAYER_ID);
  if (map.getSource(TEMP_SOURCE_ID)) map.removeSource(TEMP_SOURCE_ID);
}

// --- Wind layer ---
function addWindLayer() {
  if (!map.getSource(WIND_SOURCE_ID)) {
    map.addSource(WIND_SOURCE_ID, {
      type: 'raster-array',
      url: 'mapbox://mapbox.gfs-winds',
      tileSize: 512,
    });
  }
  if (!map.getLayer(WIND_LAYER_ID)) {
    map.addLayer({
      id: WIND_LAYER_ID,
      type: 'raster-particle',
      source: WIND_SOURCE_ID,
      'source-layer': '10winds',
      paint: {
        'raster-particle-speed-factor': 0.4,
        'raster-particle-fade-opacity-factor': 0.9,
        'raster-particle-reset-rate-factor': 0.4,
        'raster-particle-count': 4000,
        'raster-particle-max-speed': 40,
      },
    });
  }
}
function removeWindLayer() {
  if (map.getLayer(WIND_LAYER_ID)) map.removeLayer(WIND_LAYER_ID);
  if (map.getSource(WIND_SOURCE_ID)) map.removeSource(WIND_SOURCE_ID);
}

// --- Rain Radar ---
async function _fetchRainviewerTimeline() {
  const res = await fetch(
    'https://api.rainviewer.com/public/weather-maps.json'
  );
  if (!res.ok) throw new Error('RainViewer fetch failed');
  const json = await res.json();
  const frames = [...(json.radar?.past || [])];
  if (json.radar?.nowcast?.length) frames.push(...json.radar.nowcast);
  return frames;
}
function _setRainFrame(frame) {
  if (!frame) return;
  const tileUrl = `https://tilecache.rainviewer.com${frame.path}/256/{z}/{x}/{y}/2/1_1.png`;
  if (!map.getSource(RAIN_SOURCE_ID)) {
    map.addSource(RAIN_SOURCE_ID, {
      type: 'raster',
      tiles: [tileUrl],
      tileSize: 256,
    });
  } else {
    map.getSource(RAIN_SOURCE_ID).tiles = [tileUrl];
    map.getSource(RAIN_SOURCE_ID).setTiles([tileUrl]);
  }
  if (!map.getLayer(RAIN_LAYER_ID)) {
    map.addLayer(
      {
        id: RAIN_LAYER_ID,
        type: 'raster',
        source: RAIN_SOURCE_ID,
        paint: { 'raster-opacity': 0.7 },
      },
      map.getStyle().layers.find((l) => /-label$/.test(l.id))?.id
    );
  }
  _updateRainTimestamp(frame.time);
}
async function addRainLayer() {
  try {
    _rainFrames = await _fetchRainviewerTimeline();
    if (!_rainFrames.length) return;
    _rainIndex = _rainFrames.length - 1;
    _setRainFrame(_rainFrames[_rainIndex]);
    startRainAnimation();
  } catch (e) {
    console.error(e);
  }
}
function startRainAnimation() {
  stopRainLayer();
  _rainTimer = setInterval(() => {
    _rainIndex = (_rainIndex + 1) % _rainFrames.length;
    _setRainFrame(_rainFrames[_rainIndex]);
  }, _rainSpeed);
}
function stopRainLayer() {
  if (_rainTimer) {
    clearInterval(_rainTimer);
    _rainTimer = null;
  }
}
function removeRainLayer() {
  stopRainLayer();
  if (map.getLayer(RAIN_LAYER_ID)) map.removeLayer(RAIN_LAYER_ID);
  if (map.getSource(RAIN_SOURCE_ID)) map.removeSource(RAIN_SOURCE_ID);
}
function _updateRainTimestamp(ts) {
  let el = document.getElementById('rain-ts');
  if (!el) {
    el = document.createElement('div');
    el.id = 'rain-ts';
    el.style.cssText =
      'position:absolute;bottom:8px;left:12px;padding:4px 8px;background:rgba(0,0,0,.45);color:#fff;border-radius:6px;font:12px system-ui;z-index:1200';
    document.body.appendChild(el);
  }
  const d = new Date(ts * 1000);
  el.textContent = `Radar: ${d.toLocaleString()}`;
}

function rehydrateAtmosphereIfNeeded() {
  const w = document.getElementById('toggle-wind');
  const t = document.getElementById('toggle-temp');
  const r = document.getElementById('toggle-rain');
  if (t?.checked) addTemperatureLayer();
  if (w?.checked) addWindLayer();
  if (r?.checked) addRainLayer();
}

function initAtmosControls() {
  const id = 'atmos-controls';
  if (!document.getElementById(id)) {
    const panel = document.createElement('div');
    panel.id = id;
    panel.style.cssText =
      'background:#fff;border:1px solid #e9ecef;border-radius:8px;padding:10px;margin-top:10px;box-shadow:0 1px 2px rgba(0,0,0,.05);font:14px/1.4 system-ui,sans-serif';
    panel.innerHTML = `
      <div style="font-weight:700;margin-bottom:6px">Atmosphere</div>
      <label><input id="toggle-wind" type="checkbox"> Wind (particles)</label><br>
      <label><input id="toggle-temp" type="checkbox"> Temperature (GFS 2m)</label><br>
      <label><input id="toggle-rain" type="checkbox"> Rain (radar)</label>
      <div id="rain-controls" style="display:none;margin-left:20px;margin-top:4px;font:12px system-ui">
        <button id="rain-playpause">⏸ Pause</button>
        Speed: <input id="rain-speed" type="range" min="100" max="1000" step="100" value="400">
      </div>`;
    (
      document.getElementById('filter') ||
      document.querySelector('.panel') ||
      document.getElementById('sidebar') ||
      document.body
    ).appendChild(panel);
  }

  const w = document.getElementById('toggle-wind');
  const t = document.getElementById('toggle-temp');
  const r = document.getElementById('toggle-rain');
  if (w && !w._wired) {
    w._wired = true;
    w.addEventListener('change', (e) =>
      e.target.checked ? addWindLayer() : removeWindLayer()
    );
  }
  if (t && !t._wired) {
    t._wired = true;
    t.addEventListener('change', (e) =>
      e.target.checked ? addTemperatureLayer() : removeTemperatureLayer()
    );
  }
  if (r && !r._wired) {
    r._wired = true;
    r.addEventListener('change', (e) => {
      const ctrls = document.getElementById('rain-controls');
      if (e.target.checked) {
        ctrls.style.display = 'block';
        addRainLayer();
      } else {
        ctrls.style.display = 'none';
        removeRainLayer();
      }
    });
  }

  const playBtn = document.getElementById('rain-playpause');
  const speedInput = document.getElementById('rain-speed');
  if (playBtn && !playBtn._wired) {
    playBtn._wired = true;
    playBtn.addEventListener('click', () => {
      if (_rainTimer) {
        stopRainLayer();
        playBtn.textContent = '▶️ Play';
      } else {
        startRainAnimation();
        playBtn.textContent = '⏸ Pause';
      }
    });
  }
  if (speedInput && !speedInput._wired) {
    speedInput._wired = true;
    speedInput.addEventListener('input', (e) => {
      _rainSpeed = parseInt(e.target.value, 10);
      if (_rainTimer) startRainAnimation();
    });
  }
}

// ---------------- Data Type Filter ----------------
const dataTypeSel = document.getElementById('data-type-select');
function setLayerVisibility(id, show) {
  if (map.getLayer(id))
    map.setLayoutProperty(id, 'visibility', show ? 'visible' : 'none');
}
function applyDataTypeFilter() {
  const v = dataTypeSel?.value || 'all';
  const showWind = v === 'all' || v === 'wind';
  const showSolar = v === 'all' || v === 'solar';
  setLayerVisibility('windmill-points', showWind);
  setLayerVisibility('solar-points', showSolar);
}
if (dataTypeSel) dataTypeSel.addEventListener('change', applyDataTypeFilter);

window.openWeatherModal = openWeatherModal;
window.closeWeatherModal = closeWeatherModal;
window.downloadWindReport = downloadWindReport;
window.downloadSolarReport = downloadSolarReport;

window.openModal = openModal;
window.closeModal = closeModal;
window.openInfoModal = openInfoModal;
window.closeInfoModal = closeInfoModal;

export {
  toggleSidebar,
  initializeMap,
  addLayersAndSources,
  populateProvinces,
  applyProvinceFilter,
  bindClicks,
  openModal,
  closeModal,
  openWeatherModal,
  closeWeatherModal,
  downloadWindReport,
  downloadSolarReport,
  applyDataTypeFilter,
};
