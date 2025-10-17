import React, { useEffect, useState } from 'react';
import { AuthProvider, useAuth } from './auth/AuthContext';
import Login from './components/Login';
import SettingsModal from './components/SettingsModal';
import './legacy/indexLegacy.css';

/* Root wrapper so auth is global */
export default function App() {
  return (
    <AuthProvider>
      <AppInner />
    </AuthProvider>
  );
}

function AppInner() {
  const { currentUser, logout } = useAuth();
  const [Layout, setLayout] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  // Completely clean the previous Mapbox instance so re-login works without refresh
  const resetMap = () => {
    try {
      if (
        window.__legacyMapInstance &&
        typeof window.__legacyMapInstance.remove === 'function'
      ) {
        window.__legacyMapInstance.remove();
      }
    } catch {}
    window.__legacyMapInstance = null;
    window.__legacyMapInit = false;
    const el = document.getElementById('map');
    if (el) el.innerHTML = '';
  };

  const handleLogout = () => {
    resetMap();
    logout();
  };

  // Role class (hides graph buttons for non-admin via CSS)
  useEffect(() => {
    document.body.classList.remove('role-admin', 'role-user');
    if (currentUser?.role === 'admin')
      document.body.classList.add('role-admin');
    if (currentUser?.role === 'user') document.body.classList.add('role-user');
  }, [currentUser]);

  // Mount/unmount legacy layout around auth changes
  useEffect(() => {
    if (!currentUser) {
      setLayout(null);
      resetMap();
      return;
    }
    import('./components/LegacyLayout').then((m) => setLayout(() => m.default));
  }, [currentUser]);

  // Initialize the legacy map exactly once per login (StrictMode-safe)
  useEffect(() => {
    if (!currentUser || !Layout) return;

    resetMap(); // ensure clean slate before init

    let tick = setInterval(async () => {
      if (!window.mapboxgl || !document.getElementById('map')) return;

      // Patch Map constructor once so we can track/remove the live instance
      if (!window.__mapboxPatchApplied) {
        const OriginalMap = window.mapboxgl.Map;
        window.mapboxgl.Map = function (...args) {
          const inst = new OriginalMap(...args);
          window.__legacyMapInstance = inst;
          return inst;
        };
        window.mapboxgl.Map.prototype = OriginalMap.prototype;
        Object.setPrototypeOf(window.mapboxgl.Map, OriginalMap);
        window.__mapboxPatchApplied = true;
      }

      if (window.__legacyMapInit) {
        clearInterval(tick);
        return;
      }

      const legacy = await import('./legacy/indexLegacy');
      Object.assign(window, legacy); // expose your original functions
      window.__legacyMapInit = true; // guard against double init
      legacy.initializeMap(); // call your original entry point
      setTimeout(() => window.dispatchEvent(new Event('resize')), 0);
      clearInterval(tick);
    }, 50);

    return () => clearInterval(tick);
  }, [currentUser, Layout]);

  if (!currentUser) return <Login />;

  return (
    <>
      <button id="settingsBtn" onClick={() => setSettingsOpen(true)}>
        Settings
      </button>
      <button
        id="logoutBtn"
        onClick={handleLogout}
        style={{ position: 'fixed', top: 12, right: 112, zIndex: 1200 }}
      >
        Log out
      </button>

      {Layout && <Layout />}

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      />
    </>
  );
}
