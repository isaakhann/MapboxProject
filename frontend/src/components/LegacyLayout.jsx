import React from 'react';

export default function LegacyLayout() {
  return (
    <div>
      <div
        id="toggleBtn"
        onClick={() => window.toggleSidebar && window.toggleSidebar()}
      >
        ☰
      </div>
      <div id="container">
        <div id="filter">
          <label htmlFor="province-select">Filter by Province:</label>
          <select id="province-select">
            <option value="all">All</option>
          </select>

          <label htmlFor="layer-style">Map Style:</label>
          <select id="layer-style">
            <option value="satellite-streets-v12">Satellite</option>
            <option value="streets-v12">Streets</option>
            <option value="light-v11">Light</option>
            <option value="dark-v11">Dark</option>
            <option value="outdoors-v12">Outdoors</option>
          </select>

          <label htmlFor="data-type-select">Data Type:</label>
          <select id="data-type-select">
            <option value="all">All</option>
            <option value="wind">Wind</option>
            <option value="solar">Solar</option>
            <option value="none">None</option>{' '}
          </select>

          <button
            className="graph-btn"
            onClick={() => window.openModal && window.openModal('scatter')}
          >
            Scatter
          </button>
          <button
            className="graph-btn"
            onClick={() => window.openModal && window.openModal('bar')}
          >
            Histogram
          </button>
          <button
            className="graph-btn"
            onClick={() => window.openModal && window.openModal('scatter_int')}
          >
            Interactive Graph
          </button>
          <button
            className="graph-btn"
            onClick={() => window.openInfoModal && window.openInfoModal()}
          >
            About Scatter
          </button>
        </div>

        <div id="map"></div>
      </div>

      <div id="graphModal" className="modal" style={{ display: 'none' }}>
        <div className="modal-content">
          <div className="modal-header">
            <span id="modalTitle">Graph</span>
            <span
              className="close"
              onClick={() => window.closeModal && window.closeModal()}
            >
              &times;
            </span>
          </div>
          <iframe id="graphFrame" title="graph" style={{ border: 0 }} />
        </div>
      </div>

      <div id="weatherModal" className="modal" style={{ display: 'none' }}>
        <div className="modal-content">
          <div className="modal-header">
            <span>Weather</span>
            <span
              className="close"
              onClick={() =>
                window.closeWeatherModal && window.closeWeatherModal()
              }
            >
              &times;
            </span>
          </div>
          <iframe id="weatherFrame" title="weather" style={{ border: 0 }} />
        </div>
      </div>

      <div id="infoModal" className="modal" style={{ display: 'none' }}>
        <div className="modal-content">
          <div className="modal-header">
            <span>About</span>
            <span
              className="close"
              onClick={() => window.closeInfoModal && window.closeInfoModal()}
            >
              &times;
            </span>
          </div>
          <div className="modal-body">
            <p>This modal shows information about the graphs.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
