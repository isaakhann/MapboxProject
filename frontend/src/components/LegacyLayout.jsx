import React, { useState } from 'react';

export default function LegacyLayout() {
  const [showDateModal, setShowDateModal] = useState(false);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [analysisType, setAnalysisType] = useState('BOTH');
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Called when button is clicked
  const handleOpenAnalysis = () => {
    if (window.storedPolygon) {
      setShowDateModal(true);
    } else {
      alert(
        'No polygon drawn yet. Please use the pentagon tool on the map to draw an area first.'
      );
    }
  };

  const handleRunAnalysis = async () => {
    if (!startDate || !endDate) {
      alert('Please select both start and end dates.');
      return;
    }

    setIsAnalyzing(true);

    const payload = {
      start_date: startDate,
      end_date: endDate,
      index_type: analysisType, // NOW supports: BOTH | AGRI | ALL
      geometry: {
        type: 'Polygon',
        coordinates: window.storedPolygon,
      },
    };

    try {
      const res = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (res.ok && data.status === 'success') {
        // --- NEW: Display the AI Analysis Text Immediately ---
        // Using window.confirm allows showing text + option to open report
        const wantsReport = window.confirm(
          '🤖 AI ANALYSIS REPORT:\n\n' +
            (data.ai_analysis || 'No analysis text returned.') +
            '\n\n--------------------------------\n' +
            'Click OK to open the full raw data report (JSON).'
        );

        if (wantsReport) {
          window.open(data.report_url, '_blank');
        }
      } else {
        alert('Analysis Failed: ' + (data.message || 'Unknown error'));
      }
    } catch (err) {
      console.error(err);
      alert('Error contacting server.');
    } finally {
      setIsAnalyzing(false);
      setShowDateModal(false);
    }
  };

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

          {/* --- Updated Button Name --- */}
          <button
            className="graph-btn"
            onClick={handleOpenAnalysis}
            style={{ backgroundColor: '#28a745' }}
          >
            Run Advanced Analysis
          </button>

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

      {/* --- ANALYSIS MODAL --- */}
      {showDateModal && (
        <div className="modal" style={{ display: 'flex' }}>
          <div className="modal-content" style={{ width: '400px' }}>
            <div className="modal-header">
              <span>Run Analysis</span>
              <span className="close" onClick={() => setShowDateModal(false)}>
                &times;
              </span>
            </div>
            <div className="modal-body" style={{ padding: '20px' }}>
              {/* Type Selection */}
              <div style={{ marginBottom: '15px' }}>
                <label
                  style={{
                    display: 'block',
                    marginBottom: '5px',
                    fontWeight: 'bold',
                  }}
                >
                  Analysis Type:
                </label>
                <select
                  value={analysisType}
                  onChange={(e) => setAnalysisType(e.target.value)}
                  style={{ width: '100%', padding: '8px' }}
                >
                  <option value="BOTH">🔥 Burn Analysis (BAI + BAIS2)</option>
                  <option value="AGRI">
                    🌾 Agricultural Report (NDVI/NDMI/NDWI/NDRE)
                  </option>
                  <option value="ALL">
                    🔥🌾 Burn + Agriculture (BAIS2/BAI + NDVI/NDMI/NDWI/NDRE)
                  </option>
                </select>
              </div>

              <div style={{ marginBottom: '15px' }}>
                <label style={{ display: 'block', marginBottom: '5px' }}>
                  Start Date:
                </label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  style={{ width: '100%', padding: '8px' }}
                />
              </div>
              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', marginBottom: '5px' }}>
                  End Date:
                </label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  style={{ width: '100%', padding: '8px' }}
                />
              </div>
              <button
                className="graph-btn"
                onClick={handleRunAnalysis}
                disabled={isAnalyzing}
                style={{ backgroundColor: isAnalyzing ? '#ccc' : '#0d6efd' }}
              >
                {isAnalyzing ? 'Processing...' : 'Run Analysis'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Existing Modals */}
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
