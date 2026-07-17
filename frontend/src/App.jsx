import { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';

const API_BASE = "http://localhost:8000";

function App() {
  const [analytics, setAnalytics] = useState({
    status: "SAFE",
    inflow_rate: 0,
    eta_seconds: null,
    current_count: 0,
    threshold: 100,
    message: "Initializing..."
  });
  
  const [config, setConfig] = useState({
    mode: "LIVE (YOLOv8)",
    line_y_fraction: 0.5,
    frame_skip: 2,
    capacity: 100,
    emergency_override: false
  });

  const [history, setHistory] = useState([]);

  // Fetch initial config
  useEffect(() => {
    fetch(`${API_BASE}/config`)
      .then(res => res.json())
      .then(data => setConfig(data))
      .catch(err => console.error("Config fetch error:", err));
  }, []);

  // Setup SSE for real-time analytics
  useEffect(() => {
    const eventSource = new EventSource(`${API_BASE}/analytics`);
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setAnalytics(data);
      
      // Update chart history
      setHistory(prev => {
        const newHist = [...prev, { time: new Date().toLocaleTimeString(), count: data.current_count }];
        if (newHist.length > 20) return newHist.slice(newHist.length - 20);
        return newHist;
      });
    };

    eventSource.onerror = (err) => {
      console.error("SSE Error:", err);
    };

    return () => eventSource.close();
  }, []);

  const updateConfig = (updates) => {
    const newConfig = { ...config, ...updates };
    setConfig(newConfig);
    
    fetch(`${API_BASE}/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updates)
    }).catch(err => console.error("Config update error:", err));
  };

  return (
    <div className="app-container">
      {/* SIDEBAR: Controls */}
      <div className="sidebar glass-panel">
        <div>
          <h2>StampedeZero</h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>
            Decoupled Vision Core
          </p>
        </div>

        <div className="control-group">
          <label>Operating Mode</label>
          <select 
            value={config.mode} 
            onChange={(e) => updateConfig({ mode: e.target.value })}
          >
            <option value="LIVE (YOLOv8)">LIVE (YOLOv8)</option>
            <option value="STRESS TEST (CSRNet)">STRESS TEST (CSRNet)</option>
          </select>
        </div>

        {config.mode === "LIVE (YOLOv8)" && (
          <div className="control-group">
            <label>
              Virtual Line Position
              <span>{Math.round(config.line_y_fraction * 100)}%</span>
            </label>
            <input 
              type="range" min="0.1" max="0.9" step="0.05"
              value={config.line_y_fraction}
              onChange={(e) => updateConfig({ line_y_fraction: parseFloat(e.target.value) })}
            />
          </div>
        )}

        <div className="control-group">
          <label>
            Venue Capacity Limit
            <span>{config.capacity} ppl</span>
          </label>
          <input 
            type="range" min="10" max="5000" step="10"
            value={config.capacity}
            onChange={(e) => updateConfig({ capacity: parseInt(e.target.value) })}
          />
        </div>

        <div className="control-group">
          <label>
            Performance Frame Skip
            <span>{config.frame_skip}</span>
          </label>
          <input 
            type="range" min="1" max="10" step="1"
            value={config.frame_skip}
            onChange={(e) => updateConfig({ frame_skip: parseInt(e.target.value) })}
          />
        </div>

        <hr style={{ borderColor: 'rgba(255,255,255,0.1)', margin: '24px 0' }} />

        <div className="control-group">
          <label style={{ color: config.emergency_override ? 'var(--critical-color)' : 'inherit' }}>
            🚨 Emergency Override
          </label>
          <button 
            className={`btn-override ${config.emergency_override ? 'active' : ''}`}
            onClick={() => updateConfig({ emergency_override: !config.emergency_override })}
            style={{
              width: '100%', padding: '12px', borderRadius: '8px', border: 'none', cursor: 'pointer',
              fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '1px',
              backgroundColor: config.emergency_override ? 'var(--critical-color)' : 'rgba(255,255,255,0.1)',
              color: '#fff', transition: 'all 0.3s ease'
            }}
          >
            {config.emergency_override ? 'OVERRIDE ACTIVE' : 'FORCE CRITICAL'}
          </button>
        </div>
      </div>

      {/* MAIN: Video Feed */}
      <div className="main-content">
        <div className={`status-banner status-${analytics.status}`}>
          {analytics.message}
        </div>
        
        <div className="video-container glass-panel">
          {/* MJPEG Stream directly from FastAPI */}
          <img 
            src={`${API_BASE}/video_feed`} 
            alt="Live Camera Feed"
            onError={(e) => {
              e.target.style.display = 'none';
              e.target.parentNode.style.display = 'flex';
              e.target.parentNode.style.alignItems = 'center';
              e.target.parentNode.style.justifyContent = 'center';
              e.target.parentNode.innerHTML = '<div style="color:var(--text-secondary)">Video stream offline. Make sure the FastAPI server is running.</div>';
            }}
          />
        </div>
      </div>

      {/* RIGHT PANEL: Analytics */}
      <div className="analytics-panel glass-panel">
        <h3>Real-time Analytics</h3>
        
        <div className="metric-card glass-panel" style={{ background: "rgba(0,0,0,0.2)", marginBottom: '16px' }}>
          <span className="title">Current Density</span>
          <span className="value">{analytics.current_count} <span style={{fontSize:'1rem', color:'var(--text-secondary)'}}>/ {config.capacity}</span></span>
          
          {/* Capacity Progress Bar */}
          <div style={{ width: '100%', backgroundColor: 'rgba(255,255,255,0.05)', borderRadius: '4px', height: '8px', marginTop: '12px', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.1)' }}>
            <div style={{
              width: `${Math.min(100, (analytics.current_count / (config.capacity || 1)) * 100)}%`,
              backgroundColor: (analytics.current_count / (config.capacity || 1)) >= 0.85 ? '#e74c3c' : ((analytics.current_count / (config.capacity || 1)) >= 0.7 ? '#ff8c42' : '#2ecc71'),
              height: '100%',
              transition: 'width 0.3s ease-in-out'
            }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            <span>0%</span>
            <span>{Math.round(Math.min(100, (analytics.current_count / (config.capacity || 1)) * 100))}% Allowed</span>
            <span>100%</span>
          </div>
        </div>

        {/* Set Capacity Limit Input Box / Preset Tabs */}
        <div className="metric-card glass-panel" style={{ background: "rgba(255,255,255,0.02)", marginBottom: '16px', padding: '16px' }}>
          <span className="title" style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Set Allowed Screen Limit</span>
          <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
            <input 
              type="number" 
              min="1" 
              max="5000" 
              value={config.capacity}
              onChange={(e) => updateConfig({ capacity: parseInt(e.target.value) || 10 })}
              style={{
                flex: 1,
                background: 'rgba(0,0,0,0.3)',
                border: '1px solid var(--panel-border)',
                borderRadius: '8px',
                color: 'white',
                padding: '8px 12px',
                fontSize: '1rem',
                outline: 'none'
              }}
            />
            <button 
              className={`preset-btn ${config.capacity === 10 ? 'active' : ''}`}
              onClick={() => updateConfig({ capacity: 10 })}
              style={{ background: config.capacity === 10 ? 'var(--accent-color)' : 'rgba(255,255,255,0.05)', border: '1px solid var(--panel-border)', color: 'white', borderRadius: '8px', padding: '0 12px', cursor: 'pointer', fontSize: '0.9rem', transition: 'all 0.2s' }}
            >
              10
            </button>
            <button 
              className={`preset-btn ${config.capacity === 50 ? 'active' : ''}`}
              onClick={() => updateConfig({ capacity: 50 })}
              style={{ background: config.capacity === 50 ? 'var(--accent-color)' : 'rgba(255,255,255,0.05)', border: '1px solid var(--panel-border)', color: 'white', borderRadius: '8px', padding: '0 12px', cursor: 'pointer', fontSize: '0.9rem', transition: 'all 0.2s' }}
            >
              50
            </button>
            <button 
              className={`preset-btn ${config.capacity === 100 ? 'active' : ''}`}
              onClick={() => updateConfig({ capacity: 100 })}
              style={{ background: config.capacity === 100 ? 'var(--accent-color)' : 'rgba(255,255,255,0.05)', border: '1px solid var(--panel-border)', color: 'white', borderRadius: '8px', padding: '0 12px', cursor: 'pointer', fontSize: '0.9rem', transition: 'all 0.2s' }}
            >
              100
            </button>
          </div>
        </div>

        {/* Counters */}
        <div style={{ display: 'flex', gap: '12px', marginBottom: '16px' }}>
          <div className="metric-card glass-panel" style={{ flex: 1, background: "rgba(46, 204, 113, 0.05)", borderLeft: '3px solid #2ecc71', margin: 0, padding: '12px' }}>
            <span className="title" style={{ fontSize: '0.8rem', color: '#2ecc71' }}>Entered</span>
            <span className="value" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{analytics.total_in || 0}</span>
          </div>
          <div className="metric-card glass-panel" style={{ flex: 1, background: "rgba(231, 76, 60, 0.05)", borderLeft: '3px solid #e74c3c', margin: 0, padding: '12px' }}>
            <span className="title" style={{ fontSize: '0.8rem', color: '#e74c3c' }}>Exited</span>
            <span className="value" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{analytics.total_out || 0}</span>
          </div>
        </div>

        {/* History Chart */}
        <div style={{ height: 200, marginTop: 12 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="time" hide />
              <YAxis stroke="rgba(255,255,255,0.5)" />
              <Tooltip 
                contentStyle={{ background: '#191c24', border: 'none', borderRadius: 8, color: '#fff' }}
              />
              <Line 
                type="monotone" 
                dataKey="count" 
                stroke="var(--accent-color)" 
                strokeWidth={3}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {analytics.pdf_generated && (
          <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(50, 200, 100, 0.2)', borderLeft: '4px solid #32c864', borderRadius: '4px', fontSize: '0.9rem' }}>
            📄 <strong>PDF Generated:</strong> {analytics.pdf_generated}
          </div>
        )}

      </div>
    </div>
  );
}

export default App;
