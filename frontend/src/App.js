import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import Login from './components/Login';
import Register from './components/Register';
import './App.css';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
    ArcElement,
} from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
    ArcElement
);

const API_BASE_URL = 'http://localhost:8000';

// Protected Route Wrapper
const ProtectedRoute = ({ children, isAuthenticated }) => {
    const location = useLocation();
    if (!isAuthenticated) {
        return <Navigate to="/login" state={{ from: location }} replace />;
    }
    return children;
};

function Dashboard({ onLogout }) {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [mediaType, setMediaType] = useState('image');
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);
    const [systemHealth, setSystemHealth] = useState(null);

    useEffect(() => {
        fetch(`${API_BASE_URL}/health`)
            .then((res) => res.json())
            .then((data) => setSystemHealth(data))
            .catch((err) => console.error('Health check failed:', err));
    }, []);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setSelectedFile(file);
            setResults(null);
            setError(null);
            const objectUrl = URL.createObjectURL(file);
            setPreviewUrl(objectUrl);
            if (file.type.startsWith('video/')) {
                setMediaType('video');
            } else {
                setMediaType('image');
            }
        }
    };

    const handleAnalyze = async () => {
        if (!selectedFile) return;
        setIsAnalyzing(true);
        setError(null);
        setResults(null);

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('media_type', mediaType);

        try {
            const response = await fetch(`${API_BASE_URL}/detect`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Analysis failed');
            }

            const data = await response.json();
            setResults(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsAnalyzing(false);
        }
    };

    const renderHealthStatus = () => {
        if (!systemHealth) return <span className="text-secondary">Checking system...</span>;
        const status = systemHealth.overall_api_status || 'Unknown';
        const color = status === 'healthy' ? 'var(--success)' : 'var(--danger)';
        return (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: color, boxShadow: `0 0 8px ${color}` }}></span>
                <span style={{ color: color, fontWeight: 600, fontSize: '0.875rem' }}>{status.toUpperCase()}</span>
            </div>
        );
    };

    const renderResults = () => {
        if (!results) return null;
        const isFake = results.is_likely_deepfake;
        const confidence = (results.deepfake_probability * 100).toFixed(1);
        const verdictColor = isFake ? 'var(--danger)' : 'var(--success)';
        const verdictText = isFake ? 'FAKE' : 'REAL';

        const barData = {
            labels: Object.keys(results.model_results || {}),
            datasets: [
                {
                    label: 'Fake Probability',
                    data: Object.values(results.model_results || {}).map(r => r.probability ? r.probability * 100 : 0),
                    backgroundColor: 'rgba(59, 130, 246, 0.6)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 1,
                    borderRadius: 4,
                },
            ],
        };

        const doughnutData = {
            labels: ['Real', 'Fake'],
            datasets: [
                {
                    data: [100 - (results.deepfake_probability * 100), results.deepfake_probability * 100],
                    backgroundColor: ['rgba(16, 185, 129, 0.6)', 'rgba(239, 68, 68, 0.6)'],
                    borderColor: ['rgba(16, 185, 129, 1)', 'rgba(239, 68, 68, 1)'],
                    borderWidth: 1,
                },
            ],
        };

        return (
            <div className="animate-fade-in" style={{ marginTop: '2rem' }}>
                <div className="card" style={{ textAlign: 'center', marginBottom: '2rem', borderTop: `4px solid ${verdictColor}` }}>
                    <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Analysis Verdict</h2>
                    <div style={{ fontSize: '3rem', fontWeight: '800', color: verdictColor, textShadow: `0 0 20px ${verdictColor}40` }}>
                        {verdictText}
                    </div>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem' }}>
                        Confidence: <strong style={{ color: 'var(--text-primary)' }}>{confidence}%</strong>
                    </p>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem' }}>
                    <div className="card">
                        <h3 style={{ marginBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.5rem' }}>Ensemble Score</h3>
                        <div style={{ height: '250px', display: 'flex', justifyContent: 'center' }}>
                            <Doughnut data={doughnutData} options={{ maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8' } } } }} />
                        </div>
                    </div>
                    <div className="card">
                        <h3 style={{ marginBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.5rem' }}>Individual Model Scores</h3>
                        <div style={{ height: '250px' }}>
                            <Bar data={barData} options={{ maintainAspectRatio: false, scales: { y: { beginAtZero: true, max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }, x: { grid: { display: false }, ticks: { color: '#94a3b8' } } }, plugins: { legend: { display: false } } }} />
                        </div>
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="App">
            <header style={{
                padding: '1.5rem 2rem',
                borderBottom: '1px solid rgba(255,255,255,0.05)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                background: 'rgba(15, 23, 42, 0.8)',
                backdropFilter: 'blur(10px)',
                position: 'sticky',
                top: 0,
                zIndex: 100
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <div style={{
                        width: '40px',
                        height: '40px',
                        background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
                        borderRadius: '8px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontWeight: 'bold',
                        fontSize: '1.2rem'
                    }}>DS</div>
                    <h1 style={{ fontSize: '1.5rem', fontWeight: '700', letterSpacing: '-0.025em' }}>DeepSafe</h1>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
                    {renderHealthStatus()}
                    <button onClick={onLogout} className="btn btn-secondary" style={{ padding: '0.5rem 1rem', fontSize: '0.9rem' }}>Logout</button>
                </div>
            </header>

            <main className="container">
                <div style={{ textAlign: 'center', marginBottom: '3rem', paddingTop: '2rem' }}>
                    <h2 className="text-gradient" style={{ fontSize: '3rem', fontWeight: '800', marginBottom: '1rem', lineHeight: 1.2 }}>
                        Detect Deepfakes with<br />Enterprise Precision
                    </h2>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '1.25rem', maxWidth: '600px', margin: '0 auto' }}>
                        Upload your media to analyze it against our multi-model ensemble engine.
                    </p>
                </div>

                <div className="card" style={{ maxWidth: '800px', margin: '0 auto', padding: '3rem', borderStyle: 'dashed', borderWidth: '2px', borderColor: 'rgba(255,255,255,0.1)', backgroundColor: 'rgba(30, 41, 59, 0.5)' }}>
                    <input
                        type="file"
                        id="file-upload"
                        style={{ display: 'none' }}
                        onChange={handleFileChange}
                        accept="image/*,video/*"
                    />
                    <label htmlFor="file-upload" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', cursor: 'pointer' }}>
                        <div style={{
                            width: '64px',
                            height: '64px',
                            borderRadius: '50%',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            marginBottom: '1.5rem',
                            color: 'var(--accent-primary)'
                        }}>
                            <svg width="32" height="32" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                        </div>
                        <span style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '0.5rem' }}>
                            {selectedFile ? selectedFile.name : 'Click to Upload or Drag & Drop'}
                        </span>
                        <span style={{ color: 'var(--text-secondary)' }}>Supported formats: JPG, PNG, MP4, AVI</span>
                    </label>
                </div>

                {previewUrl && (
                    <div className="animate-fade-in" style={{ marginTop: '2rem', maxWidth: '800px', margin: '2rem auto 0' }}>
                        <div className="card" style={{ overflow: 'hidden', padding: 0 }}>
                            <div style={{ position: 'relative', width: '100%', height: '400px', backgroundColor: '#000' }}>
                                {mediaType === 'video' ? (
                                    <video src={previewUrl} controls style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                                ) : (
                                    <img src={previewUrl} alt="Preview" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                                )}
                            </div>
                            <div style={{ padding: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: 'var(--bg-secondary)' }}>
                                <div>
                                    <h3 style={{ fontSize: '1.1rem', fontWeight: '600' }}>Ready to Analyze</h3>
                                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>{selectedFile.size > 1024 * 1024 ? `${(selectedFile.size / (1024 * 1024)).toFixed(2)} MB` : `${(selectedFile.size / 1024).toFixed(2)} KB`}</p>
                                </div>
                                <button
                                    className="btn btn-primary"
                                    onClick={handleAnalyze}
                                    disabled={isAnalyzing}
                                    style={{ opacity: isAnalyzing ? 0.7 : 1, minWidth: '150px' }}
                                >
                                    {isAnalyzing ? 'Processing...' : 'Run DeepSafe'}
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {error && (
                    <div className="animate-fade-in" style={{ maxWidth: '800px', margin: '2rem auto 0', padding: '1rem', borderRadius: 'var(--radius-md)', backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', color: 'var(--danger)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        {error}
                    </div>
                )}

                <div style={{ maxWidth: '800px', margin: '0 auto' }}>
                    {renderResults()}
                </div>
            </main>

            <footer style={{ textAlign: 'center', padding: '3rem 0', color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '4rem', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                <p>&copy; {new Date().getFullYear()} DeepSafe Platform. Open Source & Enterprise Ready.</p>
            </footer>
        </div>
    );
}

function App() {
    const [isAuthenticated, setIsAuthenticated] = useState(() => {
        return localStorage.getItem('token') !== null;
    });

    const handleLogin = (token) => {
        localStorage.setItem('token', token);
        setIsAuthenticated(true);
    };

    const handleLogout = () => {
        localStorage.removeItem('token');
        setIsAuthenticated(false);
    };

    return (
        <Router>
            <Routes>
                <Route path="/login" element={<Login onLogin={handleLogin} />} />
                <Route path="/register" element={<Register onLogin={handleLogin} />} />
                <Route
                    path="/"
                    element={
                        <ProtectedRoute isAuthenticated={isAuthenticated}>
                            <Dashboard onLogout={handleLogout} />
                        </ProtectedRoute>
                    }
                />
            </Routes>
        </Router>
    );
}

export default App;
