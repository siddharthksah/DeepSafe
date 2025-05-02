import React, { useState } from 'react';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Handle file selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setError(null);
    setResult(null);
    
    if (file) {
      if (!file.type.match('image.*')) {
        setError('Please select an image file');
        setSelectedFile(null);
        setPreview(null);
        return;
      }
      
      setSelectedFile(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target.result);
      reader.readAsDataURL(file);
    }
  };

  // Handle form submission
  const handleSubmit = async (event) => {
    event.preventDefault();
    
    if (!selectedFile) {
      setError('Please select an image file first');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      const response = await fetch(`${API_URL}/detect`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze image');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error('Error during analysis:', err);
      setError(err.message || 'An error occurred during analysis');
    } finally {
      setLoading(false);
    }
  };

  // Format probability as percentage
  const formatProbability = (prob) => {
    return (prob * 100).toFixed(2) + '%';
  };

  // Get color based on probability
  const getProbabilityColor = (prob) => {
    if (prob < 0.3) return '#4caf50';  // Green for likely real
    if (prob < 0.7) return '#ff9800';  // Orange for uncertain
    return '#f44336';  // Red for likely fake
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Deepfake Detection</h1>
        <p>Upload an image to analyze if it's a deepfake</p>
      </header>
      
      <main className="app-main">
        <section className="upload-section">
          <form onSubmit={handleSubmit}>
            <div className="file-input-container">
              <input
                type="file"
                id="image-upload"
                accept="image/*"
                onChange={handleFileChange}
                className="file-input"
              />
              <label htmlFor="image-upload" className="file-input-label">
                {selectedFile ? selectedFile.name : 'Choose an image'}
              </label>
            </div>
            
            <button 
              type="submit" 
              className="analyze-button"
              disabled={!selectedFile || loading}
            >
              {loading ? 'Analyzing...' : 'Analyze Image'}
            </button>
          </form>
          
          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </section>
        
        <div className="result-container">
          <div className="preview-section">
            {preview && (
              <div className="image-preview-container">
                <h3>Selected Image</h3>
                <img src={preview} alt="Preview" className="image-preview" />
              </div>
            )}
          </div>
          
          {result && (
            <div className="results-section">
              <h3>Analysis Result</h3>
              
              <div className="probability-meter">
                <div className="probability-label">
                  DeepFake Probability: 
                  <span 
                    style={{ color: getProbabilityColor(result.deepfake_probability) }}
                    className="probability-value"
                  >
                    {formatProbability(result.deepfake_probability)}
                  </span>
                </div>
                
                <div className="probability-bar-container">
                  <div 
                    className="probability-bar"
                    style={{ 
                      width: `${result.deepfake_probability * 100}%`,
                      backgroundColor: getProbabilityColor(result.deepfake_probability)
                    }}
                  />
                </div>
                
                <div className="conclusion">
                  This image is {result.is_likely_deepfake ? 
                    <span style={{ color: '#f44336' }}>likely a deepfake</span> : 
                    <span style={{ color: '#4caf50' }}>likely authentic</span>
                  }
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
      
      <footer className="app-footer">
        <p>Powered by CNNDetection - A deep learning model for detecting CNN-generated images</p>
      </footer>
    </div>
  );
}

export default App;