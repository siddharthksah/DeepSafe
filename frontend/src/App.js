import React, { useState } from 'react';
import './App.css';
import Header from './components/Header';
import Footer from './components/Footer';
import UploadSection from './components/UploadSection';
import ResultsSection from './components/ResultsSection';
import SettingsPanel from './components/SettingsPanel';
import InfoSection from './components/InfoSection';

const App = () => {
  // State variables
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [processStage, setProcessStage] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [selectedModels, setSelectedModels] = useState([]);
  const [threshold, setThreshold] = useState(0.5);
  const [ensembleMethod, setEnsembleMethod] = useState('voting');
  const [showSettings, setShowSettings] = useState(false);

  // All available models
  const availableModels = [
    { id: 'cnndetection', name: 'CNN Detection', description: 'Detects CNN-generated images' },
    { id: 'ganimagedetection', name: 'GAN Image Detection', description: 'Specialized for GAN-generated images' },
    { id: 'universalfakedetect', name: 'Universal Fake Detect', description: 'General-purpose fake image detection' },
    { id: 'hifi_ifdl', name: 'HiFi-IFDL', description: 'High fidelity image forensics' },
    { id: 'npr_deepfakedetection', name: 'NPR Deepfake Detection', description: 'Neural Pattern Recognition for deepfakes' },
    { id: 'dmimagedetection', name: 'DM Image Detection', description: 'Diffusion Model image detection' },
    { id: 'caddm', name: 'CADDM', description: 'Convolutional Artifact Detection' },
    { id: 'faceforensics_plus_plus', name: 'FaceForensics++', description: 'Face manipulation detection' }
  ];

  // Handle file selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setError(null);
    setResult(null);
    
    if (file) {
      if (!file.type.match('image.*')) {
        setError('Please select an image file (JPEG, PNG, or WebP)');
        setSelectedFile(null);
        setPreview(null);
        return;
      }
      
      // Check file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        setError('File size exceeds 10MB limit');
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

// Simulate a file drop or camera capture for demo purposes
const handleDemoImage = (demoType) => {
  // In a real app, this would trigger the camera or handle dropped files
  setError(null);
  setResult(null);
  setLoading(true);
  setProcessStage('Preparing demo image...');
  
  // Simulate loading
  setTimeout(() => {
    // Use the actual demo image from public folder
    setPreview('/images/demo_image.jpg');
    
    // Create a synthetic file object similar to what would come from a file input
    fetch('/images/demo_image.jpg')
      .then(response => response.blob())
      .then(blob => {
        const file = new File([blob], 'demo_image.jpg', { type: 'image/jpeg' });
        setSelectedFile(file);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error loading demo image:', error);
        setError('Failed to load demo image. Please try uploading an image instead.');
        setLoading(false);
      });
  }, 1000);
};

  // Toggle settings panel
  const toggleSettings = () => {
    setShowSettings(!showSettings);
  };

  // Toggle model selection
  const toggleModel = (modelId) => {
    setSelectedModels(prevModels => 
      prevModels.includes(modelId)
        ? prevModels.filter(id => id !== modelId)
        : [...prevModels, modelId]
    );
  };

  // Handle form submission
  const handleSubmit = async () => {
    if (!selectedFile) {
      setError('Please select an image file first');
      return;
    }
    
    setLoading(true);
    setError(null);
    setProcessStage('Preparing image for analysis...');
    
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('threshold', threshold);
      formData.append('ensemble_method', ensembleMethod);
      
      if (selectedModels.length > 0) {
        formData.append('models', selectedModels.join(','));
      }
      
      setProcessStage('Uploading image to analysis server...');
      
      const response = await fetch('/api/detect', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze image');
      }
      
      setProcessStage('Processing results...');
      const data = await response.json();
      
      // Short delay to ensure smooth UI transition
      setTimeout(() => {
        setResult(data);
        setLoading(false);
      }, 500);
      
    } catch (err) {
      console.error('Error during analysis:', err);
      setError(err.message || 'An error occurred during analysis');
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <Header 
        showSettings={showSettings} 
        toggleSettings={toggleSettings} 
      />

      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {/* Settings Panel */}
        {showSettings && (
          <SettingsPanel 
            threshold={threshold}
            setThreshold={setThreshold}
            ensembleMethod={ensembleMethod}
            setEnsembleMethod={setEnsembleMethod}
            selectedModels={selectedModels}
            toggleModel={toggleModel}
            availableModels={availableModels}
          />
        )}

        {/* Upload Section */}
        <UploadSection 
          selectedFile={selectedFile}
          setSelectedFile={setSelectedFile}
          preview={preview}
          setPreview={setPreview}
          setResult={setResult}
          loading={loading}
          processStage={processStage}
          error={error}
          handleFileChange={handleFileChange}
          handleDemoImage={handleDemoImage}
          handleSubmit={handleSubmit}
        />
        
        {/* Results Section */}
        {result && <ResultsSection result={result} />}
        
        {/* Information Section */}
        <InfoSection />
      </main>
      
      <Footer />
    </div>
  );
};

export default App;