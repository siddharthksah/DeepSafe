// frontend/src/App.js
import React, { useState, useEffect, useCallback, useRef } from 'react';
import Header from './components/Header';
import Footer from './components/Footer';
import UploadSection from './components/UploadSection';
import ResultsSection from './components/ResultsSection';
import SettingsPanel from './components/SettingsPanel';
import InfoSection from './components/InfoSection';
import NavigationBar from './components/NavigationBar';
import ErrorBoundary from './components/ErrorBoundary';
import ModelHealthDashboard from './components/ModelHealthDashboard';
import Toast from './components/Toast';
import { exportResults, downloadPDF } from './utils/exportUtils';

const POLLING_INTERVALS = {
  FIRST_CHECK_DELAY: 5000, 
  INITIAL_FAST: 7000,      
  NORMAL_UNHEALTHY: 20000, 
  STABLE_HEALTHY: 60000,   
};

const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [processStage, setProcessStage] = useState('');
  const [modelProgress, setModelProgress] = useState({});
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [debugMode, setDebugMode] = useState(process.env.NODE_ENV === 'development');
  const [toast, setToast] = useState(null);
  
  const [selectedModels, setSelectedModels] = useState(() => {
    const saved = localStorage.getItem('deepsafeSelectedModels');
    return saved ? JSON.parse(saved) : [];
  });
  
  const [threshold, setThreshold] = useState(() => {
    const saved = localStorage.getItem('deepsafeThreshold');
    return saved ? parseFloat(saved) : 0.5;
  });
  
  const [ensembleMethod, setEnsembleMethod] = useState(() => {
    const saved = localStorage.getItem('deepsafeEnsembleMethod');
    return saved || 'stacking'; 
  });

  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('deepsafeDarkMode');
    if (saved !== null) return saved === 'true';
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  const [showSettings, setShowSettings] = useState(false);
  const [activeTab, setActiveTab] = useState('detect'); 
  const [activeMediaType, setActiveMediaType] = useState('image'); 

  const [availableModels, setAvailableModels] = useState([]);
  const [modelHealthStatus, setModelHealthStatus] = useState({});
  const [stackingAvailableByMediaType, setStackingAvailableByMediaType] = useState({});
  
  const [isHealthCheckInProgress, setIsHealthCheckInProgress] = useState(false);
  const healthCheckTimeoutRef = useRef(null);
  const mountedRef = useRef(true); 

  const [isModelDashboardExpanded, setIsModelDashboardExpanded] = useState(false);

  const baseModelInfo = {
    "npr_deepfakedetection": { name: 'NPR DeepFake', description: 'Neural Pattern Recognition for image deepfakes.', type: 'image' },
    "yermandy_clip_detection": { name: 'Yermandy CLIP', description: 'CLIP-based image deepfake detection.', type: 'image' },
    "wavelet_clip_detection": { name: 'Wavelet CLIP', description: 'Wavelet Transform + CLIP for image forensics.', type: 'image' },
    "universalfakedetect": { name: 'Universal Detector', description: 'General deepfake detection for images.', type: 'image' },
    "trufor": { name: 'TruFor', description: 'Transformer for image Forgery Detection.', type: 'image' },
    "spsl_deepfake_detection": { name: 'SPSL DeepFake', description: 'SPSL (DeepfakeBench) for image deepfakes.', type: 'image' },
    "ucf_deepfake_detection": { name: 'UCF DeepFake', description: 'Uncovering Common Features (DeepfakeBench) for images.', type: 'image' },
    "cross_efficient_vit": { name: "CrossEfficientViT", description: "Combines EfficientNet and Vision Transformers for video deepfake detection.", type: "video" },
  };

  const showToast = useCallback((message, type = 'success', duration = 3000) => {
    setToast({ message, type, id: Date.now() });
    setTimeout(() => setToast(null), duration);
  }, []);

  useEffect(() => {
    localStorage.setItem('deepsafeSelectedModels', JSON.stringify(selectedModels));
  }, [selectedModels]);

  useEffect(() => {
    localStorage.setItem('deepsafeThreshold', threshold.toString());
  }, [threshold]);

  useEffect(() => {
    localStorage.setItem('deepsafeEnsembleMethod', ensembleMethod);
  }, [ensembleMethod]);

  useEffect(() => {
    localStorage.setItem('deepsafeDarkMode', darkMode.toString());
    // Apply dark class to both html and body for better coverage
    if (darkMode) {
      document.documentElement.classList.add('dark');
      document.body.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
      document.body.classList.remove('dark');
    }
  }, [darkMode]);

  const performAndScheduleHealthCheck = useCallback(async (isInitialFastPollPhase = true) => {
    if (isHealthCheckInProgress || !mountedRef.current) return;
    setIsHealthCheckInProgress(true);
    let nextInterval = POLLING_INTERVALS.NORMAL_UNHEALTHY;
    try {
      const response = await fetch('/api/health');
      if (!mountedRef.current) return;
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Health check request failed." }));
        console.error('Health check API request failed:', response.status, errorData);
        setStackingAvailableByMediaType({});
        const errorStatus = {};
        Object.keys(baseModelInfo).forEach(id => {
          errorStatus[id] = { status: 'error', message: 'Health check failed' };
        });
        setModelHealthStatus(errorStatus);
        setAvailableModels(Object.keys(baseModelInfo).map(id => ({
          id, name: baseModelInfo[id]?.name || id, description: baseModelInfo[id]?.description || 'Model',
          type: baseModelInfo[id]?.type || 'unknown', status: 'error'
        })));
      } else {
        const data = await response.json();
        if (!mountedRef.current) return;

        const apiMediaTypeDetails = data.media_type_details || {};
        let allModelsFromApi = [];
        let newStackingStatus = {};
        let newModelHealth = {};

        Object.entries(apiMediaTypeDetails).forEach(([mediaType, typeDetails]) => {
          newStackingStatus[mediaType] = typeDetails.stacking_ensemble_loaded || false;
          const modelsForType = typeDetails.models || {};
          Object.entries(modelsForType).forEach(([modelId, modelDetails]) => {
            newModelHealth[modelId] = modelDetails;
            if (!allModelsFromApi.find(m => m.id === modelId)) {
              allModelsFromApi.push({
                id: modelId,
                name: baseModelInfo[modelId]?.name || modelId.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                description: baseModelInfo[modelId]?.description || 'Deepfake detection model',
                type: baseModelInfo[modelId]?.type || mediaType, 
                status: modelDetails?.status || 'unknown'
              });
            }
          });
        });
        
        setStackingAvailableByMediaType(newStackingStatus);
        setModelHealthStatus(newModelHealth);

        if (allModelsFromApi.length > 0) {
            setAvailableModels(allModelsFromApi);
        } else {
          console.warn("API health check returned no specific model statuses in media_type_details. Falling back to baseModelInfo for available models list.");
          allModelsFromApi = Object.keys(baseModelInfo).map(id => ({
            id, name: baseModelInfo[id]?.name || id, description: baseModelInfo[id]?.description || 'Model',
            type: baseModelInfo[id]?.type || 'unknown', status: 'unknown'
          }));
          setAvailableModels(allModelsFromApi);
        }
        
        const isSystemFullyHealthy = data.overall_api_status === 'healthy';
        if (isSystemFullyHealthy) nextInterval = POLLING_INTERVALS.STABLE_HEALTHY;
        else if (isInitialFastPollPhase) nextInterval = POLLING_INTERVALS.INITIAL_FAST;
      }
    } catch (err) {
      console.error('Health check fetch/processing exception:', err);
      if (!mountedRef.current) return;
      setStackingAvailableByMediaType({});
      const errorStatus = {};
      Object.keys(baseModelInfo).forEach(id => {
        errorStatus[id] = { status: 'error', message: 'Health check exception' };
      });
      setModelHealthStatus(errorStatus);
      setAvailableModels(Object.keys(baseModelInfo).map(id => ({
        id, name: baseModelInfo[id]?.name || id, description: baseModelInfo[id]?.description || 'Model',
        type: baseModelInfo[id]?.type || 'unknown', status: 'error'
      })));
    } finally {
      if (mountedRef.current) {
        setIsHealthCheckInProgress(false);
        if (healthCheckTimeoutRef.current) clearTimeout(healthCheckTimeoutRef.current);
        let nextPollPhase = isInitialFastPollPhase;
        if (isInitialFastPollPhase && nextInterval !== POLLING_INTERVALS.INITIAL_FAST) nextPollPhase = false;
        else if (nextInterval === POLLING_INTERVALS.STABLE_HEALTHY) nextPollPhase = false;
        healthCheckTimeoutRef.current = setTimeout(() => performAndScheduleHealthCheck(nextPollPhase), nextInterval);
      }
    }
  }, [baseModelInfo, isHealthCheckInProgress]);

  useEffect(() => {
    mountedRef.current = true;
    const initialTimeoutId = setTimeout(() => {
      if (mountedRef.current) performAndScheduleHealthCheck(true);
    }, POLLING_INTERVALS.FIRST_CHECK_DELAY);
    return () => {
      mountedRef.current = false;
      clearTimeout(initialTimeoutId);
      if (healthCheckTimeoutRef.current) clearTimeout(healthCheckTimeoutRef.current);
    };
  }, [performAndScheduleHealthCheck]);

  useEffect(() => {
    if (availableModels.length > 0) {
      const availableModelIds = availableModels.map(m => m.id);
      setSelectedModels(prevSelectedModels => 
        prevSelectedModels.filter(smId => availableModelIds.includes(smId))
      );
    }
  }, [availableModels]);

  const handleFileChange = useCallback((event) => {
    const file = event.target.files[0];
    setError(null); setResult(null); setModelProgress({});
    if (file) {
      let validMimeTypes, maxFileSize, mediaTypeName;
      switch(activeMediaType) {
        case 'video':
          validMimeTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska', 'video/x-m4v'];
          maxFileSize = 500 * 1024 * 1024; // 500MB for videos
          mediaTypeName = "video";
          break;
        case 'audio':
          validMimeTypes = ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/flac', 'audio/x-m4a'];
          maxFileSize = 200 * 1024 * 1024; // 200MB for audio
          mediaTypeName = "audio";
          break;
        case 'image':
        default:
          validMimeTypes = ['image/jpeg', 'image/png', 'image/webp'];
          maxFileSize = 100 * 1024 * 1024; // 100MB for images
          mediaTypeName = "image";
          break;
      }

      if (!validMimeTypes.includes(file.type)) {
        setError(`Invalid file type for ${mediaTypeName}. Supported: ${validMimeTypes.map(t => t.split('/')[1]).join(', ')}.`);
        showToast(`Invalid file type. Please select a valid ${mediaTypeName} file.`, 'error');
        setSelectedFile(null); setPreview(null); return;
      }
      if (file.size > maxFileSize) {
        setError(`File size exceeds ${maxFileSize / (1024*1024)}MB limit for ${mediaTypeName}.`);
        showToast(`File too large. Maximum size: ${maxFileSize / (1024*1024)}MB`, 'error');
        setSelectedFile(null); setPreview(null); return;
      }
      setSelectedFile(file);
      showToast(`${file.name} selected successfully`, 'success');
      
      if (activeMediaType === 'image' || activeMediaType === 'video') {
        const reader = new FileReader();
        reader.onloadend = () => setPreview(reader.result);
        reader.readAsDataURL(file);
      } else {
        setPreview(null); 
      }
    } else {
      setSelectedFile(null); setPreview(null);
    }
  }, [activeMediaType, showToast]);

  const handleDemoMedia = useCallback(async (demoMediaType, demoFileNameKey) => {
    setError(null); setResult(null); setModelProgress({}); setLoading(true); setProcessStage(`Loading demo ${demoMediaType}...`);
    
    const demoFilePaths = {
      image: {
        'real_portrait': '/demo_images/real_sample.jpg', 
        'ai_face_gan': '/demo_images/fake_sample_gan.jpg',
        'stylegan_city': '/demo_images/fake_sample_stylegan.jpg',
        'face_swap': '/demo_images/fake_sample_faceswap.jpg'
      },
      video: {
        'real_video_1': '/demo_videos/real/1.mp4',
        'real_video_2': '/demo_videos/real/2.mp4',
        'fake_video_1': '/demo_videos/fake/1.mp4',
        'fake_video_2': '/demo_videos/fake/2.mp4',
      }
    };

    let mediaPath = demoFilePaths[demoMediaType]?.[demoFileNameKey];
    
    if (!mediaPath) {
      if (demoMediaType === 'image') mediaPath = demoFilePaths.image.ai_face_gan;
      else if (demoMediaType === 'video') mediaPath = demoFilePaths.video.fake_video_1;
      else {
        setError(`Demo ${demoMediaType} '${demoFileNameKey}' not available.`);
        showToast(`Demo ${demoMediaType} not available`, 'error');
        setLoading(false); setProcessStage('');
        return;
      }
    }

    try {
      const response = await fetch(mediaPath);
      if (!response.ok) throw new Error(`Failed to fetch demo ${demoMediaType} from ${mediaPath} (${response.status})`);
      const blob = await response.blob();
      const fileExtension = mediaPath.split('.').pop() || (demoMediaType === 'image' ? 'jpg' : 'mp4');
      const fileNameForFileObject = `demo_${demoFileNameKey}.${fileExtension}`;
      const file = new File([blob], fileNameForFileObject, { type: blob.type || `${demoMediaType}/${fileExtension}` });
      
      setSelectedFile(file); 
      showToast(`Demo ${demoMediaType} loaded successfully`, 'success');
      if (demoMediaType === 'image' || demoMediaType === 'video') {
        const reader = new FileReader();
        reader.onloadend = () => { setPreview(reader.result); setLoading(false); setProcessStage(''); };
        reader.readAsDataURL(file);
      } else {
        setPreview(null);
        setLoading(false); setProcessStage('');
      }
    } catch (err) {
      console.error(`Error loading demo ${demoMediaType} (${mediaPath}):`, err);
      setError(`Failed to load demo ${demoMediaType}. Please try uploading manually.`);
      showToast(`Failed to load demo ${demoMediaType}`, 'error');
      setLoading(false); setProcessStage('');
    }
  }, [showToast]);

  const toggleSettings = () => setShowSettings(prev => !prev);
  const toggleDebugMode = () => setDebugMode(prev => !prev);
  const toggleModelDashboardExpand = () => setIsModelDashboardExpanded(prev => !prev);

  const toggleModel = (modelId) => {
    setSelectedModels(prevModels =>
      prevModels.includes(modelId) ? prevModels.filter(id => id !== modelId) : [...prevModels, modelId]
    );
  };

  const handleSubmit = async () => {
    if (!selectedFile) { 
      setError(`Please select an ${activeMediaType} file first.`); 
      showToast(`Please select a file to analyze`, 'error');
      return; 
    }
    setLoading(true); setError(null); setResult(null); setModelProgress({}); setProcessStage(`Preparing ${activeMediaType}...`);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('threshold', threshold.toString());
      formData.append('ensemble_method', ensembleMethod);
      
      const modelsRelevantToMediaType = availableModels.filter(m => m.type === activeMediaType);
      const activeAndSelectedModels = selectedModels.filter(modelId => {
        const model = modelsRelevantToMediaType.find(m => m.id === modelId);
        return model && (model.status === 'healthy' || model.status === 'loading' || model.status === 'unknown');
      });
      
      if (selectedModels.length > 0 && activeAndSelectedModels.length === 0) {
        console.warn(`User selected models but none are active/available for ${activeMediaType}. API will use its default for this type.`);
      } else if (activeAndSelectedModels.length > 0) {
        formData.append('models', activeAndSelectedModels.join(','));
      }
      
      setProcessStage(`Uploading & Analyzing ${activeMediaType}...`);

      const modelsForProgress = activeAndSelectedModels.length > 0 ? activeAndSelectedModels : 
                                modelsRelevantToMediaType
                                  .filter(m => m.status === 'healthy' || m.status === 'loading' || m.status === 'unknown')
                                  .map(m => m.id);

      const progressInterval = setInterval(() => {
        setModelProgress(prev => {
          const newProgress = { ...prev };
          modelsForProgress.forEach(modelId => {
            if (!newProgress[modelId] || newProgress[modelId].status !== 'completed') {
              if (!newProgress[modelId]) newProgress[modelId] = { status: 'processing', progress: 0 };
              else if (newProgress[modelId].progress < 90) newProgress[modelId].progress = Math.min(90, newProgress[modelId].progress + Math.random() * 15);
            }
          });
          return newProgress;
        });
      }, 500);

      const response = await fetch('/api/detect', { method: 'POST', body: formData });
      clearInterval(progressInterval);
      setProcessStage('Processing results...');
      const data = await response.json();

      if (!response.ok) {
        const errorDetail = data.detail || `Server error: ${response.status}`;
        if (typeof errorDetail === 'string' && errorDetail.toLowerCase().includes("unknown model(s) specified")) {
          console.warn("API reported unknown models. Forcing health check.");
          performAndScheduleHealthCheck(true);
          throw new Error("Model selection mismatch. Please try again. Available models list updated.");
        }
        throw new Error(errorDetail);
      }
      
      const finalProgress = {};
      const returnedModelKeys = data.model_results ? Object.keys(data.model_results) : modelsForProgress;
      returnedModelKeys.forEach(modelId => {
        finalProgress[modelId] = {
          status: data.model_results && data.model_results[modelId]?.error ? 'error' : 'completed',
          progress: 100
        };
      });
      modelsForProgress.forEach(modelId => {
        if (!finalProgress[modelId]) finalProgress[modelId] = { status: 'unknown_response', progress: 100 };
      });

      setModelProgress(finalProgress);
      setResult(data);
      const verdict = data.is_likely_deepfake ? 'AI-Generated' : 'Authentic';
      showToast(`Analysis complete: ${verdict} (${(data.deepfake_probability * 100).toFixed(0)}% confidence)`, data.is_likely_deepfake ? 'warning' : 'success', 5000);
    } catch (err) {
      console.error(`Error during ${activeMediaType} analysis:`, err);
      setError(err.message || `An unexpected error occurred during ${activeMediaType} analysis.`);
      showToast(err.message || 'Analysis failed', 'error');
      setModelProgress(prev => {
        const newProgress = {...prev};
        Object.keys(newProgress).forEach(modelId => {
          if (newProgress[modelId].status !== 'completed') newProgress[modelId] = { status: 'error', progress: newProgress[modelId].progress || 0};
        });
        return newProgress;
      });
    } finally {
      setLoading(false); setProcessStage('');
    }
  };

  const handleExportResults = (format) => {
    if (!result) return;
    if (format === 'json') {
      exportResults(result, selectedFile?.name || 'analysis');
      showToast('Results exported as JSON', 'success');
    } else if (format === 'pdf') {
      downloadPDF(result, selectedFile?.name || 'analysis', activeMediaType === 'image' ? preview : null);
      showToast('PDF report generated', 'success');
    }
  };
  
  const modelsForSettingsPanel = availableModels.filter(m => m.type === activeMediaType);

  return (
    <div className="flex flex-col min-h-screen bg-white dark:bg-neutral-900 transition-colors duration-200"> 
      <Header 
        showSettings={showSettings} 
        toggleSettings={toggleSettings}
        darkMode={darkMode}
        setDarkMode={setDarkMode}
        debugMode={debugMode}
        toggleDebugMode={toggleDebugMode}
      />
      <NavigationBar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        activeMediaType={activeMediaType}
        setActiveMediaType={(type) => {
          setActiveMediaType(type);
          setSelectedFile(null); 
          setPreview(null);
          setResult(null);
          setError(null);
        }}
      />
      <main className="flex-grow w-full max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        <ErrorBoundary showDetails={debugMode}>
          {showSettings && (
            <SettingsPanel
              threshold={threshold} setThreshold={setThreshold}
              ensembleMethod={ensembleMethod} setEnsembleMethod={setEnsembleMethod}
              selectedModels={selectedModels} setSelectedModels={setSelectedModels}
              toggleModel={toggleModel}
              availableModels={modelsForSettingsPanel} 
              modelHealthStatus={modelHealthStatus}
              stackingAvailable={stackingAvailableByMediaType[activeMediaType] || false} 
              darkMode={darkMode} setDarkMode={setDarkMode}
              debugMode={debugMode} setDebugMode={setDebugMode}
              onClose={toggleSettings}
              activeMediaType={activeMediaType}
            />
          )}
          {activeTab === 'detect' && (
            <div className="space-y-8">
              <ModelHealthDashboard 
                modelHealthStatus={modelHealthStatus} 
                availableModels={availableModels} 
                isExpanded={isModelDashboardExpanded} 
                toggleExpand={toggleModelDashboardExpand}
              />
              <UploadSection
                selectedFile={selectedFile} preview={preview}
                loading={loading} processStage={processStage} error={error}
                handleFileChange={handleFileChange} 
                handleDemoMedia={(demoFileNameKey) => handleDemoMedia(activeMediaType, demoFileNameKey)} 
                handleSubmit={handleSubmit}
                mediaType={activeMediaType}
              />
              {(loading || result) && (
                <ResultsSection 
                  result={result} loading={loading} modelProgress={modelProgress}
                  debugMode={debugMode} onExport={handleExportResults}
                  fileName={selectedFile?.name} preview={preview}
                  mediaType={activeMediaType}
                />
              )}
            </div>
          )}
          {activeTab === 'batch' && ( 
            <div className="bg-white dark:bg-neutral-800 rounded-xl shadow-lg p-6 sm:p-8 border border-neutral-200 dark:border-neutral-700 animate-fade-in">
              <h2 className="text-2xl font-semibold text-neutral-800 dark:text-neutral-100 mb-4">Batch Processing</h2>
              <p className="text-neutral-600 dark:text-neutral-400 mb-6">
                Efficiently analyze multiple files. This feature is currently under development for enterprise use.
              </p>
              <div className="p-6 bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 rounded-lg text-sm text-primary-700 dark:text-primary-300">
                <p className="font-medium">Interested in Batch Processing?</p>
                <p className="mt-1">Contact our team to learn more about high-volume analysis capabilities for your organization.</p>
                <button 
                    type="button" 
                    onClick={() => window.open('mailto:deepsafe.hq@gmail.com?subject=DeepSafe Enterprise Inquiry', '_blank')}
                    className="mt-4 px-4 py-2 bg-primary-600 text-white text-xs font-medium rounded-md hover:bg-primary-700 transition-all hover:scale-105 active:scale-100"
                >
                  Contact Us
                </button>
              </div>
            </div>
          )}
          {activeTab === 'about' && <InfoSection />}
        </ErrorBoundary>
      </main>
      <Footer />
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
    </div>
  );
};

export default App;