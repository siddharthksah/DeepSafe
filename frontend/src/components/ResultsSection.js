// frontend/src/components/ResultsSection.js
import React, { useState, useEffect } from 'react';
import { 
  AlertTriangle, CheckCircle, BarChart3, Clock, Users, Info, ChevronDown, 
  Download, Copy, Shield, Activity, FileJson, FileText, AlertCircle as AlertCircleIcon, 
  ChevronUp, ListTree, PieChart as PieChartIcon, TrendingUp, Brain, Zap
} from 'lucide-react';
import { 
  formatProbability, 
  getProbabilityColorClasses,
  formatTime,
  getModelDisplayName
} from '../utils/formatters';
import VisualizationCard from './VisualizationCard';
import ModelResultsChart from './ModelResultsChart';
import DebugInfoPanel from './DebugInfoPanel.js';

const ResultsSection = ({ result, loading, modelProgress, debugMode, onExport, fileName, preview, mediaType }) => {
  const [showIndividualModelDetails, setShowIndividualModelDetails] = useState(false);
  const [showModelPredictionChart, setShowModelPredictionChart] = useState(false);
  const [showRawData, setShowRawData] = useState(false);
  const [copiedToClipboard, setCopiedToClipboard] = useState(false);
  const [showDetailedProgress, setShowDetailedProgress] = useState(false); 
  const [overallProgress, setOverallProgress] = useState(0); 
  const [activeTab, setActiveTab] = useState('summary');

  useEffect(() => {
    if (loading && modelProgress && Object.keys(modelProgress).length > 0) {
      const progresses = Object.values(modelProgress).map(p => p.progress || 0);
      const total = progresses.reduce((sum, p) => sum + p, 0);
      const avgProgress = progresses.length > 0 ? total / progresses.length : 0;
      setOverallProgress(Math.round(avgProgress));
    } else if (!loading) {
      setOverallProgress(0);
    }
  }, [loading, modelProgress]);

  const handleCopyResults = () => {
    if (!result) return;
    const summary = `DeepSafe Analysis Report\nFile: ${fileName || 'Unknown'} (${mediaType})\nVerdict: ${result.is_likely_deepfake ? 'AI-Generated (Fake)' : 'Authentic (Real)'}\nAI-Generated Probability: ${formatProbability(result.deepfake_probability)}\nMethod: ${result.ensemble_method_used || 'N/A'}\nModels Used: ${result.model_count || 0}\nRequest ID: ${result.request_id}`;
    navigator.clipboard.writeText(summary).then(() => {
      setCopiedToClipboard(true);
      setTimeout(() => setCopiedToClipboard(false), 2000);
    });
  };

  if (loading) {
    const activeModelCount = Object.keys(modelProgress).length;
    return (
      <section className="mt-10 bg-white dark:bg-neutral-800 rounded-xl shadow-lg p-6 sm:p-8 border border-neutral-200 dark:border-neutral-700 transition-all animate-fade-in">
        <div className="space-y-6">
          {/* Skeleton loader for title */}
          <div className="space-y-3">
            <div className="h-8 bg-gradient-to-r from-neutral-200 to-neutral-300 dark:from-neutral-700 dark:to-neutral-600 rounded-lg w-3/4 animate-shimmer"></div>
            <div className="h-4 bg-gradient-to-r from-neutral-200 to-neutral-300 dark:from-neutral-700 dark:to-neutral-600 rounded-md w-1/2 animate-shimmer"></div>
          </div>
          
          {/* Progress bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm font-medium">
              <span className="text-neutral-700 dark:text-neutral-300 flex items-center">
                <Activity className="h-5 w-5 mr-2 text-primary-600 dark:text-primary-400 animate-spin" />
                Analyzing {mediaType}...
              </span>
              <span className="text-primary-600 dark:text-primary-400 font-semibold">{overallProgress}%</span>
            </div>
            <div className="relative w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-3 overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-r from-primary-400/20 to-primary-600/20 dark:from-primary-400/10 dark:to-primary-600/10 animate-pulse"></div>
              <div
                className="relative bg-gradient-to-r from-primary-500 to-primary-600 dark:from-primary-400 dark:to-primary-500 h-3 rounded-full transition-all duration-500 ease-out shadow-sm"
                style={{ width: `${overallProgress}%` }}
              />
            </div>
          </div>
          
          {/* Model progress details */}
          <div className="border border-neutral-200 dark:border-neutral-600 rounded-lg overflow-hidden">
            <button
              onClick={() => setShowDetailedProgress(!showDetailedProgress)}
              className="w-full bg-neutral-50 dark:bg-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-600 px-4 py-3 flex justify-between items-center cursor-pointer transition-all duration-200 focus:outline-none"
            >
              <h3 className="text-sm font-medium text-neutral-700 dark:text-neutral-300 flex items-center">
                <Brain size={16} className="mr-2 text-neutral-500 dark:text-neutral-400"/>
                AI Models Processing ({activeModelCount > 0 ? activeModelCount : 'initializing'})
              </h3>
              {showDetailedProgress ? <ChevronUp size={20} className="text-neutral-500 dark:text-neutral-400"/> : <ChevronDown size={20} className="text-neutral-500 dark:text-neutral-400" /> }
            </button>
            {showDetailedProgress && (
              <div className="p-4 bg-white dark:bg-neutral-800 animate-slide-down space-y-3">
                {Object.entries(modelProgress).length > 0 ? Object.entries(modelProgress).map(([modelId, progress]) => (
                  <div key={modelId} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-neutral-600 dark:text-neutral-400 font-medium">{getModelDisplayName(modelId)}</span>
                      <span className={`text-xs font-medium ${
                        progress.status === 'completed' ? 'text-success-600 dark:text-success-400' : 
                        progress.status === 'error' ? 'text-danger-600 dark:text-danger-400' : 
                        'text-neutral-500 dark:text-neutral-500'
                      }`}>
                        {progress.status === 'completed' ? '✓ Complete' : 
                         progress.status === 'error' ? '✗ Failed' : 
                         `${Math.round(progress.progress || 0)}%`}
                      </span>
                    </div>
                    <div className="w-full bg-neutral-200 dark:bg-neutral-700 rounded-full h-1.5 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${
                          progress.status === 'error' ? 'bg-danger-500' : 
                          progress.status === 'completed' ? 'bg-success-500' : 
                          'bg-gradient-to-r from-primary-400 to-primary-500'
                        }`}
                        style={{ width: `${progress.progress || 0}%` }}/>
                    </div>
                  </div>
                )) : <p className="text-xs text-neutral-500 dark:text-neutral-400 py-2 text-center">Preparing models for analysis...</p>}
              </div>
            )}
          </div>
        </div>
      </section>
    );
  }

  if (!result) return null;

  const isFake = result.is_likely_deepfake;
  const ensembleProbFake = result.deepfake_probability;
  const ensembleMethodUsed = result.ensemble_method_used || 'N/A';
  
  let verdictText, verdictDetailedText, confidenceLevel;
  if (typeof ensembleProbFake !== 'number' || isNaN(ensembleProbFake)) {
      verdictText = 'Undetermined'; 
      verdictDetailedText = 'Analysis result is inconclusive.';
      confidenceLevel = 'low';
  } else if (isFake) {
    if (ensembleProbFake > 0.85) { 
      verdictText = `AI-Generated ${mediaType.charAt(0).toUpperCase() + mediaType.slice(1)}`; 
      verdictDetailedText = `Very high confidence in AI manipulation detection.`;
      confidenceLevel = 'very-high';
    } else if (ensembleProbFake > 0.6) { 
      verdictText = `Likely AI-Generated`; 
      verdictDetailedText = `Strong indicators of AI manipulation detected.`;
      confidenceLevel = 'high';
    } else { 
      verdictText = `Potentially AI-Generated`; 
      verdictDetailedText = `Some manipulation indicators found. Further review recommended.`;
      confidenceLevel = 'medium';
    }
  } else {
    const confidenceReal = 1 - ensembleProbFake;
    if (confidenceReal > 0.85) { 
      verdictText = `Authentic ${mediaType.charAt(0).toUpperCase() + mediaType.slice(1)}`; 
      verdictDetailedText = `Very high confidence in authenticity.`;
      confidenceLevel = 'very-high';
    } else if (confidenceReal > 0.6) { 
      verdictText = `Likely Authentic`; 
      verdictDetailedText = `Strong indicators of authenticity detected.`;
      confidenceLevel = 'high';
    } else { 
      verdictText = `Potentially Authentic`; 
      verdictDetailedText = `Few manipulation indicators found. Confidence is moderate.`;
      confidenceLevel = 'medium';
    }
  }
  
  const individualModelResults = result.model_results || {}; 
  const hasIndividualModelResults = Object.keys(individualModelResults).length > 0;

  const tabs = [
    { id: 'summary', label: 'Summary', icon: <TrendingUp size={16} /> },
    { id: 'details', label: 'Model Analysis', icon: <Brain size={16} /> },
    { id: 'visualizations', label: 'Visualizations', icon: <PieChartIcon size={16} /> },
  ];

  return (
    <section className="mt-10 bg-white dark:bg-neutral-800 rounded-xl shadow-lg overflow-hidden border border-neutral-200 dark:border-neutral-700 animate-fade-in transition-all">
      {/* Header with export options */}
      <div className="px-6 py-4 bg-gradient-to-r from-neutral-50 to-neutral-100 dark:from-neutral-800 dark:to-neutral-700 border-b border-neutral-200 dark:border-neutral-700">
        <div className="flex flex-wrap gap-2 items-center justify-between">
          <div className="flex items-center">
            <div className="overflow-hidden rounded-full mr-2">
              <img 
                src="/assets/deepsafe.png" 
                alt="DeepSafe" 
                className="h-6 w-6 object-cover"
              />
            </div>
            <h2 className="text-lg font-semibold text-neutral-800 dark:text-neutral-100">Analysis Results</h2>
          </div>
          <div className="flex flex-wrap gap-2">
            <button onClick={handleCopyResults} className="px-3 py-1.5 bg-white dark:bg-neutral-700 text-neutral-700 dark:text-neutral-300 rounded-lg text-sm hover:bg-neutral-100 dark:hover:bg-neutral-600 transition-all duration-200 flex items-center hover:shadow-md hover:scale-105 active:scale-95">
              {copiedToClipboard ? <CheckCircle className="h-4 w-4 mr-1 text-success-500" /> : <Copy className="h-4 w-4 mr-1" />} 
              {copiedToClipboard ? 'Copied!' : 'Copy'}
            </button>
            <button onClick={() => onExport('json')} className="px-3 py-1.5 bg-white dark:bg-neutral-700 text-neutral-700 dark:text-neutral-300 rounded-lg text-sm hover:bg-neutral-100 dark:hover:bg-neutral-600 transition-all duration-200 flex items-center hover:shadow-md hover:scale-105 active:scale-95">
              <FileJson className="h-4 w-4 mr-1" /> JSON
            </button>
            <button onClick={() => onExport('pdf')} className="px-3 py-1.5 bg-white dark:bg-neutral-700 text-neutral-700 dark:text-neutral-300 rounded-lg text-sm hover:bg-neutral-100 dark:hover:bg-neutral-600 transition-all duration-200 flex items-center hover:shadow-md hover:scale-105 active:scale-95">
              <FileText className="h-4 w-4 mr-1" /> PDF
            </button>
            {debugMode && (
              <button onClick={() => setShowRawData(!showRawData)} className="px-3 py-1.5 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded-lg text-sm hover:bg-red-200 dark:hover:bg-red-900/40 transition-all duration-200 flex items-center hover:shadow-md hover:scale-105 active:scale-95">
                <Info className="h-4 w-4 mr-1" /> {showRawData ? 'Hide' : 'Show'} Raw
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Main verdict banner */}
      <div className={`relative overflow-hidden px-6 py-6 text-white ${
        isFake ? 'bg-gradient-to-br from-danger-600 via-danger-500 to-danger-600 dark:from-danger-700 dark:via-danger-600 dark:to-danger-700' 
               : 'bg-gradient-to-br from-success-600 via-success-500 to-success-600 dark:from-success-700 dark:via-success-600 dark:to-success-700'
      }`}>
        <div className="absolute inset-0 bg-white/5 backdrop-blur-[1px]"></div>
        <div className="relative flex items-start sm:items-center">
          <div className={`p-3 rounded-full ${isFake ? 'bg-white/20' : 'bg-white/20'} mr-4 flex-shrink-0`}>
            {isFake ? <AlertTriangle className="h-8 w-8" /> : <CheckCircle className="h-8 w-8" />}
          </div>
          <div className="flex-1">
            <h2 className="text-2xl sm:text-3xl font-bold">{verdictText}</h2>
            <p className="text-sm sm:text-base opacity-95 mt-1">{verdictDetailedText}</p>
            <div className="flex flex-wrap items-center gap-4 mt-3">
              <div className="flex items-center">
                <span className="text-xs uppercase tracking-wider opacity-75 mr-2">Confidence:</span>
                <span className="text-lg font-bold">{formatProbability(ensembleProbFake)}</span>
              </div>
              <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                confidenceLevel === 'very-high' ? 'bg-white/30 text-white' :
                confidenceLevel === 'high' ? 'bg-white/25 text-white' :
                'bg-white/20 text-white/90'
              }`}>
                {confidenceLevel.replace('-', ' ').toUpperCase()}
              </div>
            </div>
            {result.ensemble_method_requested && result.ensemble_method_used !== result.ensemble_method_requested && (
              <p className="text-xs opacity-80 mt-3 flex items-center bg-white/10 rounded-md px-2 py-1 inline-flex">
                <AlertCircleIcon className="h-3 w-3 mr-1" />
                Used '{result.ensemble_method_used}' method (requested: '{result.ensemble_method_requested}')
              </p>
            )}
          </div>
        </div>
      </div>
      
      {/* Tab navigation */}
      <div className="border-b border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800">
        <nav className="flex -mb-px" aria-label="Tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                group relative px-6 py-3 text-sm font-medium transition-all duration-200 focus:outline-none
                ${activeTab === tab.id
                  ? 'text-primary-600 dark:text-primary-400 border-b-2 border-primary-500'
                  : 'text-neutral-500 dark:text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-700'
                }
              `}
            >
              <span className="flex items-center">
                <span className={`mr-2 ${activeTab === tab.id ? 'text-primary-500' : 'text-neutral-400'}`}>
                  {tab.icon}
                </span>
                {tab.label}
              </span>
            </button>
          ))}
        </nav>
      </div>

      {/* Tab content */}
      <div className="p-6">
        {activeTab === 'summary' && (
          <div className="space-y-6 animate-fade-in">
            {/* Key metrics */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <MetricCard 
                icon={<Clock className="text-primary-600 dark:text-primary-400" />} 
                title="Analysis Time" 
                value={formatTime(result.response_time || result.total_inference_time_seconds)} 
                subtext={`${result.model_count || 0} models processed`}
              />
              <MetricCard 
                icon={<Users className="text-primary-600 dark:text-primary-400" />} 
                title="Model Consensus" 
                value={`${result.fake_votes || 0} vs ${result.real_votes || 0}`} 
                subtext={`Fake vs Real votes`}
              />
              <MetricCard 
                icon={<Shield className="text-primary-600 dark:text-primary-400" />} 
                title="Detection Method" 
                value={ensembleMethodUsed.charAt(0).toUpperCase() + ensembleMethodUsed.slice(1)} 
                subtext={result.processing_mode || 'Standard processing'}
              />
            </div>

            {/* Confidence bar */}
            <div className="bg-neutral-50 dark:bg-neutral-700/50 rounded-lg border border-neutral-200 dark:border-neutral-600 p-6">
              <h3 className="text-sm font-semibold mb-4 text-neutral-800 dark:text-neutral-100 flex items-center">
                <BarChart3 className="h-5 w-5 mr-2 text-neutral-500 dark:text-neutral-400" /> 
                Authenticity Assessment
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between text-sm font-medium">
                  <span className="text-success-700 dark:text-success-400">Authentic</span>
                  <span className={`font-bold text-lg ${getProbabilityColorClasses(ensembleProbFake, 'text')}`}>
                    {formatProbability(ensembleProbFake)}
                  </span>
                  <span className="text-danger-700 dark:text-danger-400">AI-Generated</span>
                </div>
                <div className="relative h-8 w-full bg-gradient-to-r from-success-100 via-warning-100 to-danger-100 dark:from-success-900/30 dark:via-warning-900/30 dark:to-danger-900/30 rounded-full overflow-hidden shadow-inner">
                  <div 
                    className="absolute top-0 bottom-0 w-1 bg-neutral-800 dark:bg-neutral-200 shadow-lg transform -translate-x-1/2 transition-all duration-1000 ease-out"
                    style={{ left: `${ensembleProbFake * 100}%` }}
                  >
                    <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-neutral-800 dark:bg-neutral-200 text-white dark:text-neutral-800 px-2 py-1 rounded text-xs font-medium whitespace-nowrap">
                      {formatProbability(ensembleProbFake)}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'details' && hasIndividualModelResults && (
          <div className="space-y-6 animate-fade-in">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(individualModelResults).map(([modelId, modelRes]) => (
                <ModelResultCard key={modelId} modelId={modelId} modelRes={modelRes} />
              ))}
            </div>
            {result.request_id && (
              <div className="mt-6 pt-4 border-t border-neutral-200 dark:border-neutral-600">
                <p className="text-xs text-neutral-500 dark:text-neutral-400">
                  Request ID: <code className="font-mono bg-neutral-100 dark:bg-neutral-700 px-2 py-0.5 rounded">{result.request_id}</code>
                </p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'visualizations' && (
          <div className="space-y-6 animate-fade-in">
            {(result.fake_votes !== undefined && result.real_votes !== undefined) && (
              <VisualizationCard result={result} />
            )}
            {hasIndividualModelResults && (
              <div className="bg-white dark:bg-neutral-700/50 rounded-lg border border-neutral-200 dark:border-neutral-600 overflow-hidden">
                <ModelResultsChart modelResults={individualModelResults} />
              </div>
            )}
          </div>
        )}

        {debugMode && showRawData && (
          <div className="mt-6">
            <DebugInfoPanel result={result} />
          </div>
        )}
      </div>
    </section>
  );
};

const MetricCard = ({ icon, title, value, subtext }) => (
  <div className="bg-gradient-to-br from-neutral-50 to-neutral-100 dark:from-neutral-700/50 dark:to-neutral-700/30 p-5 rounded-lg border border-neutral-200 dark:border-neutral-600 hover:shadow-md transition-all duration-200">
    <div className="flex items-start space-x-3">
      <div className="flex-shrink-0 flex items-center justify-center h-10 w-10 rounded-lg bg-white dark:bg-neutral-800 shadow-sm">
        {React.cloneElement(icon, { size: 20 })}
      </div>
      <div className="flex-1">
        <p className="text-xs uppercase text-neutral-500 dark:text-neutral-400 font-medium tracking-wider">{title}</p>
        <p className="text-lg font-bold text-neutral-800 dark:text-neutral-100 mt-1">{value}</p>
        {subtext && <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">{subtext}</p>}
      </div>
    </div>
  </div>
);

const ModelResultCard = ({ modelId, modelRes }) => {
  const modelName = getModelDisplayName(modelId);
  
  return (
    <div className="group p-4 rounded-lg bg-gradient-to-br from-neutral-50 to-white dark:from-neutral-700/80 dark:to-neutral-700/50 border border-neutral-200 dark:border-neutral-600 hover:shadow-md transition-all duration-200">
      <div className="flex justify-between items-start mb-3">
        <h4 className="font-medium text-neutral-800 dark:text-neutral-200 text-sm">
          {modelName}
        </h4>
        {!modelRes.error && modelRes.class && (
          <span className={`text-xs px-2.5 py-1 rounded-full font-semibold
            ${modelRes.class === 'fake' 
              ? 'bg-danger-100 dark:bg-danger-900/40 text-danger-700 dark:text-danger-300' 
              : 'bg-success-100 dark:bg-success-900/40 text-success-700 dark:text-success-300'}`}>
           {modelRes.class?.toUpperCase()}
         </span>
        )}
      </div>
      
      {modelRes.error ? (
        <div className="flex items-start text-sm text-danger-600 dark:text-danger-400">
          <AlertCircleIcon size={16} className="mr-1 mt-0.5 flex-shrink-0"/> 
          <span>Error: {modelRes.error}</span>
        </div>
      ) : (
        <>
          <div className="space-y-2">
            <div className="flex justify-between text-xs text-neutral-600 dark:text-neutral-400">
              <span>Authentic</span>
              <span className="font-medium text-neutral-700 dark:text-neutral-300">
                {formatProbability(modelRes.probability)}
              </span>
              <span>AI-Generated</span>
            </div>
            <div className="relative h-2 w-full bg-gradient-to-r from-success-100 to-danger-100 dark:from-success-900/30 dark:to-danger-900/30 rounded-full overflow-hidden">
              <div 
                className={`absolute top-0 bottom-0 left-0 h-full transition-all duration-1000 ease-out ${getProbabilityColorClasses(modelRes.probability, 'bg')}`}
                style={{ width: `${(modelRes.probability || 0) * 100}%` }}
              />
            </div>
          </div>
          {modelRes.inference_time !== undefined && (
            <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-3 flex items-center">
              <Zap size={12} className="mr-1"/> 
              Processing time: {formatTime(modelRes.inference_time)}
            </p>
          )}
        </>
      )}
    </div>
  );
};

export default ResultsSection;