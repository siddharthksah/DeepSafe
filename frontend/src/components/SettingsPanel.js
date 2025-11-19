// frontend/src/components/SettingsPanel.js
import React, { useState } from 'react';
import { 
  Sliders, Users, LayoutGrid, Info, ChevronDown, Check, 
  AlertCircle, Zap, Moon, Sun, Bug, X, Palette
} from 'lucide-react';

const SettingsPanel = ({
  threshold,
  setThreshold,
  ensembleMethod,
  setEnsembleMethod,
  selectedModels,
  toggleModel,
  availableModels,
  modelHealthStatus,
  stackingAvailable,
  darkMode,
  setDarkMode,
  debugMode,
  setDebugMode,
  onClose
}) => {
  const [activeTab, setActiveTab] = useState('detection');
  const [showModelDetails, setShowModelDetails] = useState(true);

  const getModelStatusInfo = (modelId) => {
    const statusEntry = modelHealthStatus[modelId];
    if (!statusEntry || !statusEntry.status) {
      return { icon: <AlertCircle className="h-4 w-4 text-neutral-400" />, text: 'Status Unknown', color: 'text-neutral-400' };
    }
    switch (statusEntry.status) {
      case 'healthy':
        return { icon: <Check className="h-4 w-4 text-success-500" />, text: 'Active', color: 'text-success-500' };
      case 'loading':
        return { icon: <Zap className="h-4 w-4 text-warning-500 animate-pulse" />, text: 'Loading', color: 'text-warning-500' };
      case 'degraded_not_loaded':
      case 'degraded_components_not_loaded':
        return { icon: <AlertCircle className="h-4 w-4 text-warning-500" />, text: 'Degraded', color: 'text-warning-500' };
      default:
        return { icon: <X className="h-4 w-4 text-danger-500" />, text: 'Offline', color: 'text-danger-500' };
    }
  };
  
  const handleSelectAllModels = () => {
    const selectableModelIds = availableModels
        .filter(m => modelHealthStatus[m.id]?.status === 'healthy' || modelHealthStatus[m.id]?.status === 'loading' || !modelHealthStatus[m.id]?.status || modelHealthStatus[m.id]?.status === 'unknown')
        .map(m => m.id);
    const allSelectableAreSelected = selectableModelIds.length > 0 && 
        selectableModelIds.every(id => selectedModels.includes(id)) &&
        selectedModels.filter(id => selectableModelIds.includes(id)).length === selectableModelIds.length;
    if (allSelectableAreSelected) {
      selectableModelIds.forEach(id => { if (selectedModels.includes(id)) toggleModel(id); });
    } else {
      selectableModelIds.forEach(id => { if (!selectedModels.includes(id)) toggleModel(id); });
    }
  };
  
  const selectableModelsCount = availableModels.filter(m => modelHealthStatus[m.id]?.status === 'healthy' || modelHealthStatus[m.id]?.status === 'loading' || !modelHealthStatus[m.id]?.status || modelHealthStatus[m.id]?.status === 'unknown').length;
  const isAllSelectableModelsSelected = selectedModels.length > 0 &&
    selectedModels.length === selectableModelsCount &&
    availableModels
        .filter(m => modelHealthStatus[m.id]?.status === 'healthy' || modelHealthStatus[m.id]?.status === 'loading' || !modelHealthStatus[m.id]?.status || modelHealthStatus[m.id]?.status === 'unknown')
        .every(m => selectedModels.includes(m.id));
  const isNoModelsEffectivelySelected = selectedModels.length === 0;

  const getThresholdDescription = (value) => {
    if (value < 0.3) return "Very lenient (More likely to classify as Authentic)";
    if (value < 0.5) return "Lenient";
    if (value === 0.5) return "Balanced (Default)";
    if (value < 0.7) return "Strict";
    return "Very strict (More likely to classify as AI-Generated)";
  };

  const tabs = [
    { id: 'detection', label: 'Detection', icon: <Sliders size={16} /> },
    { id: 'models', label: 'Models', icon: <LayoutGrid size={16} /> },
    { id: 'appearance', label: 'Appearance', icon: <Palette size={16} /> },
  ];

  return (
    <div className="fixed inset-0 bg-black/30 dark:bg-black/50 z-50 flex items-center justify-center p-4 backdrop-blur-sm animate-fade-in">
      <div className="bg-white dark:bg-neutral-800 rounded-2xl shadow-2xl border border-neutral-200 dark:border-neutral-700 transition-all w-full max-w-3xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-800">
          <div className="flex items-center">
            <div className="p-2 bg-primary-100 dark:bg-primary-900/30 rounded-lg mr-3 overflow-hidden">
              <img 
                src="/assets/deepsafe.png" 
                alt="DeepSafe" 
                className="h-6 w-6 object-cover rounded-md"
              />
            </div>
            <h2 className="text-xl font-semibold text-neutral-800 dark:text-neutral-100">Settings</h2>
          </div>
          <button 
            onClick={onClose} 
            className="p-2 hover:bg-neutral-100 dark:hover:bg-neutral-700 rounded-lg transition-all duration-200 hover:scale-110 active:scale-95"
          >
            <X className="h-5 w-5 text-neutral-600 dark:text-neutral-400"/>
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="px-6 border-b border-neutral-200 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800/50">
          <nav className="flex -mb-px space-x-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  px-4 py-3 text-sm font-medium rounded-t-lg transition-all duration-200 flex items-center
                  ${activeTab === tab.id
                    ? 'bg-white dark:bg-neutral-800 text-primary-600 dark:text-primary-400 border-b-2 border-primary-500'
                    : 'text-neutral-500 dark:text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-300'
                  }
                `}
              >
                <span className={`mr-2 ${activeTab === tab.id ? 'text-primary-500' : 'text-neutral-400'}`}>
                  {tab.icon}
                </span>
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {activeTab === 'detection' && (
            <div className="space-y-6 animate-fade-in">
              {/* Threshold Setting */}
              <div className="bg-neutral-50 dark:bg-neutral-700/50 p-5 rounded-xl border border-neutral-200 dark:border-neutral-600">
                <label htmlFor="threshold-slider" className="block text-sm font-semibold text-neutral-700 dark:text-neutral-300 mb-4">
                  Detection Sensitivity
                </label>
                <div className="space-y-4">
                  <input 
                    id="threshold-slider" 
                    type="range" 
                    min="0.1" 
                    max="0.9" 
                    step="0.01" 
                    value={threshold} 
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    className="w-full h-2 bg-neutral-200 dark:bg-neutral-600 rounded-lg appearance-none cursor-pointer slider-thumb" 
                  />
                  <div className="flex justify-between text-xs text-neutral-500 dark:text-neutral-400">
                    <span>Lenient</span>
                    <span>Balanced</span>
                    <span>Strict</span>
                  </div>
                  <div className="text-center">
                    <div className="inline-flex items-center px-3 py-1.5 bg-primary-100 dark:bg-primary-900/30 rounded-lg">
                      <span className="text-sm font-semibold text-primary-700 dark:text-primary-300">
                        Threshold: {threshold.toFixed(2)}
                      </span>
                    </div>
                    <p className="text-xs text-neutral-600 dark:text-neutral-400 mt-2">{getThresholdDescription(threshold)}</p>
                  </div>
                </div>
              </div>

              {/* Ensemble Method */}
              <div className="bg-neutral-50 dark:bg-neutral-700/50 p-5 rounded-xl border border-neutral-200 dark:border-neutral-600">
                <label className="block text-sm font-semibold text-neutral-700 dark:text-neutral-300 mb-4 flex items-center">
                  <Users className="h-4 w-4 mr-2 text-neutral-500 dark:text-neutral-400" />
                  Ensemble Strategy
                </label>
                <div className="space-y-3">
                  <EnsembleButton 
                    currentMethod={ensembleMethod} 
                    setMethod={setEnsembleMethod} 
                    methodValue="stacking" 
                    label="Stacking (Meta Learner)" 
                    description="Uses ML to combine model predictions optimally"
                    isDefault={true} 
                    disabled={!stackingAvailable}
                    tooltip={!stackingAvailable ? "Stacking model not available on backend" : "Recommended for best accuracy"} 
                  />
                  <EnsembleButton 
                    currentMethod={ensembleMethod} 
                    setMethod={setEnsembleMethod} 
                    methodValue="voting" 
                    label="Voting" 
                    description="Simple majority vote from all models"
                    tooltip="Fast and interpretable" 
                  />
                  <EnsembleButton 
                    currentMethod={ensembleMethod} 
                    setMethod={setEnsembleMethod} 
                    methodValue="average" 
                    label="Average" 
                    description="Averages probability scores from models"
                    tooltip="Balanced approach" 
                  />
                </div>
              </div>
            </div>
          )}

          {activeTab === 'models' && (
            <div className="space-y-4 animate-fade-in">
              <div className="bg-neutral-50 dark:bg-neutral-700/50 p-5 rounded-xl border border-neutral-200 dark:border-neutral-600">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-sm font-semibold text-neutral-700 dark:text-neutral-300 flex items-center">
                    <LayoutGrid className="h-4 w-4 mr-2 text-neutral-500 dark:text-neutral-400" />
                    AI Models ({isNoModelsEffectivelySelected ? 'All Active' : `${selectedModels.length} Selected`})
                  </h3>
                  <button 
                    onClick={handleSelectAllModels} 
                    disabled={selectableModelsCount === 0 && availableModels.length > 0}
                    className="text-xs font-medium text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 transition-colors disabled:opacity-50"
                  >
                    {isAllSelectableModelsSelected || (isNoModelsEffectivelySelected && selectableModelsCount > 0) ? 'Deselect All' : 'Select All'}
                  </button>
                </div>
                
                <div className="space-y-2 max-h-96 overflow-y-auto pr-2 -mr-2">
                  {availableModels.length > 0 ? availableModels.map((model) => {
                    const statusInfo = getModelStatusInfo(model.id);
                    const isSelectable = modelHealthStatus[model.id]?.status === 'healthy' || 
                                       modelHealthStatus[model.id]?.status === 'loading' || 
                                       !modelHealthStatus[model.id]?.status || 
                                       modelHealthStatus[model.id]?.status === 'unknown';
                    return (
                      <label 
                        key={model.id} 
                        htmlFor={`model-${model.id}`}
                        className={`
                          flex items-center p-3 rounded-lg transition-all duration-200 border
                          ${isSelectable ? (
                            selectedModels.includes(model.id) 
                              ? 'bg-primary-50 dark:bg-primary-900/30 border-primary-300 dark:border-primary-700 shadow-sm' 
                              : 'bg-white dark:bg-neutral-600/30 border-neutral-200 dark:border-neutral-600 hover:border-primary-300 dark:hover:border-primary-600 hover:shadow-sm cursor-pointer'
                          ) : 'bg-neutral-100 dark:bg-neutral-700 border-neutral-200 dark:border-neutral-600 opacity-60 cursor-not-allowed'}
                        `}
                      >
                        <input 
                          id={`model-${model.id}`} 
                          type="checkbox" 
                          checked={selectedModels.includes(model.id)} 
                          onChange={() => isSelectable && toggleModel(model.id)} 
                          disabled={!isSelectable}
                          className="h-4 w-4 text-primary-600 dark:text-primary-400 border-neutral-300 dark:border-neutral-500 rounded focus:ring-primary-500 dark:focus:ring-primary-400 focus:ring-offset-1"
                        />
                        <div className="ml-3 flex-1">
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300">{model.name}</span>
                            <div className={`flex items-center space-x-1 ${statusInfo.color}`}>
                              {statusInfo.icon}
                              <span className="text-xs">{statusInfo.text}</span>
                            </div>
                          </div>
                          {model.description && (
                            <span className="text-xs text-neutral-500 dark:text-neutral-400 block mt-0.5">{model.description}</span>
                          )}
                        </div>
                      </label>
                    );
                  }) : (
                    <p className="text-sm text-neutral-500 dark:text-neutral-400 text-center py-8">
                      Loading available models...
                    </p>
                  )}
                </div>
                
                <div className="mt-4 p-3 bg-primary-50 dark:bg-primary-900/20 rounded-lg">
                  <p className="text-xs text-primary-700 dark:text-primary-300 flex items-start">
                    <Info size={14} className="mr-1.5 flex-shrink-0 mt-0.5" />
                    {isNoModelsEffectivelySelected 
                      ? "When no models are selected, all available models will be used automatically." 
                      : "Only selected models will be used for analysis. The system will handle any unavailable models gracefully."
                    }
                  </p>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'appearance' && (
            <div className="space-y-4 animate-fade-in">
              <div className="bg-neutral-50 dark:bg-neutral-700/50 p-5 rounded-xl border border-neutral-200 dark:border-neutral-600">
                <h3 className="text-sm font-semibold text-neutral-700 dark:text-neutral-300 mb-4">Theme Settings</h3>
                <div className="space-y-3">
                  <ToggleSwitch 
                    label="Dark Mode" 
                    description="Switch between light and dark themes"
                    Icon={darkMode ? Moon : Sun} 
                    enabled={darkMode} 
                    setEnabled={setDarkMode} 
                  />
                </div>
              </div>
              
              {process.env.NODE_ENV === 'development' && (
                <div className="bg-neutral-50 dark:bg-neutral-700/50 p-5 rounded-xl border border-neutral-200 dark:border-neutral-600">
                  <h3 className="text-sm font-semibold text-neutral-700 dark:text-neutral-300 mb-4">Developer Options</h3>
                  <div className="space-y-3">
                    <ToggleSwitch 
                      label="Debug Mode" 
                      description="Show additional debugging information"
                      Icon={Bug} 
                      enabled={debugMode} 
                      setEnabled={setDebugMode} 
                      accentColor="red" 
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const EnsembleButton = ({ currentMethod, setMethod, methodValue, label, description, isDefault = false, disabled = false, tooltip }) => (
  <button 
    type="button" 
    onClick={() => !disabled && setMethod(methodValue)} 
    disabled={disabled} 
    title={tooltip}
    className={`
      relative w-full p-4 rounded-lg text-left transition-all duration-200 border-2
      ${currentMethod === methodValue 
        ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 border-primary-500 dark:border-primary-400 shadow-md' 
        : disabled 
          ? 'bg-neutral-100 dark:bg-neutral-700 text-neutral-400 dark:text-neutral-500 border-neutral-200 dark:border-neutral-600 cursor-not-allowed' 
          : 'bg-white dark:bg-neutral-600 text-neutral-700 dark:text-neutral-300 border-neutral-200 dark:border-neutral-500 hover:border-primary-400 dark:hover:border-primary-500 hover:shadow-sm'
      }
    `}
  >
    <div className="flex items-start justify-between">
      <div>
        <div className="flex items-center">
          <span className="font-medium">{label}</span>
          {isDefault && !disabled && (
            <span className="ml-2 text-xs px-2 py-0.5 bg-primary-100 dark:bg-primary-800 text-primary-700 dark:text-primary-200 rounded-full">
              Recommended
            </span>
          )}
        </div>
        <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">{description}</p>
      </div>
      {disabled && <AlertCircle className="h-4 w-4 text-neutral-400 dark:text-neutral-500 flex-shrink-0 ml-2" />}
    </div>
    {currentMethod === methodValue && (
      <div className="absolute inset-0 rounded-lg ring-2 ring-primary-500 dark:ring-primary-400 ring-offset-2 dark:ring-offset-neutral-800"></div>
    )}
  </button>
);

const ToggleSwitch = ({ label, description, Icon, enabled, setEnabled, accentColor = "primary" }) => (
  <label className="flex items-center justify-between p-4 bg-white dark:bg-neutral-800 rounded-lg border border-neutral-200 dark:border-neutral-600 cursor-pointer hover:shadow-sm transition-all duration-200">
    <div className="flex items-center">
      <div className={`p-2 rounded-lg mr-3 ${enabled ? (accentColor === "red" ? 'bg-red-100 dark:bg-red-900/30' : 'bg-primary-100 dark:bg-primary-900/30') : 'bg-neutral-100 dark:bg-neutral-700'}`}>
        <Icon className={`h-5 w-5 ${enabled ? (accentColor === "red" ? 'text-red-600 dark:text-red-400' : 'text-primary-600 dark:text-primary-400') : 'text-neutral-500 dark:text-neutral-400'}`} />
      </div>
      <div>
        <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300">{label}</span>
        {description && <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">{description}</p>}
      </div>
    </div>
    <button 
      onClick={() => setEnabled(!enabled)}
      className={`
        relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200
        ${enabled ? (accentColor === "red" ? 'bg-red-600' : 'bg-primary-600') : 'bg-neutral-200 dark:bg-neutral-600'}
      `}
    >
      <span className={`
        inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-200
        ${enabled ? 'translate-x-6' : 'translate-x-1'}
      `} />
    </button>
  </label>
);

export default SettingsPanel;