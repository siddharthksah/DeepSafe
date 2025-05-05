import React from 'react';

const SettingsPanel = ({ 
  threshold, 
  setThreshold, 
  ensembleMethod, 
  setEnsembleMethod, 
  selectedModels, 
  toggleModel,
  availableModels 
}) => {
  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-8 border border-gray-200">
      <h2 className="text-lg font-medium mb-4 text-gray-900">Detection Settings</h2>
      
      <div className="space-y-6">
        {/* Threshold Setting */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Detection Threshold: {threshold}
          </label>
          <input 
            type="range" 
            min="0.1" 
            max="0.9" 
            step="0.05"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>More Lenient (0.1)</span>
            <span>Default (0.5)</span>
            <span>More Strict (0.9)</span>
          </div>
        </div>
        
        {/* Ensemble Method */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Ensemble Method
          </label>
          <div className="flex space-x-4">
            <label className="inline-flex items-center">
              <input
                type="radio"
                className="form-radio text-indigo-600"
                name="ensemble-method"
                value="voting"
                checked={ensembleMethod === 'voting'}
                onChange={() => setEnsembleMethod('voting')}
              />
              <span className="ml-2 text-gray-700">Voting (Majority)</span>
            </label>
            <label className="inline-flex items-center">
              <input
                type="radio"
                className="form-radio text-indigo-600"
                name="ensemble-method"
                value="average"
                checked={ensembleMethod === 'average'}
                onChange={() => setEnsembleMethod('average')}
              />
              <span className="ml-2 text-gray-700">Averaging (Mean)</span>
            </label>
          </div>
          <p className="text-xs text-gray-500 mt-1">
            Voting counts models' verdicts, while averaging uses probability scores
          </p>
        </div>
        
        {/* Model Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Select Models {selectedModels.length > 0 ? `(${selectedModels.length} selected)` : '(All)'}
          </label>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
            {availableModels.map(model => (
              <div key={model.id} className="flex items-start">
                <div className="flex items-center h-5">
                  <input
                    id={`model-${model.id}`}
                    type="checkbox"
                    checked={selectedModels.includes(model.id)}
                    onChange={() => toggleModel(model.id)}
                    className="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300 rounded"
                  />
                </div>
                <div className="ml-3 text-sm">
                  <label htmlFor={`model-${model.id}`} className="font-medium text-gray-700">{model.name}</label>
                  <p className="text-gray-500">{model.description}</p>
                </div>
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Leave all unchecked to use all available models
          </p>
        </div>
      </div>
    </div>
  );
};

export default SettingsPanel;