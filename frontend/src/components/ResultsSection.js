import React from 'react';
import { AlertTriangle, CheckCircle, BarChart3 } from 'lucide-react';

const ResultsSection = ({ result }) => {
  // Format probability as percentage
  const formatProbability = (prob) => {
    return (prob * 100).toFixed(1) + '%';
  };

  // Get color based on probability
  const getProbabilityColor = (prob) => {
    if (prob < 0.3) return 'bg-green-500'; // Likely real
    if (prob < 0.7) return 'bg-amber-500'; // Uncertain
    return 'bg-red-500'; // Likely fake
  };
  
  // Get text color for probability
  const getProbabilityTextColor = (prob) => {
    if (prob < 0.3) return 'text-green-600'; // Likely real
    if (prob < 0.7) return 'text-amber-600'; // Uncertain
    return 'text-red-600'; // Likely fake
  };

  if (!result) return null;

  return (
    <section className="mt-8 bg-white rounded-lg shadow-md p-6 border border-gray-200">
      <h2 className="text-lg font-medium mb-6 text-gray-900">Analysis Results</h2>
      
      <div className="space-y-8">
        {/* Verdict Card */}
        <div className={`rounded-lg p-6 shadow-sm border ${result.is_likely_deepfake ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'}`}>
          <div className="flex items-center">
            {result.is_likely_deepfake ? (
              <AlertTriangle className="h-8 w-8 text-red-600 mr-3" />
            ) : (
              <CheckCircle className="h-8 w-8 text-green-600 mr-3" />
            )}
            <div>
              <h3 className={`text-xl font-bold ${result.is_likely_deepfake ? 'text-red-900' : 'text-green-900'}`}>
                {result.is_likely_deepfake ? 'Likely Fake Image' : 'Likely Authentic Image'}
              </h3>
              <p className="text-sm mt-1">
                {result.is_likely_deepfake 
                  ? `This image shows signs of manipulation with ${formatProbability(result.deepfake_probability)} confidence` 
                  : `This image appears to be authentic with ${formatProbability(1 - result.deepfake_probability)} confidence`}
              </p>
            </div>
          </div>
        </div>
        
        {/* Probability Visualization */}
        <div>
          <h3 className="text-md font-medium mb-3 text-gray-900 flex items-center">
            <BarChart3 className="h-5 w-5 mr-2 text-gray-500" />
            Deepfake Probability
          </h3>
          
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Authentic</span>
              <span className={getProbabilityTextColor(result.deepfake_probability)}>
                {formatProbability(result.deepfake_probability)}
              </span>
              <span>Fake</span>
            </div>
            
            <div className="h-4 w-full bg-gray-200 rounded-full overflow-hidden">
              <div 
                className={`h-full rounded-full ${getProbabilityColor(result.deepfake_probability)} transition-all duration-500 ease-out`}
                style={{ width: `${result.deepfake_probability * 100}%` }}
              />
            </div>
            
            <div className="flex justify-between text-xs text-gray-500">
              <span>0%</span>
              <span>50%</span>
              <span>100%</span>
            </div>
          </div>
        </div>
        
        {/* Models Statistics */}
        <div>
          <h3 className="text-md font-medium mb-3 text-gray-900">Model Statistics</h3>
          
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
              <div className="text-xs uppercase text-gray-500 font-medium">Detection Time</div>
              <div className="text-lg font-bold">{result.response_time.toFixed(2)}s</div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
              <div className="text-xs uppercase text-gray-500 font-medium">Total Models</div>
              <div className="text-lg font-bold">{result.model_count}</div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
              <div className="text-xs uppercase text-gray-500 font-medium">Model Votes</div>
              <div className="flex items-center space-x-3">
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-red-500 mr-1"></div>
                  <span className="text-sm font-medium">{result.fake_votes} Fake</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-green-500 mr-1"></div>
                  <span className="text-sm font-medium">{result.real_votes} Real</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Request ID for reference */}
        <div className="text-xs text-gray-500 pt-2 border-t border-gray-200">
          Request ID: {result.request_id}
        </div>
      </div>
    </section>
  );
};

export default ResultsSection;