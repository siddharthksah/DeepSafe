// frontend/src/components/ModelHealthDashboard.js
import React from 'react';
import { Activity, CheckCircle, AlertCircle, XCircle, Loader, ShieldOff, ChevronDown, ChevronUp } from 'lucide-react'; 

const ModelHealthDashboard = ({ modelHealthStatus, availableModels, isExpanded, toggleExpand }) => {
  
  const getStatusDetails = (statusEntry) => {
    if (!statusEntry || !statusEntry.status) {
      return { icon: <AlertCircle className="h-3 w-3 text-neutral-400" />, text: 'Unknown', color: 'text-neutral-400' };
    }
    switch (statusEntry.status) {
      case 'healthy':
        return { icon: <CheckCircle className="h-3 w-3 text-success-500" />, text: 'Operational', color: 'text-success-500' };
      case 'loading':
        return { icon: <Loader className="h-3 w-3 text-warning-500 animate-spin" />, text: 'Loading...', color: 'text-warning-500' };
      case 'error_missing_weights':
      case 'missing_weights':
        return { icon: <XCircle className="h-3 w-3 text-danger-500" />, text: 'Weights Missing', color: 'text-danger-500' };
      case 'error':
         return { icon: <XCircle className="h-3 w-3 text-danger-500" />, text: 'Error', color: 'text-danger-500' };
      case 'degraded_not_loaded':
      case 'degraded_components_not_loaded':
        return { icon: <ShieldOff className="h-3 w-3 text-warning-500" />, text: 'Degraded', color: 'text-warning-500' }; 
      default:
        return { icon: <AlertCircle className="h-3 w-3 text-neutral-400" />, text: statusEntry.status, color: 'text-neutral-400' };
    }
  };

  const healthyCount = availableModels.filter(m => modelHealthStatus[m.id]?.status === 'healthy').length;
  const totalConfiguredModels = availableModels.length;

  let overallStatusColor = 'text-success-500';
  let statusBgColor = 'bg-success-500/10 dark:bg-success-500/20';
  let statusBorderColor = 'border-success-500/20';
  
  if (totalConfiguredModels > 0) {
      if (healthyCount < totalConfiguredModels && healthyCount > 0) {
        overallStatusColor = 'text-warning-500';
        statusBgColor = 'bg-warning-500/10 dark:bg-warning-500/20';
        statusBorderColor = 'border-warning-500/20';
      } else if (healthyCount === 0) {
        overallStatusColor = 'text-danger-500';
        statusBgColor = 'bg-danger-500/10 dark:bg-danger-500/20';
        statusBorderColor = 'border-danger-500/20';
      }
  } else {
      overallStatusColor = 'text-neutral-500';
      statusBgColor = 'bg-neutral-500/10 dark:bg-neutral-500/20';
      statusBorderColor = 'border-neutral-500/20';
  }

  return (
    <div className={`${statusBgColor} ${statusBorderColor} backdrop-blur-sm border rounded-xl p-3 sm:p-4 transition-all duration-300 hover:shadow-md bg-white/50 dark:bg-neutral-800/50`}>
      <div className="flex items-center justify-between cursor-pointer" onClick={toggleExpand}>
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-lg ${statusBgColor} ${overallStatusColor}`}>
            <Activity className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h3 className="text-sm font-medium text-neutral-700 dark:text-neutral-200">
                System Status
              </h3>
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusBgColor} ${overallStatusColor}`}>
                {healthyCount}/{totalConfiguredModels} Models Active
              </span>
            </div>
          </div>
        </div>
        <button
          className={`p-1.5 rounded-md hover:bg-neutral-100 dark:hover:bg-neutral-700 transition-all duration-200 text-neutral-500 dark:text-neutral-400 hover:scale-110`}
          aria-label={isExpanded ? "Collapse model details" : "Expand model details"}
        >
          {isExpanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </button>
      </div>
      
      {isExpanded && (
        <div className="mt-4 pt-4 border-t border-neutral-200/50 dark:border-neutral-700/50 animate-slide-down">
          {availableModels.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
              {availableModels.map(model => {
                const statusDetails = getStatusDetails(modelHealthStatus[model.id]);
                const modelInfo = modelHealthStatus[model.id];
                const tooltipMessage = modelInfo?.message || model.description || statusDetails.text;
                return (
                  <div
                    key={model.id}
                    className="group relative flex items-center justify-between p-2.5 bg-white/50 dark:bg-neutral-800/50 rounded-lg border border-neutral-200/50 dark:border-neutral-700/50 hover:shadow-sm transition-all duration-200"
                    title={tooltipMessage}
                  >
                    <div className="flex items-center space-x-2.5">
                      <div className={`p-1.5 rounded-md ${statusDetails.color === 'text-success-500' ? 'bg-success-100 dark:bg-success-900/30' : statusDetails.color === 'text-warning-500' ? 'bg-warning-100 dark:bg-warning-900/30' : 'bg-danger-100 dark:bg-danger-900/30'}`}>
                        {statusDetails.icon}
                      </div>
                      <div>
                        <p className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
                          {model.name}
                        </p>
                        <p className={`text-[10px] ${statusDetails.color}`}>
                          {statusDetails.text}
                        </p>
                      </div>
                    </div>
                    {modelInfo?.device && (
                      <span className="text-[10px] px-1.5 py-0.5 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-full">
                        {modelInfo.device.toUpperCase()}
                      </span>
                    )}
                    {/* Tooltip on hover */}
                    <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-neutral-900 text-white text-[10px] rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none whitespace-nowrap z-10">
                      {tooltipMessage}
                      <div className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1 w-0 h-0 border-4 border-transparent border-t-neutral-900"></div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-xs text-neutral-500 dark:text-neutral-400 py-2 text-center">Initializing model health monitoring...</p>
          )}
        </div>
      )}
    </div>
  );
};

export default ModelHealthDashboard;