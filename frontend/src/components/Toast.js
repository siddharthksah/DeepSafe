// frontend/src/components/Toast.js
import React, { useEffect } from 'react';
import { CheckCircle, AlertCircle, AlertTriangle, Info, X } from 'lucide-react';

const Toast = ({ message, type = 'success', onClose }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      onClose();
    }, 5000);

    return () => clearTimeout(timer);
  }, [onClose]);

  const getIcon = () => {
    switch (type) {
      case 'success':
        return <CheckCircle className="h-5 w-5" />;
      case 'error':
        return <AlertCircle className="h-5 w-5" />;
      case 'warning':
        return <AlertTriangle className="h-5 w-5" />;
      case 'info':
      default:
        return <Info className="h-5 w-5" />;
    }
  };

  const getStyles = () => {
    switch (type) {
      case 'success':
        return 'bg-success-50 dark:bg-success-900/30 text-success-800 dark:text-success-200 border-success-200 dark:border-success-800';
      case 'error':
        return 'bg-danger-50 dark:bg-danger-900/30 text-danger-800 dark:text-danger-200 border-danger-200 dark:border-danger-800';
      case 'warning':
        return 'bg-warning-50 dark:bg-warning-900/30 text-warning-800 dark:text-warning-200 border-warning-200 dark:border-warning-800';
      case 'info':
      default:
        return 'bg-primary-50 dark:bg-primary-900/30 text-primary-800 dark:text-primary-200 border-primary-200 dark:border-primary-800';
    }
  };

  const getIconStyles = () => {
    switch (type) {
      case 'success':
        return 'text-success-600 dark:text-success-400';
      case 'error':
        return 'text-danger-600 dark:text-danger-400';
      case 'warning':
        return 'text-warning-600 dark:text-warning-400';
      case 'info':
      default:
        return 'text-primary-600 dark:text-primary-400';
    }
  };

  return (
    <div className="fixed top-4 right-4 z-50 animate-toast-slide-in">
      <div className={`
        flex items-center p-4 rounded-lg shadow-lg border backdrop-blur-sm
        ${getStyles()}
        min-w-[300px] max-w-md
        transform transition-all duration-300 hover:scale-105
      `}>
        <div className={`flex-shrink-0 ${getIconStyles()}`}>
          {getIcon()}
        </div>
        <div className="ml-3 flex-1">
          <p className="text-sm font-medium">
            {message}
          </p>
        </div>
        <button
          onClick={onClose}
          className="ml-4 flex-shrink-0 inline-flex text-current opacity-70 hover:opacity-100 focus:outline-none transition-opacity"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
};

export default Toast;