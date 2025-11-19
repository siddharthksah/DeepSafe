// frontend/src/components/HistoryPanel.js
import React from 'react';
import { X, Clock, Download, Trash2, ChevronRight } from 'lucide-react';
import { formatTime } from '../utils/formatters';

const HistoryPanel = ({ history, onItemClick, onClearHistory, onClose }) => {
  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)} hour${Math.floor(diffMins / 60) > 1 ? 's' : ''} ago`;
    
    return date.toLocaleDateString();
  };

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex justify-end animate-fade-in">
      <div className="bg-white dark:bg-neutral-800 w-full max-w-md h-full shadow-2xl animate-slide-in-right overflow-hidden flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-neutral-200 dark:border-neutral-700 flex items-center justify-between">
          <h2 className="text-xl font-semibold text-neutral-800 dark:text-neutral-100 flex items-center">
            <Clock className="h-5 w-5 mr-2 text-primary-600 dark:text-primary-400" />
            Analysis History
          </h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-neutral-100 dark:hover:bg-neutral-700 rounded-lg transition-colors"
          >
            <X className="h-5 w-5 text-neutral-500 dark:text-neutral-400" />
          </button>
        </div>

        {/* History List */}
        <div className="flex-1 overflow-y-auto">
          {history.length === 0 ? (
            <div className="p-6 text-center text-neutral-500 dark:text-neutral-400">
              <Clock className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No analysis history yet</p>
              <p className="text-sm mt-1">Your recent analyses will appear here</p>
            </div>
          ) : (
            <div className="divide-y divide-neutral-200 dark:divide-neutral-700">
              {history.map((item) => (
                <button
                  key={item.id}
                  onClick={() => onItemClick(item)}
                  className="w-full p-4 hover:bg-neutral-50 dark:hover:bg-neutral-700/50 transition-colors flex items-center space-x-4"
                >
                  {/* Thumbnail */}
                  {item.thumbnail && (
                    <div className="flex-shrink-0 w-16 h-16 rounded-lg overflow-hidden bg-neutral-100 dark:bg-neutral-700">
                      <img
                        src={item.thumbnail}
                        alt={item.fileName}
                        className="w-full h-full object-cover"
                      />
                    </div>
                  )}
                  
                  {/* Details */}
                  <div className="flex-1 text-left">
                    <p className="text-sm font-medium text-neutral-800 dark:text-neutral-100 truncate">
                      {item.fileName}
                    </p>
                    <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1">
                      {formatDate(item.timestamp)}
                    </p>
                    <div className="flex items-center mt-2 space-x-4">
                      <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                        item.verdict === 'fake' 
                          ? 'bg-danger-100 dark:bg-danger-900/30 text-danger-700 dark:text-danger-300' 
                          : 'bg-success-100 dark:bg-success-900/30 text-success-700 dark:text-success-300'
                      }`}>
                        {item.verdict === 'fake' ? 'AI-Generated' : 'Authentic'}
                      </span>
                      <span className="text-xs text-neutral-500 dark:text-neutral-400">
                        {(item.confidence * 100).toFixed(0)}% confidence
                      </span>
                    </div>
                  </div>
                  
                  <ChevronRight className="h-4 w-4 text-neutral-400" />
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        {history.length > 0 && (
          <div className="p-4 border-t border-neutral-200 dark:border-neutral-700">
            <button
              onClick={onClearHistory}
              className="w-full px-4 py-2 bg-danger-50 dark:bg-danger-900/20 text-danger-600 dark:text-danger-400 rounded-lg hover:bg-danger-100 dark:hover:bg-danger-900/30 transition-colors flex items-center justify-center"
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Clear History
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default HistoryPanel;