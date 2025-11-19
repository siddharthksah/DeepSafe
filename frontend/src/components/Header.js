// frontend/src/components/Header.js
import React from 'react';
import { Settings, X, Bug, Moon, Sun } from 'lucide-react';

const Header = ({ 
  showSettings, 
  toggleSettings, 
  darkMode,
  setDarkMode,
  debugMode,
  toggleDebugMode
}) => {
  return (
    <header className="bg-gradient-to-r from-primary-700 via-primary-600 to-primary-700 dark:from-primary-900 dark:via-primary-800 dark:to-primary-900 text-white shadow-lg z-50 transition-all duration-300">
      <div className="relative overflow-hidden">
        {/* Background pattern */}
        <div className="absolute inset-0 bg-white/5 backdrop-blur-[2px]"></div>
        <div className="absolute inset-0 bg-gradient-to-br from-transparent via-white/5 to-transparent"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-5">
            <div className="flex items-center space-x-4">
              <div className="relative group">
                <div className="absolute inset-0 bg-white/20 rounded-full blur-xl group-hover:blur-2xl transition-all duration-300"></div>
                <div className="relative rounded-full shadow-xl transform transition-all duration-300 group-hover:scale-110 overflow-hidden">
                  <img 
                    src="/assets/deepsafe.png" 
                    alt="DeepSafe Logo" 
                    className="h-11 w-11 object-cover rounded-full"
                  />
                </div>
              </div>
              <div>
                <h1 className="text-3xl font-bold tracking-tight flex items-baseline">
                  DeepSafe
                  <span className="ml-2 text-xs font-normal bg-white/20 px-2 py-0.5 rounded-full">Beta</span>
                </h1>
                <p className="text-sm text-primary-100 dark:text-primary-200 opacity-90 mt-0.5">
                  Advanced AI Deepfake Detection
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={() => setDarkMode(!darkMode)}
                className="relative p-2.5 rounded-lg hover:bg-white/10 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-white/20 group"
                aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                <div className="absolute inset-0 bg-white/0 group-hover:bg-white/10 rounded-lg transition-colors duration-200"></div>
                <div className="relative transform transition-transform duration-200 group-hover:scale-110 group-active:scale-95">
                  {darkMode ? <Sun size={22} /> : <Moon size={22} />}
                </div>
              </button>

              {process.env.NODE_ENV === 'development' && (
                <button
                  onClick={toggleDebugMode}
                  className={`
                    relative p-2.5 rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-white/20 group
                    ${debugMode ? 'bg-red-500/20 hover:bg-red-500/30' : 'hover:bg-white/10'}
                  `}
                  aria-label={debugMode ? 'Disable debug mode' : 'Enable debug mode'}
                >
                  <div className="relative transform transition-transform duration-200 group-hover:scale-110 group-active:scale-95">
                    <Bug size={22} className={debugMode ? 'text-red-200' : 'text-white'} />
                  </div>
                  {debugMode && (
                    <span className="absolute -top-1 -right-1 h-3 w-3 bg-red-400 rounded-full animate-pulse"></span>
                  )}
                </button>
              )}

              <button
                onClick={toggleSettings}
                className={`
                  relative p-2.5 rounded-lg focus:outline-none focus:ring-2 focus:ring-white/20 transition-all duration-200 group
                  ${showSettings ? 'bg-white/20 hover:bg-white/30' : 'hover:bg-white/10'}
                `}
                aria-label={showSettings ? 'Hide settings' : 'Show settings'}
              >
                <div className="relative transform transition-all duration-300 group-hover:scale-110 group-active:scale-95">
                  {showSettings ? (
                    <X size={22} className="transform rotate-0 transition-transform duration-300" />
                  ) : (
                    <Settings size={22} className="transform rotate-0 hover:rotate-90 transition-transform duration-300" />
                  )}
                </div>
              </button>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;