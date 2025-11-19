// frontend/src/components/NavigationBar.js
import React from 'react';
import { Image as ImageIcon, Video, Mic, ListChecks, Info } from 'lucide-react';

const NavigationBar = ({ activeTab, setActiveTab, activeMediaType, setActiveMediaType }) => {
  const mainTabs = [
    { id: 'detect', label: 'Detect Media', icon: <ImageIcon size={18} /> },
    { id: 'batch', label: 'Batch Analysis', icon: <ListChecks size={18} /> },
    { id: 'about', label: 'About Technology', icon: <Info size={18} /> },
  ];

  const mediaTypeTabs = [
    { id: 'image', label: 'Image', icon: <ImageIcon size={16} /> },
    { id: 'video', label: 'Video', icon: <Video size={16} /> },
    { id: 'audio', label: 'Audio', icon: <Mic size={16} />, disabled: true },
  ];

  return (
    <div className="bg-white dark:bg-neutral-800 backdrop-blur-sm shadow-md z-40 transition-all duration-300 border-b border-neutral-200/50 dark:border-neutral-700/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col sm:flex-row justify-between items-center">
          {/* Main Navigation Tabs */}
          <nav className="flex -mb-px space-x-2 sm:space-x-4" aria-label="Main Tabs">
            {mainTabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  group inline-flex items-center py-3 sm:py-4 px-3 border-b-2 font-medium text-sm
                  focus:outline-none transition-all duration-200 ease-out hover:scale-105 active:scale-95
                  ${
                    activeTab === tab.id
                      ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                      : 'border-transparent text-neutral-500 dark:text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-300 hover:border-neutral-300 dark:hover:border-neutral-600'
                  }
                `}
                aria-current={activeTab === tab.id ? 'page' : undefined}
              >
                <span className={`mr-2 transition-transform duration-200 ${activeTab === tab.id ? 'text-primary-500 dark:text-primary-400 scale-110' : 'text-neutral-400 dark:text-neutral-500 group-hover:text-neutral-500 dark:group-hover:text-neutral-400 group-hover:scale-110'}`}>
                  {tab.icon}
                </span>
                <span className="relative">
                  {tab.label}
                  {activeTab === tab.id && (
                    <span className="absolute -bottom-1 left-0 right-0 h-0.5 bg-primary-500 dark:bg-primary-400 rounded-full animate-scale-x"></span>
                  )}
                </span>
              </button>
            ))}
          </nav>

          {/* Media Type Tabs - Only show if 'Detect Media' is active */}
          {activeTab === 'detect' && (
            <nav className="flex mt-2 sm:mt-0 space-x-2 sm:space-x-3 border-t sm:border-t-0 pt-2 sm:pt-0 border-neutral-200 dark:border-neutral-700 sm:border-none animate-fade-in" aria-label="Media Type">
              {mediaTypeTabs.map((typeTab) => (
                <button
                  key={typeTab.id}
                  onClick={() => !typeTab.disabled && setActiveMediaType(typeTab.id)}
                  disabled={typeTab.disabled}
                  className={`
                    px-4 py-2 rounded-full text-xs font-medium flex items-center transition-all duration-200 hover:scale-105 active:scale-95
                    ${activeMediaType === typeTab.id 
                      ? 'bg-gradient-to-r from-primary-600 to-primary-500 text-white shadow-lg shadow-primary-500/25 dark:shadow-primary-500/20' 
                      : typeTab.disabled 
                        ? 'bg-neutral-200 dark:bg-neutral-700 text-neutral-400 dark:text-neutral-500 cursor-not-allowed opacity-60'
                        : 'bg-neutral-100 dark:bg-neutral-700 text-neutral-600 dark:text-neutral-300 hover:bg-neutral-200 dark:hover:bg-neutral-600 hover:shadow-md'
                    }
                  `}
                  title={typeTab.disabled ? "Audio analysis coming soon" : `Switch to ${typeTab.label} analysis`}
                >
                  <span className={`mr-1.5 transition-transform duration-200 ${activeMediaType === typeTab.id ? 'scale-110' : ''}`}>
                    {typeTab.icon}
                  </span>
                  {typeTab.label}
                  {typeTab.disabled && (
                    <span className="ml-2 text-[10px] bg-neutral-300 dark:bg-neutral-600 px-1.5 py-0.5 rounded-full">Soon</span>
                  )}
                </button>
              ))}
            </nav>
          )}
        </div>
      </div>
    </div>
  );
};

export default NavigationBar;