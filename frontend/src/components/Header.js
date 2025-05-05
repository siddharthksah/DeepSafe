import React from 'react';
import { Shield, Info } from 'lucide-react';

const Header = ({ showSettings, toggleSettings }) => {
  return (
    <header className="bg-white shadow-md">
      <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8 flex justify-between items-center">
        <div className="flex items-center space-x-3">
          <Shield className="h-8 w-8 text-indigo-600" />
          <div>
            <h1 className="text-2xl font-bold text-gray-900">DeepSafe</h1>
            <p className="text-sm text-gray-500">Advanced Deepfake Detection System</p>
          </div>
        </div>
        <button 
          onClick={toggleSettings}
          className="px-4 py-2 bg-indigo-50 text-indigo-700 rounded-md flex items-center space-x-2 hover:bg-indigo-100 transition"
        >
          <Info size={18} />
          <span>{showSettings ? 'Hide Settings' : 'Settings'}</span>
        </button>
      </div>
    </header>
  );
};

export default Header;