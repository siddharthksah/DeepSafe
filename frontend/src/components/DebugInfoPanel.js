// frontend/src/components/DebugInfoPanel.js
import React, { useState } from 'react';
import { Code, Copy, CheckCircle } from 'lucide-react';

const DebugInfoPanel = ({ result }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify(result, null, 2)).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="bg-neutral-900 rounded-lg p-4 overflow-hidden">
      <div className="flex justify-between items-center mb-3">
        <h4 className="text-sm font-mono text-red-400 flex items-center">
          <Code className="h-4 w-4 mr-2" />
          Debug: Raw API Response
        </h4>
        <button
          onClick={handleCopy}
          className="px-3 py-1 bg-neutral-800 text-neutral-300 rounded text-xs hover:bg-neutral-700 transition-colors flex items-center"
        >
          {copied ? (
            <>
              <CheckCircle className="h-3 w-3 mr-1 text-success-400" />
              Copied
            </>
          ) : (
            <>
              <Copy className="h-3 w-3 mr-1" />
              Copy
            </>
          )}
        </button>
      </div>
      <pre className="text-xs text-neutral-300 overflow-x-auto bg-neutral-800 p-3 rounded">
        {JSON.stringify(result, null, 2)}
      </pre>
    </div>
  );
};

export default DebugInfoPanel;