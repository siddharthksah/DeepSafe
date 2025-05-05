import React from 'react';
import { FileUp, XCircle, Camera, Loader2, AlertTriangle } from 'lucide-react';

const UploadSection = ({
  selectedFile,
  setSelectedFile,
  preview,
  setPreview,
  setResult,
  loading,
  processStage,
  error,
  handleFileChange,
  handleDemoImage,
  handleSubmit
}) => {
  return (
    <section className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
      <h2 className="text-lg font-medium mb-6 text-gray-900">Upload an Image to Analyze</h2>
      
      <div className="space-y-6">
        {/* File Upload Area */}
        <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-lg p-6 bg-gray-50 hover:bg-gray-100 transition-colors">
          <input
            type="file"
            id="image-upload"
            accept="image/jpeg,image/png,image/webp"
            onChange={handleFileChange}
            className="hidden"
          />
          
          {!preview ? (
            <div className="text-center">
              <FileUp className="mx-auto h-12 w-12 text-gray-400" />
              <div className="mt-4 flex text-sm text-gray-600">
                <label
                  htmlFor="image-upload"
                  className="relative cursor-pointer font-medium text-indigo-600 hover:text-indigo-500"
                >
                  <span>Upload a file</span>
                </label>
                <p className="pl-1">or drag and drop</p>
              </div>
              <p className="text-xs text-gray-500">
                PNG, JPG, WebP up to 10MB
              </p>
              
              {/* Demo Options */}
              <div className="mt-4 pt-4 border-t border-gray-200">
                <p className="text-sm text-gray-600 mb-2">Or try a demo:</p>
                <div className="flex space-x-4 justify-center">
                  <button
                    type="button"
                    onClick={() => handleDemoImage('sample')}
                    className="px-3 py-2 bg-gray-200 text-gray-700 rounded text-sm hover:bg-gray-300"
                  >
                    Sample Image
                  </button>
                  <button
                    type="button"
                    onClick={() => handleDemoImage('camera')}
                    className="px-3 py-2 bg-gray-200 text-gray-700 rounded text-sm hover:bg-gray-300 flex items-center"
                  >
                    <Camera size={16} className="mr-1" />
                    Take Photo
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="relative">
              <img 
                src={preview} 
                alt="Preview" 
                className="max-h-80 rounded shadow-sm" 
              />
              <button
                type="button"
                onClick={() => {
                  setSelectedFile(null);
                  setPreview(null);
                  setResult(null);
                }}
                className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600"
              >
                <XCircle size={20} />
              </button>
              <div className="mt-3 text-center text-sm text-gray-500">
                {selectedFile?.name || "Selected Image"}
              </div>
            </div>
          )}
        </div>
        
        {/* Submit Button */}
        <div>
          <button
            type="button"
            onClick={handleSubmit}
            disabled={!selectedFile || loading}
            className={`w-full flex justify-center items-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white 
              ${!selectedFile || loading ? 'bg-indigo-300 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500'}`}
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin -ml-1 mr-2 h-4 w-4" />
                Analyzing...
              </>
            ) : (
              'Analyze Image'
            )}
          </button>
        </div>
        
        {/* Loading Progress Stage */}
        {loading && processStage && (
          <div className="text-center text-sm text-gray-600">
            {processStage}
          </div>
        )}
        
        {/* Error Message */}
        {error && (
          <div className="flex items-center p-3 bg-red-50 border border-red-200 rounded-md text-red-800 text-sm">
            <AlertTriangle className="flex-shrink-0 h-5 w-5 mr-2 text-red-600" />
            {error}
          </div>
        )}
      </div>
    </section>
  );
};

export default UploadSection;