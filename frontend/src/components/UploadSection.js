// frontend/src/components/UploadSection.js
import React, { useState, useRef } from 'react';
import { FileUp, Loader2, AlertTriangle, Image as ImageIcon, Video as VideoIcon, Mic as MicIcon, RefreshCw, ChevronDown, Sparkles, FileSearch, Shield } from 'lucide-react';

const UploadSection = ({
  selectedFile,
  preview,
  loading,
  processStage,
  error,
  handleFileChange,
  handleDemoMedia,
  handleSubmit,
  mediaType 
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [showDemoOptions, setShowDemoOptions] = useState(false);
  const inputRef = useRef(null);

  const getMediaSpecifics = () => {
    switch (mediaType) {
      case 'video':
        return {
          icon: <VideoIcon className={`h-12 w-12 ${dragActive ? 'text-primary-600 dark:text-primary-400' : 'text-neutral-400 dark:text-neutral-500'} transition-all duration-300`} />,
          label: 'Analyze Video for AI Manipulation',
          accept: 'video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/x-m4v',
          fileTypes: 'MP4, MOV, AVI, MKV up to 500MB',
          demoFiles: [ 
            { id: 'fake_video_1', label: 'AI Video Sample 1', description: 'Deepfake example', badge: 'AI-Generated' },
            { id: 'fake_video_2', label: 'AI Video Sample 2', description: 'Manipulated video', badge: 'AI-Generated' },
            { id: 'real_video_1', label: 'Real Video Sample 1', description: 'Authentic footage', badge: 'Authentic' },
            { id: 'real_video_2', label: 'Real Video Sample 2', description: 'Original video', badge: 'Authentic' },
          ],
          emptyStateIcon: <VideoIcon className="h-24 w-24 text-neutral-300 dark:text-neutral-600" />,
          emptyStateText: "Drag and drop your video here to check for AI manipulation"
        };
      case 'audio':
        return {
          icon: <MicIcon className={`h-12 w-12 ${dragActive ? 'text-primary-600 dark:text-primary-400' : 'text-neutral-400 dark:text-neutral-500'} transition-all duration-300`} />,
          label: 'Analyze Audio for AI Manipulation',
          accept: 'audio/mpeg,audio/wav,audio/ogg,audio/flac,audio/x-m4a',
          fileTypes: 'MP3, WAV, OGG, FLAC up to 200MB',
          demoFiles: [],
          emptyStateIcon: <MicIcon className="h-24 w-24 text-neutral-300 dark:text-neutral-600" />,
          emptyStateText: "Audio analysis coming soon!"
        };
      case 'image':
      default:
        return {
          icon: <ImageIcon className={`h-12 w-12 ${dragActive ? 'text-primary-600 dark:text-primary-400' : 'text-neutral-400 dark:text-neutral-500'} transition-all duration-300`} />,
          label: 'Analyze Image for AI Manipulation',
          accept: 'image/jpeg,image/png,image/webp',
          fileTypes: 'PNG, JPG, WebP up to 100MB',
          demoFiles: [
            { id: 'ai_face_gan', label: 'AI Face (GAN)', description: 'StyleGAN generated', badge: 'AI-Generated' },
            { id: 'real_portrait', label: 'Real Portrait', description: 'Authentic photo', badge: 'Authentic' },
            { id: 'stylegan_city', label: 'StyleGAN City', description: 'AI-generated scene', badge: 'AI-Generated' },
            { id: 'face_swap', label: 'Face Swap', description: 'Altered identity', badge: 'Manipulated' },
          ],
          emptyStateIcon: <ImageIcon className="h-24 w-24 text-neutral-300 dark:text-neutral-600" />,
          emptyStateText: "Drag and drop your image here to check for AI manipulation"
        };
    }
  };

  const specifics = getMediaSpecifics();

  const handleDrag = (e) => {
    e.preventDefault(); e.stopPropagation(); if (loading) return;
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation(); if (loading) return;
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const syntheticEvent = { target: { files: e.dataTransfer.files } };
      handleFileChange(syntheticEvent);
    }
  };

  const onUploadButtonClick = () => { if (loading) return; inputRef.current.click(); };
  const clearSelection = () => {
    const syntheticEvent = { target: { files: [] } };
    handleFileChange(syntheticEvent); 
    if (inputRef.current) inputRef.current.value = "";
  };

  const EmptyStateIllustration = () => (
    <div className="relative">
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="w-32 h-32 bg-gradient-to-br from-primary-100 to-primary-200 dark:from-primary-900/20 dark:to-primary-800/20 rounded-full blur-3xl animate-pulse"></div>
      </div>
      <div className="relative">
        <div className="w-24 h-24 mx-auto overflow-hidden rounded-full bg-white dark:bg-neutral-800 shadow-lg ring-4 ring-primary-100 dark:ring-primary-900/30">
          <img 
            src="/assets/deepsafe.png" 
            alt="DeepSafe" 
            className="w-full h-full object-cover"
          />
        </div>
        <div className="absolute -top-2 -right-2">
          <Shield className="h-8 w-8 text-primary-500 dark:text-primary-400" />
        </div>
        <div className="absolute -bottom-1 -left-1">
          <FileSearch className="h-6 w-6 text-primary-400 dark:text-primary-500" />
        </div>
      </div>
    </div>
  );

  return (
    <section className="bg-white dark:bg-neutral-800 rounded-xl shadow-lg p-6 sm:p-8 border border-neutral-200 dark:border-neutral-700 animate-fade-in transition-all duration-300 hover:shadow-xl">
      <div className="flex items-center mb-6">
        <div className="p-2 bg-gradient-to-br from-primary-100 to-primary-200 dark:from-primary-900/30 dark:to-primary-800/30 rounded-lg mr-3 overflow-hidden">
          <img 
            src="/assets/deepsafe.png" 
            alt="DeepSafe" 
            className="h-6 w-6 object-cover rounded-md"
          />
        </div>
        <h2 className="text-xl font-semibold text-neutral-800 dark:text-neutral-100">{specifics.label}</h2>
      </div>

      <div className="space-y-6">
        {mediaType === 'audio' ? (
           <div className="text-center p-12 border-2 border-dashed border-neutral-300 dark:border-neutral-600 rounded-xl bg-gradient-to-br from-neutral-50 to-neutral-100 dark:from-neutral-800 dark:to-neutral-700">
            <EmptyStateIllustration />
            <h3 className="text-lg font-medium text-neutral-700 dark:text-neutral-300 mt-6">Audio Analysis Coming Soon!</h3>
            <p className="text-sm text-neutral-500 dark:text-neutral-400 mt-2">
              We're working on bringing advanced audio deepfake detection to DeepSafe.
            </p>
            <div className="mt-4 inline-flex items-center px-3 py-1.5 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-full text-xs font-medium">
              <Sparkles className="h-3 w-3 mr-1.5" />
              Under Development
            </div>
          </div>
        ) : (
          <div
            className={`relative flex flex-col items-center justify-center border-2 
                        ${dragActive ? "border-primary-500 bg-gradient-to-br from-primary-50 to-primary-100 dark:from-primary-900/20 dark:to-primary-800/20 scale-[1.02] shadow-lg" : "border-dashed border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-800"} 
                        rounded-xl p-12 transition-all duration-300 ease-out
                        ${loading ? 'opacity-50 cursor-not-allowed' : 'hover:border-primary-400 dark:hover:border-primary-500 hover:bg-gradient-to-br hover:from-primary-50/50 hover:to-white dark:hover:from-primary-900/10 dark:hover:to-neutral-800 hover:shadow-md'}`}
            onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}
          >
            <input
              ref={inputRef} type="file" id={`${mediaType}-upload`} accept={specifics.accept}
              onChange={handleFileChange} className="hidden" disabled={loading} aria-label={`Upload ${mediaType} file`}
            />
            {!preview ? (
              <div className="text-center">
                <EmptyStateIllustration />
                <p className="mt-6 text-base text-neutral-600 dark:text-neutral-400">
                  {specifics.emptyStateText}
                </p>
                <p className="mt-3 text-sm text-neutral-600 dark:text-neutral-400">
                  <button type="button" onClick={onUploadButtonClick} 
                    className="font-semibold text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 focus:outline-none transition-colors duration-200 hover:underline underline-offset-2" 
                    disabled={loading}>
                    Click to browse
                  </button> {' '}or drag and drop
                </p>
                <p className="text-xs text-neutral-400 dark:text-neutral-500 mt-2">{specifics.fileTypes}</p>

                {specifics.demoFiles && specifics.demoFiles.length > 0 && (
                  <div className="mt-8 pt-6 border-t border-neutral-200 dark:border-neutral-700 w-full max-w-md mx-auto">
                    <button type="button" onClick={() => setShowDemoOptions(!showDemoOptions)} disabled={loading}
                      className="w-full px-4 py-3 bg-gradient-to-r from-primary-600 to-primary-500 hover:from-primary-700 hover:to-primary-600 text-white rounded-lg text-sm font-medium transition-all duration-200 shadow-md hover:shadow-lg disabled:opacity-50 flex items-center justify-center group hover:scale-[1.02] active:scale-[0.98]"
                    > 
                      <Sparkles className="mr-2 h-4 w-4 group-hover:rotate-12 transition-transform duration-300" />
                      Try Demo {mediaType.charAt(0).toUpperCase() + mediaType.slice(1)}s
                      <ChevronDown className={`ml-2 h-4 w-4 transform transition-transform duration-300 ${showDemoOptions ? 'rotate-180' : ''}`} />
                    </button>
                    {showDemoOptions && (
                      <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3 animate-slide-down">
                        {specifics.demoFiles.map((demo) => (
                          <button key={demo.id} type="button" onClick={() => { handleDemoMedia(demo.id); setShowDemoOptions(false); }}
                            disabled={loading} title={demo.description}
                            className="group p-4 bg-white dark:bg-neutral-700 border border-neutral-200 dark:border-neutral-600 rounded-lg hover:border-primary-300 dark:hover:border-primary-400 hover:bg-gradient-to-br hover:from-primary-50 hover:to-white dark:hover:from-primary-900/10 dark:hover:to-neutral-700 transition-all duration-200 text-left disabled:opacity-50 hover:shadow-md hover:scale-[1.02] active:scale-[0.98]"
                          >
                            <div className="flex justify-between items-start">
                              <div>
                                <p className="text-sm font-medium text-neutral-700 dark:text-neutral-200 group-hover:text-primary-700 dark:group-hover:text-primary-300 transition-colors">
                                  {demo.label}
                                </p>
                                <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">{demo.description}</p>
                              </div>
                              <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium whitespace-nowrap ml-2
                                ${demo.badge === 'AI-Generated' ? 'bg-danger-100 dark:bg-danger-900/40 text-danger-700 dark:text-danger-300' 
                                  : demo.badge === 'Authentic' ? 'bg-success-100 dark:bg-success-900/40 text-success-700 dark:text-success-300'
                                  : 'bg-warning-100 dark:bg-warning-900/40 text-warning-700 dark:text-warning-300'}`}>
                                {demo.badge}
                              </span>
                            </div>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : ( 
              <div className="relative group text-center">
                {mediaType === 'image' && (
                  <img src={preview} alt="Preview" className="max-h-80 rounded-lg shadow-lg object-contain inline-block transition-transform duration-300 group-hover:scale-[1.02]" />
                )}
                {mediaType === 'video' && preview && (
                  <video 
                    src={preview} 
                    controls 
                    className="max-h-80 max-w-full rounded-lg shadow-lg object-contain inline-block transition-transform duration-300 group-hover:scale-[1.02]"
                    controlsList="nodownload"
                  />
                )}
                
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-all duration-300 flex items-end justify-center rounded-lg pointer-events-none">
                  <button type="button" onClick={clearSelection} disabled={loading}
                    className="pointer-events-auto mb-4 px-4 py-2 bg-white/90 dark:bg-neutral-800/90 backdrop-blur-sm text-neutral-700 dark:text-neutral-300 rounded-lg shadow-lg opacity-0 group-hover:opacity-100 hover:bg-white dark:hover:bg-neutral-800 transition-all duration-300 transform translate-y-2 group-hover:translate-y-0 disabled:opacity-50 flex items-center"
                    aria-label={`Clear selection and upload another ${mediaType}`}
                  >
                    <RefreshCw size={16} className="mr-2" />
                    Change {mediaType}
                  </button>
                </div>
                <p className="mt-3 text-center text-xs text-neutral-500 dark:text-neutral-400 truncate max-w-xs mx-auto">
                  {selectedFile?.name || `Selected ${mediaType}`}
                </p>
              </div>
            )}
          </div>
        )} 
        
        {mediaType !== 'audio' && ( 
          <div>
            <button type="button" onClick={handleSubmit} disabled={!selectedFile || loading}
              className={`w-full flex justify-center items-center py-3.5 px-6 border border-transparent rounded-lg shadow-md text-base font-medium text-white transition-all duration-200 transform
                ${loading ? 'bg-gradient-to-r from-primary-400 to-primary-500 dark:from-primary-500 dark:to-primary-600 cursor-not-allowed animate-pulse' 
                        : (!selectedFile ? 'bg-neutral-300 dark:bg-neutral-600 cursor-not-allowed' 
                                          : 'bg-gradient-to-r from-primary-600 to-primary-500 hover:from-primary-700 hover:to-primary-600 dark:from-primary-500 dark:to-primary-400 dark:hover:from-primary-600 dark:hover:to-primary-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 hover:scale-[1.02] active:scale-[0.98] hover:shadow-lg')}`}
            > {loading ? (<><Loader2 className="animate-spin -ml-1 mr-3 h-5 w-5" />{processStage || `Analyzing ${mediaType}...`}</>) : (
              <>
                <Shield className="mr-2 h-5 w-5" />
                Analyze {mediaType.charAt(0).toUpperCase() + mediaType.slice(1)}
              </>
            )}
            </button>
          </div>
        )}
        
        {error && (
          <div className="flex items-start p-4 bg-danger-50 dark:bg-danger-900/20 border-l-4 border-danger-400 dark:border-danger-600 rounded-md text-danger-800 dark:text-danger-200 text-sm animate-shake" role="alert">
            <AlertTriangle className="flex-shrink-0 h-5 w-5 mr-3 text-danger-500 dark:text-danger-400 mt-0.5" />
            <div> 
              <p className="font-medium">Error</p> 
              <p className="mt-1">{error}</p> 
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

export default UploadSection;