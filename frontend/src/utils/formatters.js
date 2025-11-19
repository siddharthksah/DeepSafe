/**
 * formatters.js - Utility functions for formatting data in the DeepSafe UI
 */

/**
 * Formats a probability value as a percentage string.
 * @param {number} probability - Value between 0 and 1.
 * @param {number} [decimals=1] - Number of decimal places.
 * @returns {string} Formatted percentage.
 */
export const formatProbability = (probability, decimals = 1) => {
  if (typeof probability !== 'number' || isNaN(probability)) {
    return 'N/A';
  }
  return (probability * 100).toFixed(decimals) + '%';
};

/**
 * Returns Tailwind CSS color classes for probability/confidence displays.
 * @param {number} probability - Probability of being fake (0 to 1).
 * @param {string} type - 'bg' for background, 'text' for text, 'border' for border.
 * @returns {string} Tailwind CSS color class.
 */
export const getProbabilityColorClasses = (probability, type = 'bg') => {
  if (typeof probability !== 'number' || isNaN(probability)) {
    return type === 'bg' ? 'bg-neutral-500' : type === 'text' ? 'text-neutral-500' : 'border-neutral-500';
  }
  if (probability >= 0.7) { // High probability of fake
    return type === 'bg' ? 'bg-danger-500' : type === 'text' ? 'text-danger-600' : 'border-danger-500';
  }
  if (probability >= 0.4) { // Medium probability / Uncertain
    return type === 'bg' ? 'bg-warning-500' : type === 'text' ? 'text-warning-600' : 'border-warning-500';
  }
  // Low probability of fake (likely authentic)
  return type === 'bg' ? 'bg-success-500' : type === 'text' ? 'text-success-600' : 'border-success-500';
};

/**
 * Formats time in seconds to a human-readable string (e.g., "1.23s", "500ms").
 * @param {number} timeInSeconds - Time in seconds.
 * @param {number} [decimals=2] - Number of decimal places for seconds.
 * @returns {string} Formatted time string.
 */
export const formatTime = (timeInSeconds, decimals = 2) => {
  if (typeof timeInSeconds !== 'number' || isNaN(timeInSeconds)) {
    return 'N/A';
  }
  if (timeInSeconds < 0.001 && timeInSeconds > 0) { // Very small times in ms
    return `${(timeInSeconds * 1000).toFixed(0)}ms`;
  }
  if (timeInSeconds < 1) { // Times less than 1s, show more precision or in ms
    return `${(timeInSeconds * 1000).toFixed(0)}ms`;
  }
  if (timeInSeconds < 60) {
    return `${timeInSeconds.toFixed(decimals)}s`;
  }
  const minutes = Math.floor(timeInSeconds / 60);
  const seconds = (timeInSeconds % 60).toFixed(0);
  return `${minutes}m ${seconds}s`;
};

/**
 * Gets a human-readable verdict based on the API's `is_likely_deepfake` and `deepfake_probability` (which is treated as confidence).
 * @param {boolean} isLikelyDeepfake - True if the ensemble considers it a deepfake.
 * @param {number} confidence - The confidence score from the API (0 to 1).
 * @returns {{text: string, detailed: string}} Verdict object with short text and detailed explanation.
 */
export const getVerdictDetails = (isLikelyDeepfake, confidence) => {
  if (typeof confidence !== 'number' || isNaN(confidence)) {
    return { text: 'Undetermined', detailed: 'Analysis could not determine the authenticity.' };
  }

  if (isLikelyDeepfake) {
    if (confidence > 0.85) return { text: 'Highly Likely AI-Generated', detailed: `Strong indicators of manipulation detected with ${formatProbability(confidence)} confidence.` };
    if (confidence > 0.6) return { text: 'Likely AI-Generated', detailed: `AI manipulation patterns detected with ${formatProbability(confidence)} confidence.` };
    return { text: 'Potentially AI-Generated', detailed: `Some manipulation indicators found. Confidence: ${formatProbability(confidence)}.` };
  } else {
    if (confidence > 0.85) return { text: 'Highly Likely Authentic', detailed: `Strong indicators of authenticity detected with ${formatProbability(confidence)} confidence.` };
    if (confidence > 0.6) return { text: 'Likely Authentic', detailed: `Image appears authentic with ${formatProbability(confidence)} confidence.` };
    return { text: 'Potentially Authentic', detailed: `Few manipulation indicators found. Confidence: ${formatProbability(confidence)}.` };
  }
};


/**
 * Gets a display name for a model from its ID.
 * @param {string} modelId - The internal model ID.
 * @returns {string} User-friendly model name.
 */
export const getModelDisplayName = (modelId) => {
  const modelNames = {
    'npr_deepfakedetection': 'NPR DeepFake',
    'yermandy_clip_detection': 'Yermandy CLIP',
    'wavelet_clip_detection': 'Wavelet CLIP',
    'universalfakedetect': 'Universal Detector',
    'trufor': 'TruFor',
    'spsl_deepfake_detection': 'SPSL DeepFake',
    'ucf_deepfake_detection': 'UCF DeepFake',

  };
  return modelNames[modelId] || modelId.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
};

/**
 * Truncates text with ellipsis if it exceeds max length.
 * @param {string} text - Text to truncate.
 * @param {number} maxLength - Maximum allowed length.
 * @returns {string} Truncated text.
 */
export const truncateText = (text, maxLength = 50) => {
  if (!text) return '';
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength) + '...';
};