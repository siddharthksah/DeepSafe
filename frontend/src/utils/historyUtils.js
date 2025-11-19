// frontend/src/utils/historyUtils.js
const HISTORY_KEY = 'deepsafe_analysis_history';
const MAX_HISTORY_ITEMS = 50;

export const getStoredHistory = () => {
  try {
    const stored = localStorage.getItem(HISTORY_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch (error) {
    console.error('Error loading history:', error);
    return [];
  }
};

export const addToHistory = (entry) => {
  try {
    const history = getStoredHistory();
    const newHistory = [entry, ...history].slice(0, MAX_HISTORY_ITEMS);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(newHistory));
  } catch (error) {
    console.error('Error saving to history:', error);
  }
};

export const clearHistory = () => {
  try {
    localStorage.removeItem(HISTORY_KEY);
  } catch (error) {
    console.error('Error clearing history:', error);
  }
};

export const removeFromHistory = (id) => {
  try {
    const history = getStoredHistory();
    const newHistory = history.filter(item => item.id !== id);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(newHistory));
  } catch (error) {
    console.error('Error removing from history:', error);
  }
};