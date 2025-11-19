import React from 'react';
import { AlertOctagon, RefreshCcw } from 'lucide-react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
    this.setState({ errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center min-h-[50vh] p-6 my-8 bg-red-50 border-2 border-dashed border-red-200 rounded-lg text-center animate-fade-in">
          <AlertOctagon className="w-16 h-16 text-red-400 mb-4" />
          <h2 className="text-2xl font-semibold text-red-700 mb-2">Oops! Something went wrong.</h2>
          <p className="text-red-600 mb-6 max-w-md">
            We've encountered an unexpected issue. Please try refreshing the page. If the problem persists, please contact support.
          </p>
          <button
            onClick={() => window.location.reload()}
            className="px-6 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 transition-colors flex items-center"
          >
            <RefreshCcw size={18} className="mr-2" />
            Refresh Page
          </button>
          {this.props.showDetails && this.state.errorInfo && (
            <details className="mt-6 text-left text-xs text-neutral-600 bg-white p-3 rounded-md shadow w-full max-w-2xl">
              <summary className="cursor-pointer font-medium text-neutral-700">View Error Details</summary>
              <pre className="mt-2 whitespace-pre-wrap bg-neutral-50 p-2 rounded overflow-auto max-h-60">
                {this.state.error && this.state.error.toString()}
                <br />
                {this.state.errorInfo.componentStack}
              </pre>
            </details>
          )}
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;