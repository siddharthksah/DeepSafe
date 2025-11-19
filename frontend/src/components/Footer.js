// frontend/src/components/Footer.js
import React from 'react';
import { Github, Mail, ExternalLink } from 'lucide-react';

const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="bg-gradient-to-b from-neutral-800 to-neutral-900 text-neutral-300 mt-auto">
      <div className="border-t border-neutral-700/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Brand Section */}
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <div className="overflow-hidden rounded-full">
                  <img 
                    src="/assets/deepsafe.png" 
                    alt="DeepSafe Logo" 
                    className="h-8 w-8 object-cover"
                  />
                </div>
                <span className="text-xl font-bold text-white">DeepSafe</span>
              </div>
              <p className="text-sm text-neutral-400 leading-relaxed">
                Leveraging cutting-edge AI to ensure digital authenticity and protect against sophisticated deepfakes.
              </p>
              <div className="flex space-x-4">
                <a 
                  href="https://github.com/siddharthksah/deepsafe" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-neutral-400 hover:text-primary-400 transition-colors duration-200"
                  aria-label="GitHub Repository"
                >
                  <Github className="h-5 w-5" />
                </a>
                <a 
                  href="mailto:deepsafe.hq@gmail.com"
                  className="text-neutral-400 hover:text-primary-400 transition-colors duration-200"
                  aria-label="Contact Email"
                >
                  <Mail className="h-5 w-5" />
                </a>
              </div>
            </div>

            {/* Quick Links */}
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider">Resources</h3>
              <ul className="space-y-2">
                <li>
                  <a 
                    href="/docs" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-sm text-neutral-400 hover:text-primary-400 transition-colors duration-200 flex items-center group"
                  >
                    API Documentation
                    <ExternalLink className="h-3 w-3 ml-1 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </a>
                </li>
                <li>
                  <a 
                    href="https://github.com/siddharthksah/deepsafe" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-sm text-neutral-400 hover:text-primary-400 transition-colors duration-200 flex items-center group"
                  >
                    Source Code
                    <ExternalLink className="h-3 w-3 ml-1 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </a>
                </li>
                <li>
                  <a 
                    href="mailto:deepsafe.hq@gmail.com?subject=DeepSafe Enterprise Inquiry"
                    className="text-sm text-neutral-400 hover:text-primary-400 transition-colors duration-200"
                  >
                    Enterprise Solutions
                  </a>
                </li>
              </ul>
            </div>

            {/* Status */}
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider">System Status</h3>
              <div className="flex items-center space-x-2">
                <div className="h-2 w-2 bg-success-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-neutral-400">All systems operational</span>
              </div>
              <p className="text-xs text-neutral-500">
                DeepSafe is continuously improving. Report issues on our GitHub repository.
              </p>
            </div>
          </div>

          {/* Bottom Bar */}
          <div className="mt-8 pt-8 border-t border-neutral-700/50 flex flex-col sm:flex-row justify-between items-center space-y-4 sm:space-y-0">
            <div className="text-sm text-neutral-500">
              © {currentYear} DeepSafe. All rights reserved.
            </div>
            <div className="flex space-x-6 text-xs text-neutral-500">
              <a href="/privacy" className="hover:text-neutral-400 transition-colors">Privacy Policy</a>
              <a href="/terms" className="hover:text-neutral-400 transition-colors">Terms of Service</a>
              <a href="/cookies" className="hover:text-neutral-400 transition-colors">Cookie Policy</a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;