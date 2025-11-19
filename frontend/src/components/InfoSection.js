// frontend/src/components/InfoSection.js
import React from 'react';
import { Zap, Layers, Cpu, ShieldCheck, Brain, TrendingUp, Lock, Users, BarChartHorizontal } from 'lucide-react';

const InfoSection = ({ darkMode }) => {
  const features = [
    {
      name: 'Advanced Stacking Ensemble',
      description: 'Utilizes a meta-learner trained on diverse model outputs for superior accuracy. Intelligently falls back to voting/averaging if needed.',
      icon: Brain,
      highlight: true,
    },
    {
      name: 'Multiple AI Models',
      description: 'Combines 6+ specialized detection models for comprehensive artifact coverage.',
      icon: Layers,
    },
    {
      name: 'Real-time Analysis',
      description: 'Get results in seconds with parallel model processing and optimized result aggregation.',
      icon: Zap,
    },
    {
      name: 'CPU Optimized',
      description: 'Designed for efficient operation on standard CPU infrastructure, making advanced detection accessible without specialized hardware.',
      icon: Cpu,
    },
    {
      name: 'Detailed Diagnostics',
      description: 'View individual model predictions, probabilities, and ensemble decisions to understand the detection process.',
      icon: BarChartHorizontal,
    },
    {
      name: 'Privacy First',
      description: 'All processing happens on our secure servers. Your images are analyzed and not retained post-analysis.',
      icon: Lock,
    },
    {
      name: 'Enterprise Ready',
      description: 'Built for scalability with robust health monitoring, request tracking, and comprehensive API documentation.',
      icon: ShieldCheck,
    },
    {
      name: 'Continuously Evolving',
      description: 'Our open architecture allows integration of new detection models as deepfake technology advances.',
      icon: Users,
    },
  ];

  return (
    <section className="py-12 bg-white dark:bg-neutral-800 rounded-xl shadow-lg animate-fade-in transition-colors">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h2 className="text-base font-semibold text-primary-600 dark:text-primary-400 tracking-wide uppercase">Our Technology</h2>
          <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-neutral-900 dark:text-neutral-100 sm:text-4xl">
            State-of-the-Art Deepfake Detection
          </p>
          <p className="mt-4 max-w-2xl text-xl text-neutral-500 dark:text-neutral-400 lg:mx-auto">
            DeepSafe leverages a cutting-edge AI ensemble, featuring an advanced Stacking meta-learner, to protect against digital deception with industry-leading accuracy.
          </p>
        </div>

        <div className="mt-16">
          <dl className="space-y-10 md:space-y-0 md:grid md:grid-cols-2 md:gap-x-8 md:gap-y-10">
            {features.map((feature) => (
              <div key={feature.name} className={`relative p-6 rounded-lg shadow-sm transition-all hover:shadow-md
                ${feature.highlight 
                  ? 'bg-gradient-to-br from-primary-50 to-primary-100 dark:from-primary-900/20 dark:to-primary-800/20 border-2 border-primary-200 dark:border-primary-700' 
                  : 'bg-neutral-50 dark:bg-neutral-700/50 border border-neutral-200 dark:border-neutral-600'}`}>
                <dt>
                  <div className={`absolute flex items-center justify-center h-12 w-12 rounded-md text-white
                    ${feature.highlight 
                      ? 'bg-gradient-to-br from-primary-600 to-primary-700 dark:from-primary-500 dark:to-primary-600' 
                      : 'bg-primary-500 dark:bg-primary-600'}`}>
                    <feature.icon className="h-6 w-6" aria-hidden="true" />
                  </div>
                  <p className="ml-16 text-lg leading-6 font-medium text-neutral-900 dark:text-neutral-100">
                    {feature.name}
                    {feature.highlight && (
                      <span className="ml-2 text-xs px-2 py-1 bg-primary-600 text-white rounded-full">Default</span>
                    )}
                  </p>
                </dt>
                <dd className="mt-2 ml-16 text-base text-neutral-500 dark:text-neutral-400">{feature.description}</dd>
              </div>
            ))}
          </dl>
        </div>

        <div className="mt-16 bg-gradient-to-r from-primary-600 to-primary-700 dark:from-primary-700 dark:to-primary-800 rounded-2xl shadow-xl p-8 text-center text-white">
          <h3 className="text-2xl font-bold mb-4">DeepSafe Performance Highlights</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div>
              <p className="text-4xl font-bold">95%+<span className="text-2xl opacity-80">*</span></p>
              <p className="text-sm opacity-90">Detection Accuracy</p>
            </div>
            <div>
              <p className="text-4xl font-bold">{'<5s'}<span className="text-2xl opacity-80">*</span></p>
              <p className="text-sm opacity-90">Typical Analysis Time</p>
            </div>
            <div>
              <p className="text-4xl font-bold">6+</p>
              <p className="text-sm opacity-90">AI Models Combined</p>
            </div>
          </div>
          <p className="text-sm opacity-90 max-w-2xl mx-auto">
            Our ensemble with meta-learner achieves state-of-the-art performance by intelligently combining predictions. <span className="opacity-70">(*Performance metrics are indicative and may vary based on dataset and image complexity.)</span>
          </p>
        </div>

        <div className="mt-16 text-center">
          <h3 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100">Commitment to Digital Authenticity</h3>
          <p className="mt-4 text-lg text-neutral-600 dark:text-neutral-400 max-w-3xl mx-auto">
            In an era of rapidly advancing generative AI, DeepSafe provides a critical layer of defense against digital deception. Our ongoing research and development ensure that our detection capabilities evolve alongside emerging manipulation methods.
          </p>
          <div className="mt-8 flex justify-center space-x-6">
            <a href="https://github.com/siddharthksah/deepsafe" target="_blank" rel="noopener noreferrer" className="text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 font-medium transition-colors">
              View Source Code
            </a>
            <a href="/docs" target="_blank" rel="noopener noreferrer" className="text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 font-medium transition-colors">
              API Documentation
            </a>
          </div>
        </div>
      </div>
    </section>
  );
};

export default InfoSection;