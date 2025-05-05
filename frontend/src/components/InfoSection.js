import React from 'react';

const InfoSection = () => {
  return (
    <section className="mt-8 bg-white rounded-lg shadow-md p-6 border border-gray-200">
      <h2 className="text-lg font-medium mb-4 text-gray-900">About DeepSafe</h2>
      
      <div className="prose prose-sm text-gray-700">
        <p>
          DeepSafe is an enterprise-grade deepfake detection system that leverages multiple state-of-the-art 
          AI models to identify manipulated images with high accuracy.
        </p>
        <p>
          The system uses an ensemble approach, combining the strengths of different detection models 
          specialized in various manipulation techniques, including:
        </p>
        <ul className="list-disc pl-5 space-y-1">
          <li>GAN-generated images (StyleGAN, ProGAN)</li>
          <li>Diffusion model outputs (Stable Diffusion, DALL-E)</li>
          <li>Face swaps and facial manipulations</li>
          <li>General image manipulations</li>
        </ul>
        <p className="text-sm text-gray-500 mt-4">
          For production-grade deployments, professional support, or custom model integration, 
          please contact our team.
        </p>
      </div>
    </section>
  );
};

export default InfoSection;