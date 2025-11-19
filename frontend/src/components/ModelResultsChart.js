// frontend/src/components/ModelResultsChart.js
import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LabelList, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { getModelDisplayName } from '../utils/formatters';
import { BarChart3, Radar as RadarIcon } from 'lucide-react';

const ModelResultsChart = ({ modelResults }) => {
  const [chartType, setChartType] = useState('bar');
  
  const data = Object.entries(modelResults)
    .filter(([_, result]) => !result.error && result.probability !== undefined)
    .map(([modelId, result]) => ({
      name: getModelDisplayName(modelId),
      shortName: getModelDisplayName(modelId).split(' ')[0], // For radar chart
      probability: result.probability * 100,
      verdict: result.class,
      confidence: result.probability > 0.5 ? result.probability : 1 - result.probability
    }))
    .sort((a, b) => b.probability - a.probability);

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload[0]) {
      const data = payload[0].payload;
      return (
        <div className="bg-white dark:bg-neutral-800 p-4 rounded-lg shadow-xl border border-neutral-200 dark:border-neutral-600">
          <p className="text-sm font-semibold text-neutral-800 dark:text-neutral-100 mb-2">
            {data.name}
          </p>
          <div className="space-y-1">
            <p className="text-sm text-neutral-600 dark:text-neutral-400">
              AI Probability: <span className="font-medium text-neutral-800 dark:text-neutral-200">{data.probability.toFixed(1)}%</span>
            </p>
            <p className="text-sm text-neutral-600 dark:text-neutral-400">
              Verdict: <span className={`font-medium ${data.verdict === 'fake' ? 'text-danger-600 dark:text-danger-400' : 'text-success-600 dark:text-success-400'}`}>
                {data.verdict?.toUpperCase()}
              </span>
            </p>
            <p className="text-sm text-neutral-600 dark:text-neutral-400">
              Confidence: <span className="font-medium text-neutral-800 dark:text-neutral-200">{(data.confidence * 100).toFixed(1)}%</span>
            </p>
          </div>
        </div>
      );
    }
    return null;
  };

  const CustomLabel = ({ x, y, width, value }) => (
    <text 
      x={x + width / 2} 
      y={y - 5} 
      fill="#6b7280" 
      textAnchor="middle" 
      fontSize="12"
      className="font-medium"
    >
      {value.toFixed(0)}%
    </text>
  );

  const chartTypes = [
    { id: 'bar', label: 'Bar Chart', icon: <BarChart3 size={16} /> },
    { id: 'radar', label: 'Radar Chart', icon: <RadarIcon size={16} /> },
  ];

  return (
    <div className="bg-gradient-to-br from-white to-neutral-50 dark:from-neutral-800 dark:to-neutral-700 p-6 rounded-xl shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-neutral-800 dark:text-neutral-100">
          Model Predictions Analysis
        </h3>
        <div className="flex bg-neutral-100 dark:bg-neutral-700 rounded-lg p-1">
          {chartTypes.map((type) => (
            <button
              key={type.id}
              onClick={() => setChartType(type.id)}
              className={`
                px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-200 flex items-center
                ${chartType === type.id 
                  ? 'bg-white dark:bg-neutral-600 text-primary-600 dark:text-primary-400 shadow-sm' 
                  : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-800 dark:hover:text-neutral-200'
                }
              `}
            >
              <span className={`mr-1.5 ${chartType === type.id ? 'text-primary-500' : 'text-neutral-400'}`}>
                {type.icon}
              </span>
              {type.label}
            </button>
          ))}
        </div>
      </div>

      {chartType === 'bar' ? (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart 
            data={data} 
            margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" opacity={0.5} />
            <XAxis 
              dataKey="name" 
              angle={-45} 
              textAnchor="end" 
              height={80}
              tick={{ fontSize: 12, fill: '#6b7280' }}
              interval={0}
            />
            <YAxis 
              domain={[0, 100]}
              tick={{ fontSize: 12, fill: '#6b7280' }}
              label={{ 
                value: 'AI Probability (%)', 
                angle: -90, 
                position: 'insideLeft',
                style: { fontSize: 12, fill: '#6b7280' }
              }}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(0, 0, 0, 0.05)' }} />
            <Bar dataKey="probability" radius={[8, 8, 0, 0]} animationDuration={800}>
              <LabelList content={<CustomLabel />} />
              {data.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={entry.verdict === 'fake' 
                    ? `rgba(239, 68, 68, ${0.6 + (entry.probability / 100) * 0.4})` 
                    : `rgba(34, 197, 94, ${0.6 + ((100 - entry.probability) / 100) * 0.4})`
                  } 
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart data={data}>
            <PolarGrid 
              stroke="#e5e7eb" 
              strokeDasharray="3 3"
              radialLines={true}
            />
            <PolarAngleAxis 
              dataKey="shortName" 
              tick={{ fontSize: 12, fill: '#6b7280' }}
            />
            <PolarRadiusAxis 
              domain={[0, 100]} 
              tick={{ fontSize: 10, fill: '#6b7280' }}
              axisLine={false}
            />
            <Tooltip content={<CustomTooltip />} />
            <Radar 
              name="AI Probability" 
              dataKey="probability" 
              stroke="#ef4444" 
              fill="#ef4444" 
              fillOpacity={0.3}
              strokeWidth={2}
              animationDuration={800}
            />
            <Radar 
              name="Confidence" 
              dataKey={(entry) => entry.confidence * 100} 
              stroke="#3b82f6" 
              fill="#3b82f6" 
              fillOpacity={0.2}
              strokeWidth={2}
              animationDuration={800}
            />
          </RadarChart>
        </ResponsiveContainer>
      )}

      {/* Summary Statistics */}
      <div className="mt-6 pt-4 border-t border-neutral-200 dark:border-neutral-600">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
          <div>
            <p className="text-xs text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">Models</p>
            <p className="text-lg font-bold text-neutral-800 dark:text-neutral-100">{data.length}</p>
          </div>
          <div>
            <p className="text-xs text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">Avg AI Prob</p>
            <p className="text-lg font-bold text-neutral-800 dark:text-neutral-100">
              {data.length > 0 ? (data.reduce((sum, d) => sum + d.probability, 0) / data.length).toFixed(0) : 0}%
            </p>
          </div>
          <div>
            <p className="text-xs text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">Fake Votes</p>
            <p className="text-lg font-bold text-danger-600 dark:text-danger-400">
              {data.filter(d => d.verdict === 'fake').length}
            </p>
          </div>
          <div>
            <p className="text-xs text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">Real Votes</p>
            <p className="text-lg font-bold text-success-600 dark:text-success-400">
              {data.filter(d => d.verdict === 'real').length}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelResultsChart;