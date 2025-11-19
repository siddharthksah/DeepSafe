// frontend/src/components/VisualizationCard.js
import React, { useState } from 'react';
import { BarChart, PieChart, Pie, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadialBarChart, RadialBar, PolarAngleAxis } from 'recharts';
import { Layers, BarChart2, PieChart as PieChartIcon, Activity } from 'lucide-react';

const VisualizationCard = ({ result }) => {
  const [activeChart, setActiveChart] = useState('radial');
  
  if (!result || typeof result.real_votes !== 'number' || typeof result.fake_votes !== 'number') {
    return null;
  }
  
  const totalVotes = result.real_votes + result.fake_votes;
  const pieData = totalVotes > 0 ? [
    { name: 'Authentic', value: result.real_votes, color: '#22c55e', percentage: (result.real_votes / totalVotes * 100).toFixed(1) },
    { name: 'AI-Generated', value: result.fake_votes, color: '#ef4444', percentage: (result.fake_votes / totalVotes * 100).toFixed(1) },
  ] : [{ name: 'No Votes', value: 1, color: '#94a3b8', percentage: 0 }];

  const confidenceScore = result.deepfake_probability;
  const isLikelyDeepfake = result.is_likely_deepfake;

  // Data for radial bar chart
  const radialData = [{
    name: 'Confidence',
    value: (confidenceScore * 100).toFixed(1),
    fill: isLikelyDeepfake ? '#ef4444' : '#22c55e',
  }];

  const confidenceBarData = [
    {
      name: 'Analysis Confidence',
      'AI-Generated': isLikelyDeepfake ? confidenceScore : 0,
      'Authentic': !isLikelyDeepfake ? confidenceScore : 0,
    },
  ];

  // Custom label for pie chart
  const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, value, name }) => {
    const RADIAN = Math.PI / 180;
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);
    
    if (value === 0) return null;

    return (
      <text 
        x={x} 
        y={y} 
        fill="white" 
        textAnchor={x > cx ? 'start' : 'end'} 
        dominantBaseline="central" 
        className="text-sm font-bold"
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload[0]) {
      return (
        <div className="bg-white dark:bg-neutral-800 p-3 rounded-lg shadow-lg border border-neutral-200 dark:border-neutral-600">
          <p className="text-sm font-semibold text-neutral-800 dark:text-neutral-100">
            {payload[0].name || label}
          </p>
          <p className="text-sm text-neutral-600 dark:text-neutral-400">
            {payload[0].dataKey === 'value' 
              ? `Votes: ${payload[0].value}` 
              : `Confidence: ${(payload[0].value * 100).toFixed(1)}%`}
          </p>
        </div>
      );
    }
    return null;
  };

  const chartTypes = [
    { id: 'radial', label: 'Confidence Gauge', icon: <Activity size={16} /> },
    { id: 'pie', label: 'Vote Distribution', icon: <PieChartIcon size={16} /> },
    { id: 'bar', label: 'Confidence Bar', icon: <BarChart2 size={16} /> },
  ];

  return (
    <div className="bg-gradient-to-br from-neutral-50 to-white dark:from-neutral-800 dark:to-neutral-700 rounded-xl shadow-sm border border-neutral-200 dark:border-neutral-600 p-6 hover:shadow-md transition-all duration-300">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-neutral-800 dark:text-neutral-100">Analysis Insights</h3>
        <div className="flex bg-neutral-100 dark:bg-neutral-700 rounded-lg p-1">
          {chartTypes.map((type) => (
            <button
              key={type.id}
              onClick={() => setActiveChart(type.id)}
              className={`
                px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-200 flex items-center
                ${activeChart === type.id 
                  ? 'bg-white dark:bg-neutral-600 text-primary-600 dark:text-primary-400 shadow-sm' 
                  : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-800 dark:hover:text-neutral-200'
                }
              `}
            >
              <span className={`mr-1.5 ${activeChart === type.id ? 'text-primary-500' : 'text-neutral-400'}`}>
                {type.icon}
              </span>
              {type.label}
            </button>
          ))}
        </div>
      </div>

      <div className="h-[280px] animate-fade-in">
        {activeChart === 'radial' && (
          <div className="relative h-full">
            <ResponsiveContainer width="100%" height="100%">
              <RadialBarChart cx="50%" cy="50%" innerRadius="60%" outerRadius="90%" startAngle={180} endAngle={0} data={radialData}>
                <PolarAngleAxis
                  type="number"
                  domain={[0, 100]}
                  angleAxisId={0}
                  tick={false}
                />
                <RadialBar
                  background={{ fill: '#e5e7eb' }}
                  dataKey="value"
                  cornerRadius={10}
                  className="transition-all duration-500"
                />
                <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" className="text-3xl font-bold fill-current text-neutral-800 dark:text-neutral-100">
                  {radialData[0].value}%
                </text>
                <text x="50%" y="50%" dy={30} textAnchor="middle" dominantBaseline="middle" className="text-sm fill-current text-neutral-600 dark:text-neutral-400">
                  {isLikelyDeepfake ? 'AI-Generated' : 'Authentic'}
                </text>
              </RadialBarChart>
            </ResponsiveContainer>
            <div className="absolute bottom-0 left-0 right-0 flex justify-center space-x-8">
              <div className="flex items-center">
                <div className="w-3 h-3 bg-success-500 rounded-full mr-2"></div>
                <span className="text-xs text-neutral-600 dark:text-neutral-400">Authentic</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 bg-danger-500 rounded-full mr-2"></div>
                <span className="text-xs text-neutral-600 dark:text-neutral-400">AI-Generated</span>
              </div>
            </div>
          </div>
        )}

        {activeChart === 'pie' && (
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={renderCustomizedLabel}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
                animationBegin={0}
                animationDuration={800}
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend 
                verticalAlign="bottom" 
                iconType="circle"
                formatter={(value, entry) => (
                  <span className="text-sm text-neutral-700 dark:text-neutral-300">
                    {value} ({entry.payload.percentage}%)
                  </span>
                )}
              />
            </PieChart>
          </ResponsiveContainer>
        )}

        {activeChart === 'bar' && (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={confidenceBarData}
              layout="vertical"
              margin={{ top: 20, right: 30, left: 40, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" horizontal={false} />
              <XAxis 
                type="number" 
                domain={[0, 1]} 
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} 
                style={{ fontSize: '12px' }}
                stroke="#6b7280"
              />
              <YAxis 
                dataKey="name" 
                type="category" 
                hide 
              />
              <Tooltip 
                formatter={(value) => `${(value * 100).toFixed(1)}%`}
                content={<CustomTooltip />}
              />
              <Legend 
                iconType="rect"
                wrapperStyle={{ fontSize: '14px' }}
                formatter={(value) => (
                  <span className="text-sm text-neutral-700 dark:text-neutral-300">{value}</span>
                )}
              />
              <Bar 
                dataKey="Authentic" 
                fill="#22c55e" 
                barSize={40} 
                radius={[0, 8, 8, 0]}
                animationDuration={800}
              />
              <Bar 
                dataKey="AI-Generated" 
                fill="#ef4444" 
                barSize={40} 
                radius={[0, 8, 8, 0]}
                animationDuration={800}
              />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Summary Stats */}
      <div className="mt-6 pt-4 border-t border-neutral-200 dark:border-neutral-600 grid grid-cols-3 gap-4 text-center">
        <div>
          <p className="text-xs text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">Total Models</p>
          <p className="text-lg font-bold text-neutral-800 dark:text-neutral-100">{totalVotes}</p>
        </div>
        <div>
          <p className="text-xs text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">Consensus</p>
          <p className="text-lg font-bold text-neutral-800 dark:text-neutral-100">
            {totalVotes > 0 ? `${Math.max(result.fake_votes, result.real_votes) / totalVotes * 100}%` : 'N/A'}
          </p>
        </div>
        <div>
          <p className="text-xs text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">Method</p>
          <p className="text-lg font-bold text-neutral-800 dark:text-neutral-100 capitalize">
            {result.ensemble_method_used || 'N/A'}
          </p>
        </div>
      </div>
    </div>
  );
};

export default VisualizationCard;