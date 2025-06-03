'use client';

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card } from '../ui/Card';
import { LoadingSpinner } from '../ui/LoadingSpinner';
import { api } from '@/lib/api';
import { PlotType } from '@/types/plots';

interface PlotTypeSelectorProps {
  selectedType: PlotType | null;
  onTypeSelect: (type: PlotType) => void;
  className?: string;
}

const PLOT_TYPE_DESCRIPTIONS = {
  'time_series': {
    title: 'Time Series',
    description: 'Display data over time with date/datetime columns',
    icon: '📈',
    features: ['Date-based X-axis', 'Multiple metrics', 'Trend analysis']
  },
  'bar': {
    title: 'Bar Chart',
    description: 'Compare categorical data with vertical bars',
    icon: '📊',
    features: ['Categorical comparison', 'Multiple series', 'Horizontal/Vertical']
  },
  'horizontal_bar': {
    title: 'Horizontal Bar',
    description: 'Compare categorical data with horizontal bars',
    icon: '📊',
    features: ['Long category names', 'Easy comparison', 'Ranked data']
  },
  'scatter': {
    title: 'Scatter Plot',
    description: 'Show relationships between two numerical variables',
    icon: '⚫',
    features: ['Correlation analysis', 'Color coding', 'Size mapping']
  },
  'line': {
    title: 'Line Chart',
    description: 'Connect data points to show trends and patterns',
    icon: '📉',
    features: ['Trend visualization', 'Multiple lines', 'Smooth curves']
  },
  'area': {
    title: 'Area Chart',
    description: 'Show cumulative totals and part-to-whole relationships',
    icon: '🏔️',
    features: ['Cumulative data', 'Stacked areas', 'Filled regions']
  },
  'histogram': {
    title: 'Histogram',
    description: 'Show distribution of numerical data',
    icon: '📊',
    features: ['Data distribution', 'Frequency analysis', 'Bin customization']
  }
} as const;

export function PlotTypeSelector({ selectedType, onTypeSelect, className = '' }: PlotTypeSelectorProps) {
  const { data: plotTypes, isLoading, error } = useQuery({
    queryKey: ['plotTypes'],
    queryFn: api.plots.getTypes,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  if (isLoading) {
    return (
      <div className={`flex items-center justify-center p-8 ${className}`}>
        <LoadingSpinner size="lg" />
        <span className="ml-3 text-gray-600">Loading plot types...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`p-6 text-center ${className}`}>
        <div className="text-red-600 mb-2">
          ⚠️ Failed to load plot types
        </div>
        <div className="text-sm text-gray-600">
          {error instanceof Error ? error.message : 'Unknown error occurred'}
        </div>
      </div>
    );
  }

  if (!plotTypes?.plot_types?.length) {
    return (
      <div className={`p-6 text-center text-gray-600 ${className}`}>
        No plot types available
      </div>
    );
  }

  return (
    <div className={className}>
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Select Plot Type
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {plotTypes.plot_types.map((plotTypeObj: any) => {
          // Extract the type string from the object
          const type = plotTypeObj.type || plotTypeObj;
          const config = PLOT_TYPE_DESCRIPTIONS[type as keyof typeof PLOT_TYPE_DESCRIPTIONS];
          const isSelected = selectedType === type;
          
          // Use API data if available, fallback to local config
          const displayName = plotTypeObj.name || config?.title || (typeof type === 'string' ? type.replace('_', ' ') : String(type));
          const description = plotTypeObj.description || config?.description || `Create ${typeof type === 'string' ? type.replace('_', ' ') : String(type)} visualization`;
          const icon = plotTypeObj.icon || config?.icon || '📊';
          
          return (
            <Card
              key={type}
              className={`cursor-pointer transition-all duration-200 hover:shadow-md ${
                isSelected 
                  ? 'ring-2 ring-blue-500 bg-blue-50 border-blue-200' 
                  : 'hover:border-gray-300'
              }`}
              onClick={() => onTypeSelect(type as PlotType)}
            >
              <div className="p-4">
                <div className="flex items-center mb-3">
                  <span className="text-2xl mr-3" role="img" aria-label={displayName}>
                    {icon}
                  </span>
                  <h4 className="font-medium text-gray-900">
                    {displayName}
                  </h4>
                </div>
                
                <p className="text-sm text-gray-600 mb-3">
                  {description}
                </p>
                
                {config?.features && (
                  <div className="space-y-1">
                    {config.features.map((feature, index) => (
                      <div key={index} className="flex items-center text-xs text-gray-500">
                        <span className="w-1 h-1 bg-gray-400 rounded-full mr-2" />
                        {feature}
                      </div>
                    ))}
                  </div>
                )}
                
                {plotTypeObj.required_columns && (
                  <div className="mt-3">
                    <div className="text-xs text-gray-500 mb-1">Required columns:</div>
                    <div className="flex flex-wrap gap-1">
                      {plotTypeObj.required_columns.map((col: string, index: number) => (
                        <span key={index} className="px-2 py-1 bg-gray-100 text-xs rounded">
                          {col}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                
                {isSelected && (
                  <div className="mt-3 flex items-center text-sm text-blue-600">
                    <span className="w-2 h-2 bg-blue-500 rounded-full mr-2" />
                    Selected
                  </div>
                )}
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
} 