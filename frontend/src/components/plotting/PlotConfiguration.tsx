'use client';

import React, { useState, useEffect } from 'react';
import { Card } from '../ui/Card';
import { Input } from '../ui/Input';
import { Select } from '../ui/Select';
import { LoadingSpinner } from '../ui/LoadingSpinner';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../ui/Tabs';
import { PlotType } from '@/types/plots';

interface PlotConfigurationData {
  title?: string;
  subtitle?: string;
  source?: string;
  x_column?: string;
  y_column?: string;
  color_column?: string;
  size_column?: string;
  facet_column?: string;
  watermark?: string;
  prefix?: string;
  suffix?: string;
  axis_config?: {
    x_title?: string;
    y_title?: string;
  };
  style_options?: {
    theme?: string;
    font_family?: string;
    font_size?: number;
    background_color?: string;
    grid_visible?: boolean;
  };
}

interface PlotConfigurationProps {
  plotType: PlotType | string;
  data: any[];
  columns: string[];
  onConfigChange: (config: PlotConfigurationData) => void;
  initialConfiguration?: Partial<PlotConfigurationData>;
  className?: string;
}

export function PlotConfiguration({
  plotType,
  data,
  columns,
  onConfigChange,
  initialConfiguration = {},
  className = ''
}: PlotConfigurationProps) {
  const [config, setConfig] = useState<PlotConfigurationData>(initialConfiguration);

  // Use the columns passed directly as props
  const isLoadingData = false;
  const columnInfo = columns.map(col => ({ name: col, type: 'string' }));
  
  // For now, treat all columns as available for any purpose
  // In a real app, you'd analyze the data to determine column types
  const allColumns = columns.map(col => ({ 
    value: col, 
    label: col 
  }));
  
  const numericColumns = allColumns;
  const dateColumns = allColumns;
  const categoricalColumns = allColumns;

  const updateConfig = (key: string, value: any) => {
    const newConfig = {
      ...config,
      [key]: value
    };
    setConfig(newConfig);
    onConfigChange(newConfig);
  };

  const updateStyleOptions = (key: string, value: any) => {
    const newConfig = {
      ...config,
      style_options: {
        ...config.style_options,
        [key]: value
      }
    };
    setConfig(newConfig);
    onConfigChange(newConfig);
  };

  const updateAxisConfig = (key: string, value: any) => {
    const newConfig = {
      ...config,
      axis_config: {
        ...config.axis_config,
        [key]: value
      }
    };
    setConfig(newConfig);
    onConfigChange(newConfig);
  };

  if (isLoadingData) {
    return (
      <div className={`flex items-center justify-center p-8 ${className}`}>
        <LoadingSpinner size="lg" />
        <span className="ml-3 text-gray-600">Loading configuration...</span>
      </div>
    );
  }

  const getColumnOptions = (plotType: PlotType) => {
    switch (plotType) {
      case 'time_series':
        return {
          x_options: dateColumns,
          y_options: numericColumns
        };
      case 'scatter':
        return {
          x_options: numericColumns,
          y_options: numericColumns
        };
      case 'bar':
      case 'horizontal_bar':
        return {
          x_options: categoricalColumns,
          y_options: numericColumns
        };
      case 'histogram':
        return {
          x_options: numericColumns,
          y_options: []
        };
      default:
        return {
          x_options: allColumns,
          y_options: numericColumns
        };
    }
  };

  const { x_options, y_options } = getColumnOptions(plotType as PlotType);

  const renderBasicSettings = () => (
    <div className="space-y-6">
      {/* Plot Information */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-100 mb-1">
            Plot Title
          </label>
          <Input
            value={config.title || ''}
            onChange={(e) => updateConfig('title', e.target.value)}
            placeholder="Enter plot title..."
            className="w-full"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-100 mb-1">
            Subtitle
          </label>
          <Input
            value={config.subtitle || ''}
            onChange={(e) => updateConfig('subtitle', e.target.value)}
            placeholder="Enter subtitle..."
            className="w-full"
          />
        </div>
      </div>

      {/* Required Fields */}
      <div>
        <h4 className="text-md font-medium text-gray-100 mb-3">Required Settings</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {x_options.length > 0 && (
            <div>
              <label className="block text-sm font-medium text-gray-100 mb-1">
                {plotType === 'time_series' ? 'Date Column' : 
                 ['bar', 'horizontal_bar'].includes(plotType) ? 'Category Column' : 
                 'X-Axis Column'} *
              </label>
              <Select
                options={x_options}
                value={config.x_column || ''}
                onChange={(e) => updateConfig('x_column', e.target.value)}
                placeholder="Select column..."
              />
            </div>
          )}
          {y_options.length > 0 && (
            <div>
              <label className="block text-sm font-medium text-gray-100 mb-1">
                {plotType === 'histogram' ? '' : 
                 plotType === 'time_series' ? 'Value Column' : 
                 'Y-Axis Column'} {plotType !== 'histogram' && !['scatter', 'line', 'area'].includes(plotType) ? '*' : ''}
                {['scatter', 'line', 'area'].includes(plotType) && (
                  <span className="text-sm font-normal text-gray-500 ml-1">
                    (Optional - leave empty to plot all numeric columns)
                  </span>
                )}
              </label>
              <Select
                options={y_options}
                value={config.y_column || ''}
                onChange={(e) => updateConfig('y_column', e.target.value)}
                placeholder={['scatter', 'line', 'area'].includes(plotType) ? "All numeric columns (default)" : "Select column..."}
              />
            </div>
          )}
        </div>
      </div>

      {/* Optional Fields */}
      <div>
        <h4 className="text-md font-medium text-gray-100 mb-3">Optional Settings</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {categoricalColumns.length > 0 && (
            <div>
              <label className="block text-sm font-medium text-gray-100 mb-1">
                Color Column
              </label>
              <Select
                options={categoricalColumns}
                value={config.color_column || ''}
                onChange={(e) => updateConfig('color_column', e.target.value)}
                placeholder="Select column..."
              />
            </div>
          )}
          {numericColumns.length > 0 && (
            <div>
              <label className="block text-sm font-medium text-gray-100 mb-1">
                Size Column
              </label>
              <Select
                options={numericColumns}
                value={config.size_column || ''}
                onChange={(e) => updateConfig('size_column', e.target.value)}
                placeholder="Select column..."
              />
            </div>
          )}
          {categoricalColumns.length > 0 && (
            <div>
              <label className="block text-sm font-medium text-gray-100 mb-1">
                Facet Column
              </label>
              <Select
                options={categoricalColumns}
                value={config.facet_column || ''}
                onChange={(e) => updateConfig('facet_column', e.target.value)}
                placeholder="Select column..."
              />
            </div>
          )}
        </div>
      </div>

      {/* Axis Configuration */}
      <div>
        <h4 className="text-md font-medium text-gray-100 mb-3">Axis Labels</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-100 mb-1">
              X-Axis Title
            </label>
            <Input
              value={config.axis_config?.x_title || ''}
              onChange={(e) => updateAxisConfig('x_title', e.target.value)}
              placeholder="X-axis label..."
              className="w-full"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-100 mb-1">
              Y-Axis Title
            </label>
            <Input
              value={config.axis_config?.y_title || ''}
              onChange={(e) => updateAxisConfig('y_title', e.target.value)}
              placeholder="Y-axis label..."
              className="w-full"
            />
          </div>
        </div>
      </div>
    </div>
  );

  const renderStylingSettings = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-100 mb-1">
            Theme
          </label>
          <Select
            options={[
              { value: 'plotly', label: 'Default' },
              { value: 'plotly_white', label: 'White' },
              { value: 'plotly_dark', label: 'Dark' },
              { value: 'ggplot2', label: 'ggplot2' },
              { value: 'seaborn', label: 'Seaborn' }
            ]}
            value={config.style_options?.theme || ''}
            onChange={(e) => updateStyleOptions('theme', e.target.value)}
            placeholder="Select theme..."
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-100 mb-1">
            Font Family
          </label>
          <Select
            options={[
              { value: 'Arial', label: 'Arial' },
              { value: 'Helvetica', label: 'Helvetica' },
              { value: 'Times New Roman', label: 'Times New Roman' },
              { value: 'Courier New', label: 'Courier New' }
            ]}
            value={config.style_options?.font_family || ''}
            onChange={(e) => updateStyleOptions('font_family', e.target.value)}
            placeholder="Select font..."
          />
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-100 mb-1">
            Font Size
          </label>
          <Input
            type="number"
            value={config.style_options?.font_size || ''}
            onChange={(e) => updateStyleOptions('font_size', e.target.value ? Number(e.target.value) : undefined)}
            min={8}
            max={24}
            step={1}
            placeholder="12"
            className="w-full"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-100 mb-1">
            Background Color
          </label>
          <div className="flex items-center space-x-2">
            <input
              type="color"
              value={config.style_options?.background_color || '#ffffff'}
              onChange={(e) => updateStyleOptions('background_color', e.target.value)}
              className="w-12 h-8 border border-gray-300 rounded cursor-pointer"
            />
            <Input
              type="text"
              value={config.style_options?.background_color || ''}
              onChange={(e) => updateStyleOptions('background_color', e.target.value)}
              placeholder="#ffffff"
              className="flex-1"
            />
          </div>
        </div>
      </div>

      <div>
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={config.style_options?.grid_visible || false}
            onChange={(e) => updateStyleOptions('grid_visible', e.target.checked)}
            className="w-4 h-4 text-zinc-600 bg-gray-100 border-gray-300 rounded focus:ring-zinc-500"
          />
          <span className="ml-2 text-sm text-gray-200">Show Grid</span>
        </label>
        <p className="mt-1 text-sm text-gray-500">Display grid lines on the plot</p>
      </div>
    </div>
  );

  const renderAdvancedSettings = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-100 mb-1">
            Data Source
          </label>
          <Input
            value={config.source || ''}
            onChange={(e) => updateConfig('source', e.target.value)}
            placeholder="Data source attribution..."
            className="w-full"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-100 mb-1">
            Watermark
          </label>
          <Input
            value={config.watermark || ''}
            onChange={(e) => updateConfig('watermark', e.target.value)}
            placeholder="Watermark text..."
            className="w-full"
          />
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-100 mb-1">
            Value Prefix
          </label>
          <Input
            value={config.prefix || ''}
            onChange={(e) => updateConfig('prefix', e.target.value)}
            placeholder="e.g., $, €, ..."
            className="w-full"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-100 mb-1">
            Value Suffix
          </label>
          <Input
            value={config.suffix || ''}
            onChange={(e) => updateConfig('suffix', e.target.value)}
            placeholder="e.g., %, units, ..."
            className="w-full"
          />
        </div>
      </div>
    </div>
  );

  return (
    <Card className={className}>
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-100">
            Configure {plotType.replace('_', ' ')} Plot
          </h3>
        </div>

        <Tabs defaultValue="basic" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="basic">Basic Settings</TabsTrigger>
            <TabsTrigger value="styling">Styling</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
          </TabsList>
          
          <TabsContent value="basic" className="mt-6">
            {renderBasicSettings()}
          </TabsContent>
          
          <TabsContent value="styling" className="mt-6">
            {renderStylingSettings()}
          </TabsContent>
          
          <TabsContent value="advanced" className="mt-6">
            {renderAdvancedSettings()}
          </TabsContent>
        </Tabs>
      </div>
    </Card>
  );
} 