/**
 * Configuration Panel
 * ---
 * bwr-plots/frontend/src/components/dashboard/ConfigurationPanel.tsx
 * ---
 * Right sidebar component for plot type selection and configuration
 */

'use client';

import React, { useState } from 'react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Select } from '@/components/ui/Select';
import { PlotTypeSelector } from '@/components/plotting/PlotTypeSelector';
import { PlotConfiguration } from '@/components/plotting/PlotConfiguration';
import { useSession } from '@/hooks/useSession';
import { PlotType } from '@/types/plots';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Type Definitions                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

interface ConfigurationPanelProps {
  className?: string;
  onConfigChange?: (config: any) => void;
  onPlotTypeChange?: (type: PlotType) => void;
  onGeneratePlot?: () => void;
}

interface ConfigSectionProps {
  title: string;
  children: React.ReactNode;
  defaultExpanded?: boolean;
  icon?: string;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Collapsible Section Component                                                      │
// └────────────────────────────────────────────────────────────────────────────────────┘

function ConfigSection({ title, children, defaultExpanded = true, icon }: ConfigSectionProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <Card className="mb-4">
      <div className="border-b border-[var(--color-border)]">
        <Button
          variant="ghost"
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full justify-between p-4 h-auto"
        >
          <div className="flex items-center space-x-2">
            {icon && <span className="text-lg">{icon}</span>}
            <span className="font-semibold text-[var(--color-text-primary)]">
              {title}
            </span>
          </div>
          <span className={`transform transition-transform ${isExpanded ? 'rotate-90' : ''}`}>
            ▶
          </span>
        </Button>
      </div>
      
      {isExpanded && (
        <div className="p-4">
          {children}
        </div>
      )}
    </Card>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Plot Type Grid Component                                                           │
// └────────────────────────────────────────────────────────────────────────────────────┘

function PlotTypeGrid({ selectedType, onTypeSelect }: {
  selectedType: PlotType | null;
  onTypeSelect: (type: PlotType) => void;
}) {
  const plotTypes = [
    { type: 'bar', name: 'Bar Chart', icon: '📊', description: 'Compare categories' },
    { type: 'line', name: 'Line Chart', icon: '📈', description: 'Show trends' },
    { type: 'scatter', name: 'Scatter Plot', icon: '⚫', description: 'Show relationships' },
    { type: 'area', name: 'Area Chart', icon: '🏔️', description: 'Show cumulative data' },
    { type: 'time_series', name: 'Time Series', icon: '📅', description: 'Time-based data' },
    { type: 'histogram', name: 'Histogram', icon: '📊', description: 'Data distribution' }
  ];

  return (
    <div className="grid grid-cols-2 gap-2">
      {plotTypes.map((plot) => (
        <Button
          key={plot.type}
          variant={selectedType === plot.type ? 'primary' : 'outline'}
          onClick={() => onTypeSelect(plot.type as PlotType)}
          className="h-auto p-3 flex flex-col items-center space-y-1"
        >
          <span className="text-2xl">{plot.icon}</span>
          <span className="text-xs font-medium">{plot.name}</span>
        </Button>
      ))}
    </div>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Quick Settings Component                                                           │
// └────────────────────────────────────────────────────────────────────────────────────┘

function QuickSettings({ session, onConfigChange }: {
  session: any;
  onConfigChange: (config: any) => void;
}) {
  const columns = session?.uploadedData?.columns || [];
  
  const handleQuickChange = (field: string, value: string) => {
    const newConfig = {
      ...session?.plotConfig,
      [field]: value
    };
    onConfigChange(newConfig);
  };

  if (!session?.uploadedData) {
    return (
      <div className="text-center py-4">
        <div className="text-2xl mb-2">📊</div>
        <p className="text-sm text-[var(--color-text-muted)]">
          Upload data to see configuration options
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Column Selection */}
      <div className="grid grid-cols-1 gap-3">
        <Select
          label="X-Axis Column"
          value={session?.plotConfig?.x_column || ''}
          onChange={(e) => handleQuickChange('x_column', e.target.value)}
          options={columns.map((col: string) => ({ value: col, label: col }))}
          placeholder="Select X column..."
        />
        
        <Select
          label="Y-Axis Column"
          value={session?.plotConfig?.y_column || ''}
          onChange={(e) => handleQuickChange('y_column', e.target.value)}
          options={columns.map((col: string) => ({ value: col, label: col }))}
          placeholder="Select Y column..."
        />
      </div>

      {/* Basic Settings */}
      <div className="space-y-3">
        <Input
          label="Plot Title"
          value={session?.plotConfig?.title || ''}
          onChange={(e) => handleQuickChange('title', e.target.value)}
          placeholder="Enter plot title..."
        />
        
        <Input
          label="Subtitle"
          value={session?.plotConfig?.subtitle || ''}
          onChange={(e) => handleQuickChange('subtitle', e.target.value)}
          placeholder="Enter subtitle..."
        />
      </div>
    </div>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Configuration Presets Component                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

function ConfigurationPresets({ onPresetSelect }: {
  onPresetSelect: (preset: any) => void;
}) {
  const presets = [
    {
      id: 'minimal',
      name: 'Minimal',
      icon: '⚪',
      description: 'Clean, simple design'
    },
    {
      id: 'professional',
      name: 'Professional',
      icon: '👔',
      description: 'Business-ready styling'
    },
    {
      id: 'colorful',
      name: 'Colorful',
      icon: '🌈',
      description: 'Vibrant color scheme'
    },
    {
      id: 'dark',
      name: 'Dark Theme',
      icon: '🌙',
      description: 'Dark background theme'
    }
  ];

  return (
    <div className="grid grid-cols-2 gap-2">
      {presets.map((preset) => (
        <Button
          key={preset.id}
          variant="outline"
          onClick={() => onPresetSelect(preset)}
          className="h-auto p-3 flex flex-col items-center space-y-1"
        >
          <span className="text-xl">{preset.icon}</span>
          <span className="text-xs font-medium">{preset.name}</span>
        </Button>
      ))}
    </div>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Configuration Panel Component                                                      │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function ConfigurationPanel({ 
  className = '',
  onConfigChange,
  onPlotTypeChange,
  onGeneratePlot 
}: ConfigurationPanelProps) {
  const { session, updateSession } = useSession();

  const handleTypeSelect = (type: PlotType) => {
    updateSession({ plotType: type });
    onPlotTypeChange?.(type);
  };

  const handleConfigChange = (config: any) => {
    updateSession({ plotConfig: config });
    onConfigChange?.(config);
  };

  const handlePresetSelect = (preset: any) => {
    console.log('Apply preset:', preset.id);
    // TODO: Apply preset configuration
  };

  return (
    <div className={`h-full overflow-y-auto p-4 space-y-4 ${className}`}>
      {/* Plot Type Selection */}
      <ConfigSection title="Plot Type" icon="📊" defaultExpanded={true}>
        <PlotTypeGrid
          selectedType={(session?.plotType as PlotType) || null}
          onTypeSelect={handleTypeSelect}
        />
        
        {session?.suggestedPlotType && (
          <div className="mt-3 p-2 bg-[var(--color-primary-alpha-10)] rounded border border-[var(--color-primary)]">
            <div className="text-xs text-[var(--color-primary)] font-medium mb-1">
              💡 Suggested for your data:
            </div>
            <div className="text-xs text-[var(--color-text-secondary)]">
              {session.suggestedPlotType} chart would work well with your data structure
            </div>
          </div>
        )}
      </ConfigSection>

      {/* Quick Settings */}
      <ConfigSection title="Quick Settings" icon="⚡" defaultExpanded={true}>
        <QuickSettings
          session={session}
          onConfigChange={handleConfigChange}
        />
      </ConfigSection>

      {/* Style Presets */}
      <ConfigSection title="Style Presets" icon="🎨" defaultExpanded={false}>
        <ConfigurationPresets onPresetSelect={handlePresetSelect} />
      </ConfigSection>

      {/* Advanced Configuration */}
      {session?.plotType && session?.uploadedData && (
        <ConfigSection title="Advanced Settings" icon="⚙️" defaultExpanded={false}>
          <PlotConfiguration
            plotType={session.plotType}
            data={session.uploadedData.data}
            columns={session.uploadedData.columns}
            onConfigChange={handleConfigChange}
          />
        </ConfigSection>
      )}

      {/* Configuration Summary */}
      {session?.plotConfig && (
        <ConfigSection title="Configuration Summary" icon="📋" defaultExpanded={false}>
          <div className="space-y-2 text-sm">
            {Object.entries(session.plotConfig).map(([key, value]) => (
              <div key={key} className="flex justify-between">
                <span className="text-[var(--color-text-muted)] capitalize">
                  {key.replace('_', ' ')}:
                </span>
                <span className="text-[var(--color-text-secondary)] truncate ml-2">
                  {String(value)}
                </span>
              </div>
            ))}
          </div>
        </ConfigSection>
      )}

      {/* Action Buttons */}
      <div className="space-y-2 pt-4 border-t border-[var(--color-border)]">
        <Button 
          variant="primary" 
          className="w-full"
          disabled={!session?.uploadedData || !session?.plotType}
          onClick={onGeneratePlot}
        >
          🎯 Generate Plot
        </Button>
        
        <div className="grid grid-cols-2 gap-2">
          <Button variant="outline" size="sm">
            💾 Save Config
          </Button>
          <Button variant="outline" size="sm">
            🔄 Reset
          </Button>
        </div>
      </div>

      {/* Help Section */}
      <Card className="bg-[var(--color-bg-tertiary)]">
        <div className="p-3">
          <h4 className="text-sm font-semibold text-[var(--color-text-primary)] mb-2">
            💡 Quick Tips
          </h4>
          <ul className="text-xs text-[var(--color-text-muted)] space-y-1">
            <li>• Select columns that best represent your data</li>
            <li>• Try different plot types to find the best fit</li>
            <li>• Use presets for quick styling</li>
            <li>• Check the preview as you make changes</li>
          </ul>
        </div>
      </Card>
    </div>
  );
}