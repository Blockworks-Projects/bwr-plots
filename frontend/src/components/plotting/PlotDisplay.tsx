'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { LoadingSpinner } from '../ui/LoadingSpinner';

interface PlotDisplayProps {
  plotData?: any;
  plotHtml?: string;
  isLoading?: boolean;
  error?: string | null;
  onExport?: (format: 'html' | 'png' | 'svg' | 'pdf') => void;
  onRegenerateClick?: () => void;
  className?: string;
}

export function PlotDisplay({
  plotData,
  plotHtml,
  isLoading = false,
  error = null,
  onExport,
  onRegenerateClick,
  className = ''
}: PlotDisplayProps) {
  const [plotSize, setPlotSize] = useState({ width: 800, height: 600 });
  const containerRef = useRef<HTMLDivElement>(null);

  // Update plot size when container size changes
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        const container = containerRef.current;
        const width = container.clientWidth - 32; // Account for padding
        const height = Math.max(400, Math.min(800, width * 0.6)); // Maintain aspect ratio
        setPlotSize({ width, height });
      }
    };

    updateSize();
    
    const handleResize = () => updateSize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  const handleExport = (format: 'html' | 'png' | 'svg' | 'pdf') => {
    if (onExport) {
      onExport(format);
    }
  };

  const renderPlotContent = () => {
    if (isLoading) {
      return (
        <div className="flex flex-col items-center justify-center py-16">
          <LoadingSpinner size="lg" />
          <p className="mt-4 text-gray-600">Generating plot...</p>
        </div>
      );
    }

    if (error) {
      return (
        <div className="flex flex-col items-center justify-center py-16">
          <div className="text-red-600 mb-2">
            ⚠️ Plot Generation Failed
          </div>
          <div className="text-sm text-gray-600 text-center max-w-md mb-4">
            {error}
          </div>
          {onRegenerateClick && (
            <Button
              onClick={onRegenerateClick}
              variant="outline"
            >
              Try Again
            </Button>
          )}
        </div>
      );
    }

    if (!plotData && !plotHtml) {
      return (
        <div className="flex flex-col items-center justify-center py-16 text-gray-500">
          <div className="text-4xl mb-4">📊</div>
          <div className="text-lg font-medium">No Plot Data</div>
          <div className="text-sm">Configure and generate a plot to see it here</div>
        </div>
      );
    }

    // Render HTML plot if available
    if (plotHtml) {
      return (
        <div 
          className="w-full"
          dangerouslySetInnerHTML={{ __html: plotHtml }}
          style={{ minHeight: plotSize.height }}
        />
      );
    }

    // Render plot data information if we have it but no HTML
    if (plotData) {
      return (
        <div className="flex flex-col items-center justify-center py-16">
          <div className="text-4xl mb-4">📊</div>
          <div className="text-lg font-medium">Plot Data Available</div>
          <div className="text-sm text-gray-600 text-center max-w-md mb-4">
            Plot data has been generated. The full interactive plot will be available once the HTML rendering is complete.
          </div>
          <div className="bg-gray-50 rounded-lg p-4 mt-4 text-sm">
            <pre className="text-gray-700 whitespace-pre-wrap">
              {JSON.stringify(plotData, null, 2).slice(0, 200)}
              {JSON.stringify(plotData, null, 2).length > 200 && '...'}
            </pre>
          </div>
        </div>
      );
    }

    return null;
  };

  const exportFormats = [
    { value: 'png', label: 'PNG Image', icon: '🖼️' },
    { value: 'svg', label: 'SVG Vector', icon: '📐' },
    { value: 'html', label: 'HTML File', icon: '🌐' },
    { value: 'pdf', label: 'PDF Document', icon: '📄' }
  ] as const;

  return (
    <Card className={className} ref={containerRef}>
      <div className="p-6">
        {/* Header with Export Controls */}
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900">
            Plot Preview
          </h3>
          
          {(plotData || plotHtml) && onExport && (
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-600 mr-2">Export as:</span>
              {exportFormats.map((format) => (
                <Button
                  key={format.value}
                  variant="outline"
                  size="sm"
                  onClick={() => handleExport(format.value)}
                  className="flex items-center space-x-1"
                  title={`Export as ${format.label}`}
                >
                  <span className="text-sm">{format.icon}</span>
                  <span className="hidden sm:inline">{format.value.toUpperCase()}</span>
                </Button>
              ))}
            </div>
          )}
        </div>

        {/* Plot Content */}
        <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
          {renderPlotContent()}
        </div>

        {/* Plot Information */}
        {(plotData || plotHtml) && !isLoading && !error && (
          <div className="mt-4 text-xs text-gray-500 flex items-center justify-between">
            <div>
              Display size: {plotSize.width} × {plotSize.height}px
            </div>
            <div>
              {plotHtml ? 'Interactive HTML plot' : 'Plot data available'}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
} 