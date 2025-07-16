/**
 * Plot Canvas
 * ---
 * bwr-plots/frontend/src/components/dashboard/PlotCanvas.tsx
 * ---
 * Center canvas component for live plot preview and visualization
 */

'use client';

import React, { useState, useRef } from 'react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { PlotDisplay } from '@/components/plotting/PlotDisplay';
import { useSession } from '@/hooks/useSession';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Type Definitions                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

interface PlotCanvasProps {
  className?: string;
  plotHtml?: string;
  isGenerating?: boolean;
  onExportPlot?: () => void;
  onRegeneratePlot?: () => void;
}

interface PlotControlsProps {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onResetZoom: () => void;
  onFullscreen: () => void;
  zoomLevel: number;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Plot Controls Component                                                            │
// └────────────────────────────────────────────────────────────────────────────────────┘

function PlotControls({ onZoomIn, onZoomOut, onResetZoom, onFullscreen, zoomLevel }: PlotControlsProps) {
  return (
    <div className="absolute top-4 right-4 bg-[var(--color-bg-elevated)] rounded-lg shadow-lg border border-[var(--color-border)] p-2 flex items-center space-x-2 z-10">
      <Button
        variant="ghost"
        size="sm"
        onClick={onZoomOut}
        disabled={zoomLevel <= 0.5}
        title="Zoom Out"
      >
        🔍−
      </Button>
      
      <span className="text-xs text-[var(--color-text-muted)] min-w-[3rem] text-center">
        {Math.round(zoomLevel * 100)}%
      </span>
      
      <Button
        variant="ghost"
        size="sm"
        onClick={onZoomIn}
        disabled={zoomLevel >= 2}
        title="Zoom In"
      >
        🔍+
      </Button>
      
      <div className="w-px h-4 bg-[var(--color-border)]" />
      
      <Button
        variant="ghost"
        size="sm"
        onClick={onResetZoom}
        title="Reset Zoom"
      >
        ↺
      </Button>
      
      <Button
        variant="ghost"
        size="sm"
        onClick={onFullscreen}
        title="Fullscreen"
      >
        ⛶
      </Button>
    </div>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Export Options Component                                                           │
// └────────────────────────────────────────────────────────────────────────────────────┘

function ExportOptions({ onExport }: { onExport: (format: string) => void }) {
  const [showExportMenu, setShowExportMenu] = useState(false);

  const exportFormats = [
    { format: 'png', label: 'PNG Image', icon: '🖼️' },
    { format: 'svg', label: 'SVG Vector', icon: '⚡' },
    { format: 'pdf', label: 'PDF Document', icon: '📄' },
    { format: 'html', label: 'Interactive HTML', icon: '🌐' }
  ];

  return (
    <div className="relative">
      <Button
        variant="primary"
        onClick={() => setShowExportMenu(!showExportMenu)}
        className="gap-2"
      >
        📥 Export
        <span className="text-xs">▼</span>
      </Button>
      
      {showExportMenu && (
        <div className="absolute bottom-full right-0 mb-2 bg-[var(--color-bg-elevated)] rounded-lg shadow-lg border border-[var(--color-border)] p-2 min-w-[180px] z-20">
          <div className="space-y-1">
            {exportFormats.map(({ format, label, icon }) => (
              <Button
                key={format}
                variant="ghost"
                size="sm"
                onClick={() => {
                  onExport(format);
                  setShowExportMenu(false);
                }}
                className="w-full justify-start gap-2"
              >
                <span>{icon}</span>
                {label}
              </Button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Plot Status Component                                                              │
// └────────────────────────────────────────────────────────────────────────────────────┘

function PlotStatus({ 
  hasData, 
  hasPlotType, 
  isGenerating, 
  hasPlot,
  onGeneratePlot 
}: {
  hasData: boolean;
  hasPlotType: boolean;
  isGenerating: boolean;
  hasPlot: boolean;
  onGeneratePlot: () => void;
}) {
  if (isGenerating) {
    return (
      <div className="flex items-center space-x-3 text-[var(--color-text-muted)]">
        <LoadingSpinner size="sm" />
        <span>Generating plot...</span>
      </div>
    );
  }

  if (!hasData) {
    return (
      <div className="text-[var(--color-text-muted)]">
        📊 Upload data to see plot preview
      </div>
    );
  }

  if (!hasPlotType) {
    return (
      <div className="text-[var(--color-text-muted)]">
        ⚙️ Select a plot type to generate visualization
      </div>
    );
  }

  if (!hasPlot) {
    return (
      <div className="flex items-center space-x-3">
        <span className="text-[var(--color-text-muted)]">
          Ready to generate plot
        </span>
        <Button variant="primary" size="sm" onClick={onGeneratePlot}>
          Generate
        </Button>
      </div>
    );
  }

  return (
    <div className="text-[var(--color-success)] text-sm">
      ✅ Plot generated successfully
    </div>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Plot Canvas Component                                                              │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function PlotCanvas({ 
  className = '',
  plotHtml,
  isGenerating = false,
  onExportPlot,
  onRegeneratePlot 
}: PlotCanvasProps) {
  const { session } = useSession();
  const [zoomLevel, setZoomLevel] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const canvasRef = useRef<HTMLDivElement>(null);

  // Plot state checks
  const hasData = Boolean(session?.uploadedData?.data?.length);
  const hasPlotType = Boolean(session?.plotType);
  const hasPlot = Boolean(plotHtml);

  // Zoom controls
  const handleZoomIn = () => setZoomLevel(prev => Math.min(prev + 0.25, 2));
  const handleZoomOut = () => setZoomLevel(prev => Math.max(prev - 0.25, 0.5));
  const handleResetZoom = () => setZoomLevel(1);
  
  const handleFullscreen = () => {
    if (canvasRef.current) {
      if (!isFullscreen) {
        canvasRef.current.requestFullscreen?.();
      } else {
        document.exitFullscreen?.();
      }
      setIsFullscreen(!isFullscreen);
    }
  };

  const handleExport = (format: string) => {
    console.log('Export plot as:', format);
    onExportPlot?.();
  };

  const handleGeneratePlot = () => {
    console.log('Generate plot requested');
    onRegeneratePlot?.();
  };

  return (
    <div className={`h-full overflow-y-auto ${className}`} ref={canvasRef}>
      <div className="h-full flex flex-col">
        {/* Canvas Header */}
        <div className="p-4 border-b border-[var(--color-border)] bg-[var(--color-bg-secondary)]">
          <div className="flex items-center justify-between">
            <PlotStatus
              hasData={hasData}
              hasPlotType={hasPlotType}
              isGenerating={isGenerating}
              hasPlot={hasPlot}
              onGeneratePlot={handleGeneratePlot}
            />
            
            {hasPlot && onExportPlot && (
              <ExportOptions onExport={handleExport} />
            )}
          </div>
        </div>

        {/* Canvas Content */}
        <div className="flex-1 relative overflow-hidden">
        {isGenerating ? (
          /* Loading State */
          <div className="absolute inset-0 flex items-center justify-center bg-[var(--color-bg-primary)]">
            <div className="text-center">
              <LoadingSpinner size="lg" />
              <p className="mt-4 text-[var(--color-text-muted)]">
                Generating your visualization...
              </p>
            </div>
          </div>
        ) : hasPlot && plotHtml ? (
          /* Plot Display */
          <div className="absolute inset-0">
            <PlotControls
              onZoomIn={handleZoomIn}
              onZoomOut={handleZoomOut}
              onResetZoom={handleResetZoom}
              onFullscreen={handleFullscreen}
              zoomLevel={zoomLevel}
            />
            
            <div 
              className="w-full h-full overflow-auto p-4"
              style={{ 
                transform: `scale(${zoomLevel})`,
                transformOrigin: 'top left'
              }}
            >
              <PlotDisplay 
                plotHtml={plotHtml}
                plotData={session?.uploadedData?.data}
              />
            </div>
          </div>
        ) : hasData && hasPlotType ? (
          /* Ready to Generate */
          <div className="absolute inset-0 flex items-center justify-center">
            <Card className="p-8 text-center max-w-md">
              <div className="text-6xl mb-4">📊</div>
              <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-2">
                Ready to Generate Plot
              </h3>
              <p className="text-[var(--color-text-muted)] mb-4">
                Your data and plot type are configured. Click generate to create your visualization.
              </p>
              <Button variant="primary" onClick={handleGeneratePlot}>
                Generate Plot
              </Button>
            </Card>
          </div>
        ) : !hasData ? (
          /* No Data State */
          <div className="absolute inset-0 flex items-center justify-center">
            <Card className="p-8 text-center max-w-md">
              <div className="text-6xl mb-4">📁</div>
              <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-2">
                Upload Your Data
              </h3>
              <p className="text-[var(--color-text-muted)]">
                Start by uploading a CSV or Excel file in the Data panel to see your visualization here.
              </p>
            </Card>
          </div>
        ) : (
          /* No Plot Type State */
          <div className="absolute inset-0 flex items-center justify-center">
            <Card className="p-8 text-center max-w-md">
              <div className="text-6xl mb-4">⚙️</div>
              <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-2">
                Choose Plot Type
              </h3>
              <p className="text-[var(--color-text-muted)]">
                Select a plot type in the Configuration panel to generate your visualization.
              </p>
            </Card>
          </div>
        )}
      </div>

      {/* Canvas Footer */}
      {hasPlot && (
        <div className="p-3 border-t border-[var(--color-border)] bg-[var(--color-bg-secondary)]">
          <div className="flex items-center justify-between text-sm text-[var(--color-text-muted)]">
            <div>
              Plot generated • {session?.plotType || 'Unknown type'}
            </div>
            <div className="flex items-center space-x-4">
              <span>Zoom: {Math.round(zoomLevel * 100)}%</span>
              {onRegeneratePlot && (
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={onRegeneratePlot}
                  disabled={isGenerating}
                >
                  🔄 Regenerate
                </Button>
              )}
            </div>
          </div>
        </div>
      )}
      </div>
    </div>
  );
}