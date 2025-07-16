/**
 * Plots Tool
 * ---
 * bwr-tools/frontend/src/app/(tools)/plots/page.tsx
 * ---
 * Data visualization tool with three-panel layout
 */

'use client';

import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { DataPanel } from '@/components/dashboard/DataPanel';
import { PlotCanvas } from '@/components/dashboard/PlotCanvas';
import { ConfigurationPanel } from '@/components/dashboard/ConfigurationPanel';
import { useSession } from '@/hooks/useSession';
import { useLivePlot } from '@/hooks/useLivePlot';
import { PlotType } from '@/types/plots';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Plots Tool Page Component                                                          │
// └────────────────────────────────────────────────────────────────────────────────────┘

export default function PlotsTool() {
  const { session, updateSession, clearSession } = useSession();
  
  // Use live plot hook for auto-generation with debouncing
  const {
    generatePlot: handleGeneratePlot,
    isGenerating,
    plotHtml,
    error: plotError,
    canGenerate,
    isPending
  } = useLivePlot({
    debounceDelay: 1500,
    autoGenerate: true,
    onPlotGenerated: (html) => {
      console.log('Plot generated successfully');
    },
    onError: (error) => {
      console.error('Plot generation failed:', error);
    }
  });

  // ┌────────────────────────────────────────────────────────────────────────────────────┐
  // │ Event Handlers                                                                     │
  // └────────────────────────────────────────────────────────────────────────────────────┘

  const handlePlotTypeChange = (type: PlotType) => {
    updateSession({ plotType: type });
  };

  const handleConfigChange = (config: any) => {
    updateSession({ plotConfig: config });
  };

  const handleExportPlot = () => {
    if (plotHtml) {
      // For now, download as HTML
      const blob = new Blob([plotHtml], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${session?.plotConfig?.title || 'plot'}.html`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  const handleSaveProject = () => {
    // Save to localStorage for now
    if (session) {
      localStorage.setItem('bwr-plots-project', JSON.stringify(session));
      alert('Project saved to browser storage!');
    }
  };

  const handleNewProject = () => {
    if (confirm('Start a new project? This will clear your current work.')) {
      clearSession();
    }
  };

  // ┌────────────────────────────────────────────────────────────────────────────────────┐
  // │ Panel Components                                                                   │
  // └────────────────────────────────────────────────────────────────────────────────────┘

  const dataPanel = (
    <DataPanel />
  );

  const plotCanvas = (
    <PlotCanvas
      plotHtml={plotHtml || undefined}
      isGenerating={isGenerating}
      onExportPlot={handleExportPlot}
      onRegeneratePlot={handleGeneratePlot}
    />
  );

  const configPanel = (
    <ConfigurationPanel
      onConfigChange={handleConfigChange}
      onPlotTypeChange={handlePlotTypeChange}
      onGeneratePlot={handleGeneratePlot}
    />
  );

  // ┌────────────────────────────────────────────────────────────────────────────────────┐
  // │ Render Dashboard                                                                   │
  // └────────────────────────────────────────────────────────────────────────────────────┘

  return (
    <DashboardLayout
      dataPanel={dataPanel}
      plotCanvas={plotCanvas}
      configPanel={configPanel}
      onExportPlot={handleExportPlot}
      onSaveProject={handleSaveProject}
      onNewProject={handleNewProject}
    />
  );
}