/**
 * Dashboard Layout
 * ---
 * bwr-plots/frontend/src/components/layout/DashboardLayout.tsx
 * ---
 * Three-column dashboard layout with independent scrolling
 */

'use client';

import React, { ReactNode } from 'react';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Type Definitions                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

interface DashboardLayoutProps {
  dataPanel: ReactNode;
  plotCanvas: ReactNode;
  configPanel: ReactNode;
  onExportPlot?: () => void;
  onSaveProject?: () => void;
  onNewProject?: () => void;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Dashboard Layout Component                                                         │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function DashboardLayout({
  dataPanel,
  plotCanvas,
  configPanel,
}: DashboardLayoutProps) {
  return (
    <div className="h-full w-full flex bg-[var(--color-bg-primary)]">
      {/* Left Column - Data Panel */}
      <div className="w-80 flex-shrink-0 h-full bg-[var(--color-bg-secondary)] border-r border-[var(--color-border)]">
        <div className="h-full flex flex-col">
          <div className="px-4 py-3 border-b border-[var(--color-border)]">
            <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">
              Data
            </h2>
          </div>
          <div className="flex-1 overflow-y-auto">
            {dataPanel}
          </div>
        </div>
      </div>

      {/* Middle Column - Plot Canvas */}
      <div className="flex-1 h-full flex flex-col bg-[var(--color-bg-primary)]">
        <div className="px-4 py-3 border-b border-[var(--color-border)]">
          <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">
            Plotting
          </h2>
        </div>
        <div className="flex-1 overflow-y-auto">
          {plotCanvas}
        </div>
      </div>

      {/* Right Column - Configuration Panel */}
      <div className="w-96 flex-shrink-0 h-full bg-[var(--color-bg-secondary)] border-l border-[var(--color-border)]">
        <div className="h-full flex flex-col">
          <div className="px-4 py-3 border-b border-[var(--color-border)]">
            <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">
              Configuration
            </h2>
          </div>
          <div className="flex-1 overflow-y-auto">
            {configPanel}
          </div>
        </div>
      </div>
    </div>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Mobile Responsive Layout (Future Enhancement)                                      │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function usePanelState() {
  // This can be used later for mobile responsive behavior
  return {
    panelState: { dataPanel: true, configPanel: true },
    togglePanel: () => {},
    showPanel: () => {},
    hidePanel: () => {}
  };
}