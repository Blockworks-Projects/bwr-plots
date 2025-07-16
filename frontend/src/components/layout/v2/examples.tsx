/**
 * Layout System Examples
 * ---
 * bwr-plots/frontend/src/components/layout/v2/examples.tsx
 * ---
 * Practical examples of migrating components to the new layout system
 */

'use client';

import React from 'react';
import { 
  LayoutContainer, 
  Panel, 
  FlexLayout, 
  FlexItem, 
  ScrollArea,
  DashboardLayoutAdapter,
  MigrationWrapper
} from './index';
import { DataPanel } from '@/components/dashboard/DataPanel';
import { PlotCanvas } from '@/components/dashboard/PlotCanvas';
import { ConfigurationPanel } from '@/components/dashboard/ConfigurationPanel';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Example 1: Migrated DataPanel                                                      │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function MigratedDataPanel() {
  return (
    <Panel
      id="data-panel"
      title="Data"
      icon="📁"
      padding={false}
      className="bg-[var(--color-bg-secondary)] border-r border-[var(--color-border)]"
    >
      <ScrollArea persistKey="data-panel-scroll">
        <DataPanel className="p-0" />
      </ScrollArea>
    </Panel>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Example 2: Migrated PlotCanvas                                                     │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function MigratedPlotCanvas(props: any) {
  return (
    <Panel
      id="plot-canvas"
      title="Plotting"
      icon="📊"
      padding={false}
      className="bg-[var(--color-bg-primary)]"
    >
      <PlotCanvas {...props} className="h-full" />
    </Panel>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Example 3: Migrated ConfigurationPanel                                            │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function MigratedConfigurationPanel(props: any) {
  return (
    <Panel
      id="config-panel"
      title="Configuration"
      icon="⚙️"
      padding={false}
      className="bg-[var(--color-bg-secondary)] border-l border-[var(--color-border)]"
    >
      <ScrollArea persistKey="config-panel-scroll">
        <ConfigurationPanel {...props} className="p-0" />
      </ScrollArea>
    </Panel>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Example 4: New Flexible Dashboard Layout                                          │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function FlexibleDashboardLayout(props: any) {
  return (
    <LayoutContainer className="h-full">
      <FlexLayout
        direction="row"
        fullHeight
        fullWidth
        gap="none"
        className="bg-[var(--color-bg-primary)]"
      >
        {/* Data Panel */}
        <FlexItem basis={320} shrink={false} style={{ minWidth: 240, maxWidth: 480 }}>
          <MigratedDataPanel />
        </FlexItem>

        {/* Plot Canvas */}
        <FlexItem flex={1} style={{ minWidth: 400 }}>
          <MigratedPlotCanvas {...props} />
        </FlexItem>

        {/* Configuration Panel */}
        <FlexItem basis={384} shrink={false} style={{ minWidth: 300, maxWidth: 500 }}>
          <MigratedConfigurationPanel {...props} />
        </FlexItem>
      </FlexLayout>
    </LayoutContainer>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Example 5: Progressive Migration Component                                        │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function ProgressiveDashboardLayout(props: {
  dataPanel: React.ReactNode;
  plotCanvas: React.ReactNode;
  configPanel: React.ReactNode;
}) {
  const oldLayout = (
    <div className="h-full w-full flex bg-[var(--color-bg-primary)]">
      <div className="w-80 flex-shrink-0 h-full bg-[var(--color-bg-secondary)] border-r border-[var(--color-border)]">
        {props.dataPanel}
      </div>
      <div className="flex-1 h-full flex flex-col bg-[var(--color-bg-primary)]">
        {props.plotCanvas}
      </div>
      <div className="w-96 flex-shrink-0 h-full bg-[var(--color-bg-secondary)] border-l border-[var(--color-border)]">
        {props.configPanel}
      </div>
    </div>
  );

  const newLayout = (
    <FlexibleDashboardLayout {...props} />
  );

  return (
    <MigrationWrapper
      oldComponent={oldLayout}
      newComponent={newLayout}
    />
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Example 6: Adapter Usage                                                          │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function AdapterExampleLayout(props: {
  dataPanel: React.ReactNode;
  plotCanvas: React.ReactNode;
  configPanel: React.ReactNode;
}) {
  return (
    <DashboardLayoutAdapter
      dataPanel={props.dataPanel}
      plotCanvas={props.plotCanvas}
      configPanel={props.configPanel}
      useNewPanels={true}
    />
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Example 7: Responsive Layout                                                      │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function ResponsiveDashboardLayout(props: any) {
  return (
    <LayoutContainer className="h-full">
      <FlexLayout
        direction="row"
        fullHeight
        fullWidth
        gap="none"
        responsive={{
          sm: { direction: 'column' },
          md: { direction: 'row' }
        }}
        className="bg-[var(--color-bg-primary)]"
      >
        {/* On mobile, panels stack vertically */}
        <FlexItem 
          flex={false}
          basis={320}
          style={{ minWidth: 240 }}
          className="sm:h-64 md:h-full"
        >
          <MigratedDataPanel />
        </FlexItem>

        <FlexItem flex={1} style={{ minWidth: 300 }}>
          <MigratedPlotCanvas {...props} />
        </FlexItem>

        <FlexItem 
          flex={false}
          basis={384}
          style={{ minWidth: 300 }}
          className="sm:h-64 md:h-full"
        >
          <MigratedConfigurationPanel {...props} />
        </FlexItem>
      </FlexLayout>
    </LayoutContainer>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Example 8: Testing Components                                                     │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function LayoutTestBench() {
  const [useNew, setUseNew] = React.useState(false);

  return (
    <div className="h-screen flex flex-col">
      {/* Test Controls */}
      <div className="p-4 bg-gray-800 text-white flex items-center space-x-4">
        <h1 className="text-lg font-bold">Layout Test Bench</h1>
        <label className="flex items-center space-x-2">
          <input
            type="checkbox"
            checked={useNew}
            onChange={(e) => setUseNew(e.target.checked)}
          />
          <span>Use New Layout System</span>
        </label>
      </div>

      {/* Layout Area */}
      <div className="flex-1">
        {useNew ? (
          <FlexibleDashboardLayout />
        ) : (
          <div className="h-full w-full flex bg-[var(--color-bg-primary)]">
            <div className="w-80 flex-shrink-0 h-full bg-[var(--color-bg-secondary)] border-r border-[var(--color-border)]">
              <DataPanel />
            </div>
            <div className="flex-1 h-full flex flex-col bg-[var(--color-bg-primary)]">
              <PlotCanvas />
            </div>
            <div className="w-96 flex-shrink-0 h-full bg-[var(--color-bg-secondary)] border-l border-[var(--color-border)]">
              <ConfigurationPanel />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}