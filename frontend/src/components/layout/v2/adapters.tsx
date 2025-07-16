/**
 * Layout Adapters
 * ---
 * bwr-plots/frontend/src/components/layout/v2/adapters.tsx
 * ---
 * Adapter components for bridging old and new layout systems
 */

'use client';

import React from 'react';
import { Panel, PanelProps } from './Panel';
import { ScrollArea } from './ScrollArea';
import { FlexLayout, FlexItem } from './FlexLayout';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Legacy Panel Adapter                                                               │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Wraps new Panel component to work with old DashboardLayout
 * Maintains the same div structure expected by legacy code
 */
export interface LegacyPanelAdapterProps extends Omit<PanelProps, 'legacy'> {
  children: React.ReactNode;
  className?: string;
  wrapperClassName?: string;
}

export function LegacyPanelAdapter({
  children,
  className = '',
  wrapperClassName = '',
  ...panelProps
}: LegacyPanelAdapterProps) {
  // Extract classes that should go on wrapper vs panel
  const isWrapperClass = (cls: string) => {
    return cls.includes('w-') || cls.includes('flex-') || cls.includes('h-');
  };

  const classNames = className.split(' ').filter(Boolean);
  const wrapperClasses = classNames.filter(isWrapperClass);
  const panelClasses = classNames.filter(cls => !isWrapperClass(cls));

  return (
    <div className={[...wrapperClasses, wrapperClassName].join(' ')}>
      <Panel
        {...panelProps}
        className={panelClasses.join(' ')}
        scrollable={panelProps.scrollable ?? true}
        padding={panelProps.padding ?? false}
      >
        {children}
      </Panel>
    </div>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ New Layout Adapter                                                                 │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Wraps old panel components to work in new layout system
 */
export interface NewLayoutAdapterProps {
  type: 'data' | 'plot' | 'config';
  children: React.ReactNode;
  defaultSize?: number;
  minSize?: number;
  maxSize?: number;
}

export function NewLayoutAdapter({
  type,
  children,
  defaultSize,
  minSize,
  maxSize
}: NewLayoutAdapterProps) {
  const sizeDefaults = {
    data: { default: 320, min: 240, max: 480 },
    plot: { default: undefined, min: 400, max: undefined },
    config: { default: 384, min: 300, max: 500 }
  };

  const sizes = sizeDefaults[type];

  return (
    <FlexItem
      flex={type === 'plot' ? 1 : undefined}
      basis={defaultSize ?? sizes.default}
      style={{
        minWidth: minSize ?? sizes.min,
        maxWidth: maxSize ?? sizes.max
      }}
    >
      {children}
    </FlexItem>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Dashboard Layout Adapter                                                           │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Makes old DashboardLayout work with new panel components
 */
export interface DashboardLayoutAdapterProps {
  dataPanel: React.ReactNode;
  plotCanvas: React.ReactNode;
  configPanel: React.ReactNode;
  useNewPanels?: boolean;
}

export function DashboardLayoutAdapter({
  dataPanel,
  plotCanvas,
  configPanel,
  useNewPanels = false
}: DashboardLayoutAdapterProps) {
  if (!useNewPanels) {
    // Return original layout unchanged
    return (
      <div className="h-full w-full flex bg-[var(--color-bg-primary)]">
        <div className="w-80 flex-shrink-0 h-full bg-[var(--color-bg-secondary)] border-r border-[var(--color-border)]">
          {dataPanel}
        </div>
        <div className="flex-1 h-full flex flex-col bg-[var(--color-bg-primary)]">
          {plotCanvas}
        </div>
        <div className="w-96 flex-shrink-0 h-full bg-[var(--color-bg-secondary)] border-l border-[var(--color-border)]">
          {configPanel}
        </div>
      </div>
    );
  }

  // Use new FlexLayout with adapted panels
  return (
    <FlexLayout
      direction="row"
      fullHeight
      fullWidth
      gap="none"
      className="bg-[var(--color-bg-primary)]"
    >
      <NewLayoutAdapter type="data">
        <LegacyPanelAdapter
          title="Data"
          variant="default"
          className="h-full bg-[var(--color-bg-secondary)] border-r border-[var(--color-border)]"
        >
          {dataPanel}
        </LegacyPanelAdapter>
      </NewLayoutAdapter>

      <NewLayoutAdapter type="plot">
        <div className="h-full flex flex-col bg-[var(--color-bg-primary)]">
          {plotCanvas}
        </div>
      </NewLayoutAdapter>

      <NewLayoutAdapter type="config">
        <LegacyPanelAdapter
          title="Configuration"
          variant="default"
          className="h-full bg-[var(--color-bg-secondary)] border-l border-[var(--color-border)]"
        >
          {configPanel}
        </LegacyPanelAdapter>
      </NewLayoutAdapter>
    </FlexLayout>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Scroll Area Adapter                                                                │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Replaces overflow-y-auto divs with managed ScrollArea
 */
export interface ScrollAreaAdapterProps {
  children: React.ReactNode;
  className?: string;
  persistKey?: string;
}

export function ScrollAreaAdapter({
  children,
  className = '',
  persistKey
}: ScrollAreaAdapterProps) {
  // Remove overflow classes as ScrollArea handles this
  const cleanedClasses = className
    .split(' ')
    .filter(cls => !cls.includes('overflow'))
    .join(' ');

  return (
    <ScrollArea
      className={cleanedClasses}
      persistKey={persistKey}
      orientation="vertical"
      restorePosition
    >
      {children}
    </ScrollArea>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Feature Flag Hook                                                                  │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Hook to check if new layout system should be used
 */
export function useNewLayoutSystem() {
  // Check feature flag from environment or localStorage
  if (typeof window !== 'undefined') {
    const flag = localStorage.getItem('bwr-plots-new-layout');
    return flag === 'true';
  }
  
  // Default to false for gradual rollout
  return process.env.NEXT_PUBLIC_NEW_LAYOUT === 'true';
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Migration Helper Components                                                        │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Conditionally renders old or new component based on feature flag
 */
export interface MigrationWrapperProps {
  oldComponent: React.ReactNode;
  newComponent: React.ReactNode;
  forceNew?: boolean;
}

export function MigrationWrapper({
  oldComponent,
  newComponent,
  forceNew = false
}: MigrationWrapperProps) {
  const useNew = forceNew || useNewLayoutSystem();
  return <>{useNew ? newComponent : oldComponent}</>;
}