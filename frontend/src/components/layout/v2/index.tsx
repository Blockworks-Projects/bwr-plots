/**
 * Layout System V2
 * ---
 * bwr-plots/frontend/src/components/layout/v2/index.tsx
 * ---
 * Exports for the new flexible layout system
 */

// Core layout components
export { LayoutContainer, useLayoutContainer, useContainerDimensions, useIsResizing } from './LayoutContainer';
export { Panel, PanelHeader, PanelContent, PanelFooter } from './Panel';
export { FlexLayout, FlexItem, FlexSpacer, FlexDivider } from './FlexLayout';
export { ScrollArea, useScrollObserver } from './ScrollArea';

// Adapter components for migration
export {
  LegacyPanelAdapter,
  NewLayoutAdapter,
  DashboardLayoutAdapter,
  ScrollAreaAdapter,
  MigrationWrapper,
  useNewLayoutSystem
} from './adapters';

// Type exports
export type {
  PanelProps
} from './Panel';

export type {
  FlexLayoutProps,
  FlexItemProps,
  FlexDividerProps
} from './FlexLayout';

export type {
  ScrollAreaProps,
  ScrollPosition,
  ScrollAreaHandle
} from './ScrollArea';

// Re-export from other files for convenience
export type { LegacyPanelAdapterProps, NewLayoutAdapterProps, DashboardLayoutAdapterProps } from './adapters';