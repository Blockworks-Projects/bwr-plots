/**
 * Layout System Exports
 * ---
 * bwr-plots/frontend/src/components/layout/index.ts
 * ---
 * Public API exports for the layout system
 */

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Core Components                                                                    │
// └────────────────────────────────────────────────────────────────────────────────────┘

export { ViewportProvider, useViewport } from './primitives/ViewportProvider';
export { AppShell } from './primitives/AppShell';
export { FlexLayout, FlexItem } from './primitives/FlexLayout';
export { ScrollArea } from './primitives/ScrollArea';
export { Panel } from './primitives/Panel';
export { Grid, GridItem } from './primitives/Grid';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Hooks                                                                              │
// └────────────────────────────────────────────────────────────────────────────────────┘

export { useResizeObserver } from './hooks/useResizeObserver';
export { useScrollPosition } from './hooks/useScrollPosition';
export { useBreakpoint, getBreakpointValue, matchesBreakpoint, getBreakpointFromWidth } from './hooks/useBreakpoint';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Utilities                                                                          │
// └────────────────────────────────────────────────────────────────────────────────────┘

export {
  BREAKPOINTS,
  getBreakpoint,
  getDeviceType,
  isAboveBreakpoint,
  isBelowBreakpoint,
  isBetweenBreakpoints,
  getScrollPosition,
  scrollToPosition,
  scrollIntoView,
  getElementDimensions,
  getElementContentDimensions,
  getElementScrollDimensions,
  isElementInViewport,
  getElementViewportIntersection,
} from './utils/viewport';

export {
  resolveResponsiveValue,
  createResponsiveValue,
  isResponsiveValue,
  generateResponsiveCSS,
  generateResponsiveCSSVars,
  getDefinedBreakpoints,
  expandResponsiveValue,
  mergeResponsiveValues,
} from './utils/responsive';

export {
  requestAnimationFrame,
  cancelAnimationFrame,
  throttleRAF,
  debounce,
  throttle,
  createIntersectionObserver,
  observeElementVisibility,
  createResizeObserver,
  observeElementSize,
  WeakCache,
  LRUCache,
  supportsCSSContainment,
  applyCSSContainment,
  supportsContentVisibility,
  applyContentVisibility,
} from './utils/performance';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Types                                                                              │
// └────────────────────────────────────────────────────────────────────────────────────┘

export type {
  Breakpoint,
  ViewportState,
  ViewportProviderProps,
  AppShellProps,
  FlexLayoutProps,
  FlexItemProps,
  ScrollAreaProps,
  ScrollEvent,
  VirtualizerOptions,
  PanelProps,
  GridProps,
  GridItemProps,
  ResponsiveValue,
  ResizeEntry,
  UseResizeObserverOptions,
  UseScrollPositionOptions,
  UseBreakpointOptions,
} from './types';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Legacy Components (for compatibility)                                              │
// └────────────────────────────────────────────────────────────────────────────────────┘

export { DashboardLayout } from './DashboardLayout';
export { Header } from './Header';
export { Sidebar } from './Sidebar';
export { Footer } from './Footer'; 