/**
 * Layout System Type Definitions
 * ---
 * bwr-plots/frontend/src/components/layout/types/index.ts
 * ---
 * Core TypeScript interfaces and types for the layout system
 */

import { ReactNode, ReactElement, CSSProperties } from 'react';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Viewport Types                                                                     │
// └────────────────────────────────────────────────────────────────────────────────────┘

export type Breakpoint = 'sm' | 'md' | 'lg' | 'xl' | '2xl';

export interface ViewportState {
  width: number;
  height: number;
  scrollY: number;
  scrollX: number;
  breakpoint: Breakpoint;
  isDesktop: boolean;
  isMobile: boolean;
  isTablet: boolean;
}

export interface ViewportProviderProps {
  children: ReactNode;
  debounceMs?: number;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ AppShell Types                                                                     │
// └────────────────────────────────────────────────────────────────────────────────────┘

export interface AppShellProps {
  children: ReactNode;
  header?: ReactNode;
  sidebar?: ReactNode;
  footer?: ReactNode;
  sidebarPosition?: 'left' | 'right';
  sidebarCollapsed?: boolean;
  sidebarWidth?: number | string;
  headerHeight?: number;
  footerHeight?: number;
  className?: string;
  style?: CSSProperties;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ FlexLayout Types                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

export interface FlexLayoutProps {
  children: ReactElement<FlexItemProps> | ReactElement<FlexItemProps>[];
  orientation?: 'horizontal' | 'vertical';
  gap?: number;
  wrap?: boolean;
  align?: 'start' | 'center' | 'end' | 'stretch';
  justify?: 'start' | 'center' | 'end' | 'between' | 'around' | 'evenly';
  resizable?: boolean;
  onResize?: (sizes: number[]) => void;
  className?: string;
  style?: CSSProperties;
}

export interface FlexItemProps {
  children: ReactNode;
  size?: number | string;
  grow?: number;
  shrink?: number;
  basis?: number | string;
  minSize?: number | string;
  maxSize?: number | string;
  resizable?: boolean;
  className?: string;
  style?: CSSProperties;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ ScrollArea Types                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

export interface ScrollEvent {
  scrollTop: number;
  scrollLeft: number;
  scrollHeight: number;
  scrollWidth: number;
  clientHeight: number;
  clientWidth: number;
}

export interface VirtualizerOptions {
  itemCount: number;
  itemSize: number | ((index: number) => number);
  overscan?: number;
  horizontal?: boolean;
}

export interface ScrollAreaProps {
  children: ReactNode;
  height?: number | string;
  width?: number | string;
  maxHeight?: number | string;
  maxWidth?: number | string;
  scrollbarWidth?: 'thin' | 'none' | 'auto';
  scrollHideDelay?: number;
  onScroll?: (event: ScrollEvent) => void;
  virtualizer?: VirtualizerOptions;
  className?: string;
  style?: CSSProperties;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Panel Types                                                                        │
// └────────────────────────────────────────────────────────────────────────────────────┘

export interface PanelProps {
  children: ReactNode;
  title?: string;
  description?: string;
  actions?: ReactNode;
  padding?: boolean | number | { top?: number; right?: number; bottom?: number; left?: number };
  bordered?: boolean;
  elevated?: boolean;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
  onCollapse?: (collapsed: boolean) => void;
  className?: string;
  style?: CSSProperties;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Grid Types                                                                         │
// └────────────────────────────────────────────────────────────────────────────────────┘

export type ResponsiveValue<T> = T | Partial<Record<Breakpoint, T>>;

export interface GridProps {
  children: ReactNode;
  columns?: ResponsiveValue<number | string>;
  rows?: ResponsiveValue<number | string>;
  gap?: ResponsiveValue<number>;
  rowGap?: ResponsiveValue<number>;
  columnGap?: ResponsiveValue<number>;
  areas?: ResponsiveValue<string[]>;
  autoFlow?: 'row' | 'column' | 'dense' | 'row dense' | 'column dense';
  alignItems?: 'start' | 'center' | 'end' | 'stretch';
  justifyItems?: 'start' | 'center' | 'end' | 'stretch';
  className?: string;
  style?: CSSProperties;
}

export interface GridItemProps {
  children: ReactNode;
  area?: string;
  column?: string;
  row?: string;
  colSpan?: ResponsiveValue<number>;
  rowSpan?: ResponsiveValue<number>;
  className?: string;
  style?: CSSProperties;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Utility Types                                                                      │
// └────────────────────────────────────────────────────────────────────────────────────┘

export interface ResizeEntry {
  width: number;
  height: number;
  target: Element;
}

export interface UseResizeObserverOptions {
  ref: React.RefObject<Element>;
  onResize?: (entry: ResizeEntry) => void;
  debounceMs?: number;
}

export interface UseScrollPositionOptions {
  ref?: React.RefObject<Element>;
  throttleMs?: number;
}

export interface UseBreakpointOptions {
  defaultBreakpoint?: Breakpoint;
}