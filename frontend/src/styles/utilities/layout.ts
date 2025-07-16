/* ┌────────────────────────────────────────────────────────────────────────────────────┐
   │ Layout Utilities                                                                    │
   │ Helper functions for dynamic layout styling                                         │
   └────────────────────────────────────────────────────────────────────────────────────┘ */

import { clsx, type ClassValue } from 'clsx';

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CLASSNAME UTILITIES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SPACING UTILITIES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export type SpacingScale = 'none' | 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl';

export const spacing = {
  none: '0',
  xs: 'var(--space-1)',
  sm: 'var(--space-2)',
  md: 'var(--space-4)',
  lg: 'var(--space-6)',
  xl: 'var(--space-8)',
  '2xl': 'var(--space-12)',
} as const;

export function getSpacing(scale: SpacingScale): string {
  return spacing[scale];
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FLEXBOX UTILITIES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export interface FlexOptions {
  direction?: 'row' | 'column' | 'row-reverse' | 'column-reverse';
  justify?: 'start' | 'end' | 'center' | 'between' | 'around' | 'evenly';
  align?: 'start' | 'end' | 'center' | 'baseline' | 'stretch';
  wrap?: boolean | 'reverse';
  gap?: SpacingScale;
  inline?: boolean;
}

export function flex(options: FlexOptions = {}): string {
  const {
    direction = 'row',
    justify = 'start',
    align = 'stretch',
    wrap = false,
    gap,
    inline = false,
  } = options;

  const styles: Record<string, string> = {
    display: inline ? 'inline-flex' : 'flex',
    flexDirection: direction,
    justifyContent: justify === 'between' ? 'space-between' : 
                   justify === 'around' ? 'space-around' :
                   justify === 'evenly' ? 'space-evenly' :
                   `flex-${justify}`,
    alignItems: align === 'start' ? 'flex-start' :
                align === 'end' ? 'flex-end' :
                align,
    flexWrap: wrap === true ? 'wrap' : 
              wrap === 'reverse' ? 'wrap-reverse' : 
              'nowrap',
  };

  if (gap) {
    styles.gap = getSpacing(gap);
  }

  return Object.entries(styles)
    .map(([key, value]) => `${key}: ${value}`)
    .join('; ');
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GRID UTILITIES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export interface GridOptions {
  columns?: number | string;
  rows?: number | string;
  gap?: SpacingScale;
  columnGap?: SpacingScale;
  rowGap?: SpacingScale;
  autoFlow?: 'row' | 'column' | 'row dense' | 'column dense';
  alignItems?: 'start' | 'end' | 'center' | 'stretch';
  justifyItems?: 'start' | 'end' | 'center' | 'stretch';
}

export function grid(options: GridOptions = {}): string {
  const {
    columns = 1,
    rows,
    gap,
    columnGap,
    rowGap,
    autoFlow = 'row',
    alignItems = 'stretch',
    justifyItems = 'stretch',
  } = options;

  const styles: Record<string, string> = {
    display: 'grid',
    gridAutoFlow: autoFlow,
    alignItems,
    justifyItems,
  };

  // Handle columns
  if (typeof columns === 'number') {
    styles.gridTemplateColumns = `repeat(${columns}, minmax(0, 1fr))`;
  } else {
    styles.gridTemplateColumns = columns;
  }

  // Handle rows
  if (rows) {
    if (typeof rows === 'number') {
      styles.gridTemplateRows = `repeat(${rows}, minmax(0, 1fr))`;
    } else {
      styles.gridTemplateRows = rows;
    }
  }

  // Handle gaps
  if (gap) {
    styles.gap = getSpacing(gap);
  } else {
    if (columnGap) styles.columnGap = getSpacing(columnGap);
    if (rowGap) styles.rowGap = getSpacing(rowGap);
  }

  return Object.entries(styles)
    .map(([key, value]) => `${key}: ${value}`)
    .join('; ');
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RESPONSIVE UTILITIES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export type Breakpoint = 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl' | '3xl';

export const breakpoints: Record<Breakpoint, string> = {
  xs: '475px',
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px',
  '3xl': '1920px',
};

export function mediaQuery(breakpoint: Breakpoint, styles: string): string {
  return `@media (min-width: ${breakpoints[breakpoint]}) { ${styles} }`;
}

// Responsive value helper
export type ResponsiveValue<T> = T | Partial<Record<Breakpoint, T>>;

export function getResponsiveValue<T>(
  value: ResponsiveValue<T>,
  breakpoint?: Breakpoint
): T | undefined {
  if (value === null || value === undefined) return undefined;
  
  if (typeof value === 'object' && !Array.isArray(value)) {
    const responsiveObj = value as Partial<Record<Breakpoint, T>>;
    if (breakpoint && breakpoint in responsiveObj) {
      return responsiveObj[breakpoint];
    }
    // Return the default (first) value if no breakpoint matches
    return Object.values(responsiveObj)[0] as T;
  }
  
  return value as T;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CONTAINER UTILITIES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export interface ContainerOptions {
  maxWidth?: Breakpoint | 'full';
  padding?: SpacingScale;
  center?: boolean;
}

export function container(options: ContainerOptions = {}): string {
  const {
    maxWidth = 'xl',
    padding = 'md',
    center = true,
  } = options;

  const styles: Record<string, string> = {
    width: '100%',
    paddingLeft: getSpacing(padding),
    paddingRight: getSpacing(padding),
  };

  if (center) {
    styles.marginLeft = 'auto';
    styles.marginRight = 'auto';
  }

  if (maxWidth !== 'full') {
    styles.maxWidth = `var(--screen-${maxWidth})`;
  }

  return Object.entries(styles)
    .map(([key, value]) => `${key}: ${value}`)
    .join('; ');
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// STACK UTILITIES (Vertical spacing)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export interface StackOptions {
  gap?: SpacingScale;
  align?: 'start' | 'end' | 'center' | 'stretch';
  dividers?: boolean;
}

export function stack(options: StackOptions = {}): string {
  const {
    gap = 'md',
    align = 'stretch',
  } = options;

  return flex({
    direction: 'column',
    gap,
    align,
  });
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ASPECT RATIO UTILITIES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
export type AspectRatio = 'square' | 'video' | '4:3' | '21:9' | string;

export function aspectRatio(ratio: AspectRatio): string {
  const ratios: Record<string, string> = {
    square: '1 / 1',
    video: '16 / 9',
    '4:3': '4 / 3',
    '21:9': '21 / 9',
  };

  return `aspect-ratio: ${ratios[ratio] || ratio}`;
}