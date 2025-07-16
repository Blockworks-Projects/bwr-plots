/**
 * Responsive Utilities
 * ---
 * bwr-plots/frontend/src/components/layout/utils/responsive.ts
 * ---
 * Utility functions for responsive design and breakpoint handling
 */

import { ResponsiveValue, Breakpoint } from '../types';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Constants                                                                          │
// └────────────────────────────────────────────────────────────────────────────────────┘

const BREAKPOINT_ORDER: Breakpoint[] = ['sm', 'md', 'lg', 'xl', '2xl'];

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Responsive Value Resolution                                                        │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Resolve a responsive value based on the current breakpoint
 */
export function resolveResponsiveValue<T>(
  value: ResponsiveValue<T> | undefined,
  breakpoint: Breakpoint
): T | undefined {
  if (value === undefined) return undefined;
  
  // If it's not an object, return the value directly
  if (typeof value !== 'object' || value === null) {
    return value as T;
  }
  
  // Cast to responsive object
  const responsiveValue = value as Partial<Record<Breakpoint, T>>;
  
  // Get the current breakpoint index
  const currentIndex = BREAKPOINT_ORDER.indexOf(breakpoint);
  
  // Look for the value at current breakpoint or fall back to smaller ones
  for (let i = currentIndex; i >= 0; i--) {
    const bp = BREAKPOINT_ORDER[i];
    if (bp in responsiveValue && responsiveValue[bp] !== undefined) {
      return responsiveValue[bp];
    }
  }
  
  // If no matching breakpoint found, try larger breakpoints
  for (let i = currentIndex + 1; i < BREAKPOINT_ORDER.length; i++) {
    const bp = BREAKPOINT_ORDER[i];
    if (bp in responsiveValue && responsiveValue[bp] !== undefined) {
      return responsiveValue[bp];
    }
  }
  
  return undefined;
}

/**
 * Create a responsive value object from an array of values
 */
export function createResponsiveValue<T>(
  values: [T?, T?, T?, T?, T?]
): ResponsiveValue<T> {
  const [sm, md, lg, xl, xl2] = values;
  
  const result: Partial<Record<Breakpoint, T>> = {};
  
  if (sm !== undefined) result.sm = sm;
  if (md !== undefined) result.md = md;
  if (lg !== undefined) result.lg = lg;
  if (xl !== undefined) result.xl = xl;
  if (xl2 !== undefined) result['2xl'] = xl2;
  
  return result;
}

/**
 * Check if a value is a responsive value object
 */
export function isResponsiveValue<T>(value: any): value is Partial<Record<Breakpoint, T>> {
  if (typeof value !== 'object' || value === null) return false;
  
  const keys = Object.keys(value);
  return keys.some(key => BREAKPOINT_ORDER.includes(key as Breakpoint));
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ CSS Generation Utilities                                                           │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Generate CSS media queries for responsive values
 */
export function generateResponsiveCSS<T>(
  property: string,
  value: ResponsiveValue<T>,
  formatValue?: (val: T) => string
): Record<string, string> {
  const css: Record<string, string> = {};
  
  if (typeof value !== 'object' || value === null) {
    css.default = `${property}: ${formatValue ? formatValue(value as T) : value}`;
    return css;
  }
  
  const responsiveValue = value as Partial<Record<Breakpoint, T>>;
  
  // Base styles (mobile first)
  if (responsiveValue.sm !== undefined) {
    css.default = `${property}: ${formatValue ? formatValue(responsiveValue.sm) : responsiveValue.sm}`;
  }
  
  // Generate media queries
  const breakpoints = {
    md: '768px',
    lg: '1024px',
    xl: '1280px',
    '2xl': '1536px',
  };
  
  Object.entries(breakpoints).forEach(([bp, minWidth]) => {
    const breakpointValue = responsiveValue[bp as Breakpoint];
    if (breakpointValue !== undefined) {
      css[`@media (min-width: ${minWidth})`] = `${property}: ${formatValue ? formatValue(breakpointValue) : breakpointValue}`;
    }
  });
  
  return css;
}

/**
 * Generate CSS custom properties for responsive values
 */
export function generateResponsiveCSSVars<T>(
  varName: string,
  value: ResponsiveValue<T>,
  formatValue?: (val: T) => string
): Record<string, string> {
  const css: Record<string, string> = {};
  
  if (typeof value !== 'object' || value === null) {
    css.default = `--${varName}: ${formatValue ? formatValue(value as T) : value}`;
    return css;
  }
  
  const responsiveValue = value as Partial<Record<Breakpoint, T>>;
  
  // Base custom property
  if (responsiveValue.sm !== undefined) {
    css.default = `--${varName}: ${formatValue ? formatValue(responsiveValue.sm) : responsiveValue.sm}`;
  }
  
  // Generate breakpoint-specific custom properties
  Object.entries(responsiveValue).forEach(([bp, val]) => {
    if (val !== undefined && bp !== 'sm') {
      css[`--${varName}-${bp}`] = `${formatValue ? formatValue(val) : val}`;
    }
  });
  
  return css;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Utility Functions                                                                  │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Get all breakpoints where a responsive value is defined
 */
export function getDefinedBreakpoints<T>(value: ResponsiveValue<T>): Breakpoint[] {
  if (typeof value !== 'object' || value === null) return [];
  
  const responsiveValue = value as Partial<Record<Breakpoint, T>>;
  
  return BREAKPOINT_ORDER.filter(bp => responsiveValue[bp] !== undefined);
}

/**
 * Convert a responsive value to a flat object with all breakpoints
 */
export function expandResponsiveValue<T>(
  value: ResponsiveValue<T>,
  currentBreakpoint: Breakpoint
): Record<Breakpoint, T | undefined> {
  const result: Record<Breakpoint, T | undefined> = {
    sm: undefined,
    md: undefined,
    lg: undefined,
    xl: undefined,
    '2xl': undefined,
  };
  
  BREAKPOINT_ORDER.forEach(bp => {
    result[bp] = resolveResponsiveValue(value, bp);
  });
  
  return result;
}

/**
 * Merge multiple responsive values
 */
export function mergeResponsiveValues<T>(
  ...values: (ResponsiveValue<T> | undefined)[]
): ResponsiveValue<T> {
  const result: Partial<Record<Breakpoint, T>> = {};
  
  values.forEach(value => {
    if (value === undefined) return;
    
    if (typeof value !== 'object' || value === null) {
      // If it's a primitive value, apply to all breakpoints
      BREAKPOINT_ORDER.forEach(bp => {
        if (result[bp] === undefined) {
          result[bp] = value as T;
        }
      });
    } else {
      // If it's a responsive object, merge the values
      const responsiveValue = value as Partial<Record<Breakpoint, T>>;
      Object.entries(responsiveValue).forEach(([bp, val]) => {
        if (val !== undefined) {
          result[bp as Breakpoint] = val;
        }
      });
    }
  });
  
  return result;
}