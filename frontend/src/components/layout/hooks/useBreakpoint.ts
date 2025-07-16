/**
 * Breakpoint Hook
 * ---
 * bwr-plots/frontend/src/components/layout/hooks/useBreakpoint.ts
 * ---
 * Hook for responsive breakpoint detection and utilities
 */

import { useMemo } from 'react';
import { Breakpoint, UseBreakpointOptions } from '../types';
import { useViewport } from '../primitives/ViewportProvider';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Types                                                                              │
// └────────────────────────────────────────────────────────────────────────────────────┘

interface BreakpointUtils {
  breakpoint: Breakpoint;
  isSm: boolean;
  isMd: boolean;
  isLg: boolean;
  isXl: boolean;
  is2xl: boolean;
  isAbove: (bp: Breakpoint) => boolean;
  isBelow: (bp: Breakpoint) => boolean;
  isDesktop: boolean;
  isTablet: boolean;
  isMobile: boolean;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Constants                                                                          │
// └────────────────────────────────────────────────────────────────────────────────────┘

const BREAKPOINT_ORDER: Breakpoint[] = ['sm', 'md', 'lg', 'xl', '2xl'];

const BREAKPOINT_VALUES = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
} as const;

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ useBreakpoint Hook                                                                 │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function useBreakpoint({
  defaultBreakpoint = 'lg',
}: UseBreakpointOptions = {}): BreakpointUtils {
  const { breakpoint, width, isDesktop, isTablet, isMobile } = useViewport();
  const currentBreakpoint = breakpoint || defaultBreakpoint;

  const utils = useMemo(() => {
    const getCurrentIndex = () => BREAKPOINT_ORDER.indexOf(currentBreakpoint);
    
    return {
      breakpoint: currentBreakpoint,
      isSm: currentBreakpoint === 'sm',
      isMd: currentBreakpoint === 'md',
      isLg: currentBreakpoint === 'lg',
      isXl: currentBreakpoint === 'xl',
      is2xl: currentBreakpoint === '2xl',
      isDesktop,
      isTablet,
      isMobile,
      isAbove: (bp: Breakpoint) => {
        const currentIndex = getCurrentIndex();
        const targetIndex = BREAKPOINT_ORDER.indexOf(bp);
        return currentIndex > targetIndex;
      },
      isBelow: (bp: Breakpoint) => {
        const currentIndex = getCurrentIndex();
        const targetIndex = BREAKPOINT_ORDER.indexOf(bp);
        return currentIndex < targetIndex;
      },
    };
  }, [currentBreakpoint, isDesktop, isTablet, isMobile]);

  return utils;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Utility Functions                                                                  │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Get the breakpoint value in pixels
 */
export function getBreakpointValue(breakpoint: Breakpoint): number {
  return BREAKPOINT_VALUES[breakpoint];
}

/**
 * Check if a width matches a breakpoint
 */
export function matchesBreakpoint(width: number, breakpoint: Breakpoint): boolean {
  const breakpointValue = getBreakpointValue(breakpoint);
  const nextBreakpointIndex = BREAKPOINT_ORDER.indexOf(breakpoint) + 1;
  const nextBreakpoint = BREAKPOINT_ORDER[nextBreakpointIndex];
  
  if (nextBreakpoint) {
    const nextBreakpointValue = getBreakpointValue(nextBreakpoint);
    return width >= breakpointValue && width < nextBreakpointValue;
  } else {
    return width >= breakpointValue;
  }
}

/**
 * Convert a pixel value to the corresponding breakpoint
 */
export function getBreakpointFromWidth(width: number): Breakpoint {
  if (width >= BREAKPOINT_VALUES['2xl']) return '2xl';
  if (width >= BREAKPOINT_VALUES.xl) return 'xl';
  if (width >= BREAKPOINT_VALUES.lg) return 'lg';
  if (width >= BREAKPOINT_VALUES.md) return 'md';
  return 'sm';
}