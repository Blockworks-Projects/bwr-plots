/**
 * Viewport Utilities
 * ---
 * bwr-plots/frontend/src/components/layout/utils/viewport.ts
 * ---
 * Utility functions for viewport calculations and management
 */

import { Breakpoint } from '../types';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Constants                                                                          │
// └────────────────────────────────────────────────────────────────────────────────────┘

export const BREAKPOINTS = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
} as const;

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Viewport Detection Functions                                                       │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Get the current breakpoint based on window width
 */
export function getBreakpoint(width: number): Breakpoint {
  if (width >= BREAKPOINTS['2xl']) return '2xl';
  if (width >= BREAKPOINTS.xl) return 'xl';
  if (width >= BREAKPOINTS.lg) return 'lg';
  if (width >= BREAKPOINTS.md) return 'md';
  return 'sm';
}

/**
 * Determine device type based on viewport width
 */
export function getDeviceType(width: number) {
  return {
    isMobile: width < BREAKPOINTS.md,
    isTablet: width >= BREAKPOINTS.md && width < BREAKPOINTS.lg,
    isDesktop: width >= BREAKPOINTS.lg,
  };
}

/**
 * Check if viewport is at or above a breakpoint
 */
export function isAboveBreakpoint(width: number, breakpoint: Breakpoint): boolean {
  return width >= BREAKPOINTS[breakpoint];
}

/**
 * Check if viewport is below a breakpoint
 */
export function isBelowBreakpoint(width: number, breakpoint: Breakpoint): boolean {
  return width < BREAKPOINTS[breakpoint];
}

/**
 * Check if viewport is between two breakpoints
 */
export function isBetweenBreakpoints(
  width: number,
  min: Breakpoint,
  max: Breakpoint
): boolean {
  return width >= BREAKPOINTS[min] && width < BREAKPOINTS[max];
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Scroll Utilities                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Get the scroll position of an element or window
 */
export function getScrollPosition(element?: Element | Window) {
  if (!element) element = window;
  
  if (element === window) {
    return {
      x: window.scrollX,
      y: window.scrollY,
    };
  }
  
  const el = element as Element;
  return {
    x: el.scrollLeft,
    y: el.scrollTop,
  };
}

/**
 * Scroll to a specific position with smooth animation
 */
export function scrollToPosition(
  x: number,
  y: number,
  element?: Element | Window,
  smooth = true
) {
  if (!element) element = window;
  
  const scrollOptions: ScrollToOptions = {
    left: x,
    top: y,
    behavior: smooth ? 'smooth' : 'auto',
  };
  
  element.scrollTo(scrollOptions);
}

/**
 * Scroll an element into view
 */
export function scrollIntoView(
  element: Element,
  block: ScrollLogicalPosition = 'start',
  inline: ScrollLogicalPosition = 'nearest',
  smooth = true
) {
  element.scrollIntoView({
    block,
    inline,
    behavior: smooth ? 'smooth' : 'auto',
  });
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Dimension Utilities                                                                │
// └────────────────────────────────────────────────────────────────────────────────────┘

/**
 * Get element dimensions including padding and border
 */
export function getElementDimensions(element: Element) {
  const rect = element.getBoundingClientRect();
  return {
    width: rect.width,
    height: rect.height,
    top: rect.top,
    right: rect.right,
    bottom: rect.bottom,
    left: rect.left,
  };
}

/**
 * Get element content dimensions (excluding padding and border)
 */
export function getElementContentDimensions(element: Element) {
  const el = element as HTMLElement;
  return {
    width: el.clientWidth,
    height: el.clientHeight,
  };
}

/**
 * Get element scroll dimensions
 */
export function getElementScrollDimensions(element: Element) {
  const el = element as HTMLElement;
  return {
    width: el.scrollWidth,
    height: el.scrollHeight,
  };
}

/**
 * Check if element is visible in viewport
 */
export function isElementInViewport(element: Element, threshold = 0) {
  const rect = element.getBoundingClientRect();
  const windowHeight = window.innerHeight || document.documentElement.clientHeight;
  const windowWidth = window.innerWidth || document.documentElement.clientWidth;
  
  return (
    rect.top >= -threshold &&
    rect.left >= -threshold &&
    rect.bottom <= windowHeight + threshold &&
    rect.right <= windowWidth + threshold
  );
}

/**
 * Get the intersection ratio of element with viewport
 */
export function getElementViewportIntersection(element: Element) {
  const rect = element.getBoundingClientRect();
  const windowHeight = window.innerHeight || document.documentElement.clientHeight;
  const windowWidth = window.innerWidth || document.documentElement.clientWidth;
  
  const intersectionTop = Math.max(rect.top, 0);
  const intersectionLeft = Math.max(rect.left, 0);
  const intersectionBottom = Math.min(rect.bottom, windowHeight);
  const intersectionRight = Math.min(rect.right, windowWidth);
  
  const intersectionWidth = Math.max(0, intersectionRight - intersectionLeft);
  const intersectionHeight = Math.max(0, intersectionBottom - intersectionTop);
  const intersectionArea = intersectionWidth * intersectionHeight;
  
  const elementArea = rect.width * rect.height;
  
  return elementArea > 0 ? intersectionArea / elementArea : 0;
}