/**
 * Grid
 * ---
 * bwr-plots/frontend/src/components/layout/primitives/Grid.tsx
 * ---
 * Responsive CSS Grid layout system
 */

'use client';

import React, { useMemo } from 'react';
import { GridProps, GridItemProps, ResponsiveValue, Breakpoint } from '../types';
import { useViewport } from './ViewportProvider';
import styles from '../styles/layout.module.css';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Helper Functions                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

function resolveResponsiveValue<T>(
  value: ResponsiveValue<T> | undefined,
  breakpoint: Breakpoint
): T | undefined {
  if (value === undefined) return undefined;
  
  // If it's not an object, return the value directly
  if (typeof value !== 'object' || value === null) {
    return value as T;
  }
  
  // Type guard to ensure value is Partial<Record<Breakpoint, T>>
  const responsiveValue = value as Partial<Record<Breakpoint, T>>;
  
  // Otherwise, resolve based on breakpoint
  const breakpointOrder: Breakpoint[] = ['2xl', 'xl', 'lg', 'md', 'sm'];
  const currentIndex = breakpointOrder.indexOf(breakpoint);
  
  // Look for the value at current breakpoint or fall back to smaller ones
  for (let i = currentIndex; i < breakpointOrder.length; i++) {
    const bp = breakpointOrder[i];
    if (bp in responsiveValue && responsiveValue[bp] !== undefined) {
      return responsiveValue[bp];
    }
  }
  
  return undefined;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Grid Component                                                                     │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function Grid({
  children,
  columns = 1,
  rows,
  gap,
  rowGap,
  columnGap,
  areas,
  autoFlow,
  alignItems = 'stretch',
  justifyItems = 'stretch',
  className = '',
  style,
}: GridProps) {
  const { breakpoint } = useViewport();

  // Resolve responsive values
  const resolvedColumns = resolveResponsiveValue(columns, breakpoint) ?? 1;
  const resolvedRows = resolveResponsiveValue(rows, breakpoint);
  const resolvedGap = resolveResponsiveValue(gap, breakpoint);
  const resolvedRowGap = resolveResponsiveValue(rowGap, breakpoint);
  const resolvedColumnGap = resolveResponsiveValue(columnGap, breakpoint);
  const resolvedAreas = resolveResponsiveValue(areas, breakpoint);

  // Calculate grid styles
  const gridStyles = useMemo(() => {
    const styles: React.CSSProperties = {
      display: 'grid',
      alignItems,
      justifyItems,
    };

    // Set columns
    if (typeof resolvedColumns === 'number') {
      styles.gridTemplateColumns = `repeat(${resolvedColumns}, minmax(0, 1fr))`;
    } else {
      styles.gridTemplateColumns = resolvedColumns;
    }

    // Set rows
    if (resolvedRows) {
      if (typeof resolvedRows === 'number') {
        styles.gridTemplateRows = `repeat(${resolvedRows}, minmax(0, 1fr))`;
      } else {
        styles.gridTemplateRows = resolvedRows;
      }
    }

    // Set gap
    if (resolvedGap !== undefined) {
      styles.gap = `${resolvedGap}px`;
    } else {
      if (resolvedRowGap !== undefined) {
        styles.rowGap = `${resolvedRowGap}px`;
      }
      if (resolvedColumnGap !== undefined) {
        styles.columnGap = `${resolvedColumnGap}px`;
      }
    }

    // Set areas
    if (resolvedAreas) {
      styles.gridTemplateAreas = resolvedAreas.map(row => `"${row}"`).join(' ');
    }

    // Set auto flow
    if (autoFlow) {
      styles.gridAutoFlow = autoFlow;
    }

    return { ...styles, ...style };
  }, [
    resolvedColumns,
    resolvedRows,
    resolvedGap,
    resolvedRowGap,
    resolvedColumnGap,
    resolvedAreas,
    autoFlow,
    alignItems,
    justifyItems,
    style,
  ]);

  return (
    <div 
      className={`${styles.grid} ${className}`}
      style={gridStyles}
      data-testid="grid"
    >
      {children}
    </div>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ GridItem Component                                                                 │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function GridItem({
  children,
  area,
  column,
  row,
  colSpan,
  rowSpan,
  className = '',
  style,
}: GridItemProps) {
  const { breakpoint } = useViewport();
  
  const resolvedColSpan = resolveResponsiveValue(colSpan, breakpoint);
  const resolvedRowSpan = resolveResponsiveValue(rowSpan, breakpoint);
  
  const itemStyles = useMemo(() => {
    const styles: React.CSSProperties = {};

    if (area) {
      styles.gridArea = area;
    }

    if (column) {
      styles.gridColumn = column;
    } else if (resolvedColSpan) {
      styles.gridColumn = `span ${resolvedColSpan} / span ${resolvedColSpan}`;
    }

    if (row) {
      styles.gridRow = row;
    } else if (resolvedRowSpan) {
      styles.gridRow = `span ${resolvedRowSpan} / span ${resolvedRowSpan}`;
    }

    return { ...styles, ...style };
  }, [area, column, row, resolvedColSpan, resolvedRowSpan, style]);

  return (
    <div 
      className={`${styles.gridItem} ${className}`}
      style={itemStyles}
    >
      {children}
    </div>
  );
}