/**
 * Flex Layout
 * ---
 * bwr-plots/frontend/src/components/layout/v2/FlexLayout.tsx
 * ---
 * Flexible row/column layout component with gap and alignment options
 */

'use client';

import React, { forwardRef } from 'react';
import { cn } from '@/lib/utils';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Type Definitions                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

type FlexDirection = 'row' | 'column' | 'row-reverse' | 'column-reverse';
type FlexWrap = 'wrap' | 'nowrap' | 'wrap-reverse';
type FlexAlign = 'start' | 'end' | 'center' | 'stretch' | 'baseline';
type FlexJustify = 'start' | 'end' | 'center' | 'between' | 'around' | 'evenly';
type FlexGap = 'none' | 'xs' | 'sm' | 'md' | 'lg' | 'xl' | number;

export interface FlexLayoutProps extends React.HTMLAttributes<HTMLDivElement> {
  direction?: FlexDirection;
  wrap?: FlexWrap;
  align?: FlexAlign;
  justify?: FlexJustify;
  gap?: FlexGap;
  fullHeight?: boolean;
  fullWidth?: boolean;
  flex?: boolean | number;
  inline?: boolean;
  responsive?: {
    sm?: Partial<FlexLayoutProps>;
    md?: Partial<FlexLayoutProps>;
    lg?: Partial<FlexLayoutProps>;
    xl?: Partial<FlexLayoutProps>;
  };
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Style Mappings                                                                     │
// └────────────────────────────────────────────────────────────────────────────────────┘

const directionClasses: Record<FlexDirection, string> = {
  row: 'flex-row',
  column: 'flex-col',
  'row-reverse': 'flex-row-reverse',
  'column-reverse': 'flex-col-reverse'
};

const wrapClasses: Record<FlexWrap, string> = {
  wrap: 'flex-wrap',
  nowrap: 'flex-nowrap',
  'wrap-reverse': 'flex-wrap-reverse'
};

const alignClasses: Record<FlexAlign, string> = {
  start: 'items-start',
  end: 'items-end',
  center: 'items-center',
  stretch: 'items-stretch',
  baseline: 'items-baseline'
};

const justifyClasses: Record<FlexJustify, string> = {
  start: 'justify-start',
  end: 'justify-end',
  center: 'justify-center',
  between: 'justify-between',
  around: 'justify-around',
  evenly: 'justify-evenly'
};

const gapClasses: Record<string, string> = {
  none: 'gap-0',
  xs: 'gap-1',
  sm: 'gap-2',
  md: 'gap-4',
  lg: 'gap-6',
  xl: 'gap-8'
};

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Flex Layout Component                                                              │
// └────────────────────────────────────────────────────────────────────────────────────┘

export const FlexLayout = forwardRef<HTMLDivElement, FlexLayoutProps>(
  (
    {
      direction = 'row',
      wrap = 'nowrap',
      align = 'stretch',
      justify = 'start',
      gap = 'md',
      fullHeight = false,
      fullWidth = false,
      flex,
      inline = false,
      responsive,
      className,
      style,
      children,
      ...props
    },
    ref
  ) => {
    // Build base classes
    const baseClasses = [
      inline ? 'inline-flex' : 'flex',
      directionClasses[direction],
      wrapClasses[wrap],
      alignClasses[align],
      justifyClasses[justify],
      fullHeight && 'h-full',
      fullWidth && 'w-full'
    ];

    // Handle gap
    if (typeof gap === 'number') {
      // Custom gap value
      style = { ...style, gap: `${gap}px` };
    } else if (gap !== 'none') {
      baseClasses.push(gapClasses[gap]);
    }

    // Handle flex property
    if (flex === true) {
      baseClasses.push('flex-1');
    } else if (typeof flex === 'number') {
      style = { ...style, flex: flex };
    }

    // Build responsive classes
    const responsiveClasses: string[] = [];
    if (responsive) {
      Object.entries(responsive).forEach(([breakpoint, overrides]) => {
        const prefix = breakpoint + ':';
        
        if (overrides.direction) {
          responsiveClasses.push(prefix + directionClasses[overrides.direction]);
        }
        if (overrides.wrap) {
          responsiveClasses.push(prefix + wrapClasses[overrides.wrap]);
        }
        if (overrides.align) {
          responsiveClasses.push(prefix + alignClasses[overrides.align]);
        }
        if (overrides.justify) {
          responsiveClasses.push(prefix + justifyClasses[overrides.justify]);
        }
        if (overrides.gap && typeof overrides.gap === 'string') {
          responsiveClasses.push(prefix + gapClasses[overrides.gap]);
        }
      });
    }

    return (
      <div
        ref={ref}
        className={cn(
          'flex-layout',
          ...baseClasses,
          ...responsiveClasses,
          className
        )}
        style={style}
        {...props}
      >
        {children}
      </div>
    );
  }
);

FlexLayout.displayName = 'FlexLayout';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Flex Item Component                                                                │
// └────────────────────────────────────────────────────────────────────────────────────┘

export interface FlexItemProps extends React.HTMLAttributes<HTMLDivElement> {
  flex?: boolean | number | string;
  grow?: boolean | number;
  shrink?: boolean | number;
  basis?: string | number;
  alignSelf?: 'auto' | FlexAlign;
  order?: number;
}

export const FlexItem = forwardRef<HTMLDivElement, FlexItemProps>(
  (
    {
      flex,
      grow,
      shrink,
      basis,
      alignSelf,
      order,
      className,
      style,
      children,
      ...props
    },
    ref
  ) => {
    const classes: string[] = ['flex-item'];
    const styles: React.CSSProperties = { ...style };

    // Handle flex shorthand
    if (flex === true) {
      classes.push('flex-1');
    } else if (flex === false) {
      classes.push('flex-none');
    } else if (typeof flex === 'number' || typeof flex === 'string') {
      styles.flex = flex;
    }

    // Handle individual flex properties
    if (grow === true) {
      classes.push('grow');
    } else if (grow === false) {
      classes.push('grow-0');
    } else if (typeof grow === 'number') {
      styles.flexGrow = grow;
    }

    if (shrink === true) {
      classes.push('shrink');
    } else if (shrink === false) {
      classes.push('shrink-0');
    } else if (typeof shrink === 'number') {
      styles.flexShrink = shrink;
    }

    if (basis) {
      styles.flexBasis = typeof basis === 'number' ? `${basis}px` : basis;
    }

    // Handle align self
    if (alignSelf && alignSelf !== 'auto') {
      const alignSelfClasses: Record<FlexAlign, string> = {
        start: 'self-start',
        end: 'self-end',
        center: 'self-center',
        stretch: 'self-stretch',
        baseline: 'self-baseline'
      };
      classes.push(alignSelfClasses[alignSelf]);
    }

    // Handle order
    if (order !== undefined) {
      styles.order = order;
    }

    return (
      <div
        ref={ref}
        className={cn(...classes, className)}
        style={styles}
        {...props}
      >
        {children}
      </div>
    );
  }
);

FlexItem.displayName = 'FlexItem';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Utility Components                                                                 │
// └────────────────────────────────────────────────────────────────────────────────────┘

export const FlexSpacer = () => <div className="flex-1" aria-hidden="true" />;

export interface FlexDividerProps {
  orientation?: 'horizontal' | 'vertical';
  size?: 'sm' | 'md' | 'lg';
  color?: string;
  className?: string;
}

export const FlexDivider = forwardRef<HTMLDivElement, FlexDividerProps>(
  ({ orientation = 'vertical', size = 'md', color, className }, ref) => {
    const sizeClasses = {
      sm: orientation === 'vertical' ? 'w-px h-4' : 'h-px w-4',
      md: orientation === 'vertical' ? 'w-px h-6' : 'h-px w-6',
      lg: orientation === 'vertical' ? 'w-px h-8' : 'h-px w-8'
    };

    return (
      <div
        ref={ref}
        className={cn(
          'flex-divider',
          sizeClasses[size],
          'bg-[var(--color-border)]',
          className
        )}
        style={color ? { backgroundColor: color } : undefined}
        aria-hidden="true"
      />
    );
  }
);

FlexDivider.displayName = 'FlexDivider';