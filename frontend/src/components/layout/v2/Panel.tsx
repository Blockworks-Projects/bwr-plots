/**
 * Panel Component
 * ---
 * bwr-plots/frontend/src/components/layout/v2/Panel.tsx
 * ---
 * Base panel component with header, content, and footer areas
 */

'use client';

import React, { forwardRef, ReactNode } from 'react';
import { cn } from '@/lib/utils';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Type Definitions                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

export interface PanelProps {
  id?: string;
  title?: string;
  icon?: string | ReactNode;
  children: ReactNode;
  className?: string;
  headerClassName?: string;
  contentClassName?: string;
  footerClassName?: string;
  header?: ReactNode;
  footer?: ReactNode;
  actions?: ReactNode;
  scrollable?: boolean;
  padding?: boolean | 'sm' | 'md' | 'lg';
  variant?: 'default' | 'elevated' | 'bordered';
  legacy?: boolean; // For backward compatibility
}

interface PanelHeaderProps {
  title?: string;
  icon?: string | ReactNode;
  actions?: ReactNode;
  className?: string;
}

interface PanelContentProps {
  children: ReactNode;
  scrollable?: boolean;
  padding?: boolean | 'sm' | 'md' | 'lg';
  className?: string;
}

interface PanelFooterProps {
  children: ReactNode;
  className?: string;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Panel Header Component                                                             │
// └────────────────────────────────────────────────────────────────────────────────────┘

export const PanelHeader = forwardRef<HTMLDivElement, PanelHeaderProps>(
  ({ title, icon, actions, className }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          'panel-header',
          'flex items-center justify-between',
          'px-4 py-3',
          'border-b border-[var(--color-border)]',
          'bg-[var(--color-bg-secondary)]',
          className
        )}
      >
        <div className="flex items-center space-x-2">
          {icon && (
            <span className="panel-header-icon text-lg">
              {typeof icon === 'string' ? icon : icon}
            </span>
          )}
          {title && (
            <h2 className="panel-header-title text-sm font-semibold text-[var(--color-text-primary)]">
              {title}
            </h2>
          )}
        </div>
        {actions && (
          <div className="panel-header-actions flex items-center space-x-2">
            {actions}
          </div>
        )}
      </div>
    );
  }
);

PanelHeader.displayName = 'PanelHeader';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Panel Content Component                                                            │
// └────────────────────────────────────────────────────────────────────────────────────┘

export const PanelContent = forwardRef<HTMLDivElement, PanelContentProps>(
  ({ children, scrollable = true, padding = 'md', className }, ref) => {
    const paddingClasses = {
      false: '',
      true: 'p-4',
      sm: 'p-2',
      md: 'p-4',
      lg: 'p-6'
    };

    return (
      <div
        ref={ref}
        className={cn(
          'panel-content',
          'flex-1',
          scrollable && 'overflow-y-auto',
          padding && paddingClasses[padding === true ? 'md' : padding],
          className
        )}
      >
        {children}
      </div>
    );
  }
);

PanelContent.displayName = 'PanelContent';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Panel Footer Component                                                             │
// └────────────────────────────────────────────────────────────────────────────────────┘

export const PanelFooter = forwardRef<HTMLDivElement, PanelFooterProps>(
  ({ children, className }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          'panel-footer',
          'px-4 py-3',
          'border-t border-[var(--color-border)]',
          'bg-[var(--color-bg-secondary)]',
          className
        )}
      >
        {children}
      </div>
    );
  }
);

PanelFooter.displayName = 'PanelFooter';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Panel Component                                                                    │
// └────────────────────────────────────────────────────────────────────────────────────┘

export const Panel = forwardRef<HTMLDivElement, PanelProps>(
  (
    {
      id,
      title,
      icon,
      children,
      className,
      headerClassName,
      contentClassName,
      footerClassName,
      header,
      footer,
      actions,
      scrollable = true,
      padding = 'md',
      variant = 'default',
      legacy = false,
      ...props
    },
    ref
  ) => {
    const variantClasses = {
      default: 'bg-[var(--color-bg-primary)]',
      elevated: 'bg-[var(--color-bg-elevated)] shadow-lg',
      bordered: 'bg-[var(--color-bg-primary)] border border-[var(--color-border)]'
    };

    // Legacy mode for backward compatibility
    if (legacy) {
      return (
        <div ref={ref} id={id} className={className} {...props}>
          {children}
        </div>
      );
    }

    // Determine if we need to show header
    const showHeader = header || title || icon || actions;

    return (
      <div
        ref={ref}
        id={id}
        className={cn(
          'panel',
          'flex flex-col',
          'h-full',
          variantClasses[variant],
          className
        )}
        data-panel-id={id}
        {...props}
      >
        {showHeader && (
          header || (
            <PanelHeader
              title={title}
              icon={icon}
              actions={actions}
              className={headerClassName}
            />
          )
        )}

        <PanelContent
          scrollable={scrollable}
          padding={padding}
          className={contentClassName}
        >
          {children}
        </PanelContent>

        {footer && (
          <PanelFooter className={footerClassName}>
            {footer}
          </PanelFooter>
        )}
      </div>
    );
  }
);

Panel.displayName = 'Panel';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Compound Component Exports                                                         │
// └────────────────────────────────────────────────────────────────────────────────────┘

export default Object.assign(Panel, {
  Header: PanelHeader,
  Content: PanelContent,
  Footer: PanelFooter
});