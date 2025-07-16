/**
 * AppShell Component
 * ---
 * bwr-tools/frontend/src/components/layout/primitives/AppShell.tsx
 * ---
 * Main application shell providing consistent layout structure with header, sidebar, and footer slots
 */

'use client';

import React from 'react';
import { clsx } from 'clsx';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Type Definitions                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

export interface AppShellProps {
  children: React.ReactNode;
  header?: React.ReactNode;
  footer?: React.ReactNode;
  sidebar?: React.ReactNode;
  sidebarPosition?: 'left' | 'right';
  sidebarCollapsed?: boolean;
  className?: string;
  layout?: 'fixed' | 'fluid' | 'centered';
  padding?: boolean | 'sm' | 'md' | 'lg';
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ AppShell Component                                                                 │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function AppShell({
  children,
  header,
  footer,
  sidebar,
  sidebarPosition = 'left',
  sidebarCollapsed = false,
  className,
  layout = 'fixed',
  padding = true,
}: AppShellProps) {
  // Calculate padding classes
  const paddingClass = padding === true ? 'p-4' : 
                      padding === 'sm' ? 'p-2' :
                      padding === 'md' ? 'p-4' :
                      padding === 'lg' ? 'p-6' :
                      '';

  // Layout classes
  const layoutClass = layout === 'fixed' ? 'h-screen overflow-hidden' :
                     layout === 'fluid' ? 'min-h-screen' :
                     'min-h-screen max-w-7xl mx-auto';

  return (
    <div className={clsx(
      'flex flex-col bg-[var(--color-bg-primary)]',
      layoutClass,
      className
    )}>
      {/* Header */}
      {header && (
        <header className="flex-shrink-0 z-10 border-b border-[var(--color-border)]">
          {header}
        </header>
      )}

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar - Left Position */}
        {sidebar && sidebarPosition === 'left' && (
          <aside className={clsx(
            'flex-shrink-0 bg-[var(--color-bg-secondary)] border-r border-[var(--color-border)] transition-all duration-300',
            sidebarCollapsed ? 'w-16' : 'w-64'
          )}>
            <div className="h-full overflow-y-auto">
              {sidebar}
            </div>
          </aside>
        )}

        {/* Main Content */}
        <main className={clsx(
          'flex-1 overflow-y-auto bg-[var(--color-bg-primary)]',
          paddingClass
        )}>
          {children}
        </main>

        {/* Sidebar - Right Position */}
        {sidebar && sidebarPosition === 'right' && (
          <aside className={clsx(
            'flex-shrink-0 bg-[var(--color-bg-secondary)] border-l border-[var(--color-border)] transition-all duration-300',
            sidebarCollapsed ? 'w-16' : 'w-64'
          )}>
            <div className="h-full overflow-y-auto">
              {sidebar}
            </div>
          </aside>
        )}
      </div>

      {/* Footer */}
      {footer && (
        <footer className="flex-shrink-0 z-10 border-t border-[var(--color-border)]">
          {footer}
        </footer>
      )}
    </div>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ AppShell Sub-components                                                            │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function AppShellHeader({ children, className }: { 
  children: React.ReactNode; 
  className?: string;
}) {
  return (
    <div className={clsx(
      'h-16 px-4 flex items-center bg-[var(--color-bg-secondary)]',
      className
    )}>
      {children}
    </div>
  );
}

export function AppShellFooter({ children, className }: { 
  children: React.ReactNode; 
  className?: string;
}) {
  return (
    <div className={clsx(
      'px-4 py-3 bg-[var(--color-bg-secondary)]',
      className
    )}>
      {children}
    </div>
  );
}

export function AppShellSidebar({ children, className }: { 
  children: React.ReactNode; 
  className?: string;
}) {
  return (
    <div className={clsx(
      'p-4',
      className
    )}>
      {children}
    </div>
  );
}

export function AppShellMain({ children, className }: { 
  children: React.ReactNode; 
  className?: string;
}) {
  return (
    <div className={clsx(
      'h-full',
      className
    )}>
      {children}
    </div>
  );
}