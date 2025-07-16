/**
 * Tools Platform Layout
 * ---
 * bwr-tools/frontend/src/app/(tools)/layout.tsx
 * ---
 * Shared layout for all tools with persistent header
 */

import { ToolsClientLayout } from './client-layout';

export default function ToolsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="h-screen flex flex-col bg-[var(--color-bg-primary)]">
      <ToolsClientLayout>
        {children}
      </ToolsClientLayout>
    </div>
  );
}

export const metadata = {
  title: 'BWR Tools',
  description: 'Professional data visualization and analysis platform',
};