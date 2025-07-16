/**
 * Tools Platform Client Layout
 * ---
 * bwr-tools/frontend/src/app/(tools)/client-layout.tsx
 * ---
 * Client-side wrapper for tools layout to ensure proper provider context
 */

'use client';

import { PlatformHeader } from '@/components/platform/PlatformHeader';

export function ToolsClientLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <>
      {/* Persistent Platform Header */}
      <PlatformHeader />
      
      {/* Tool Content Area with Independent Scrolling */}
      <main className="flex-1 min-h-0">
        {children}
      </main>
    </>
  );
}