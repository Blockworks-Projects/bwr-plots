/**
 * BWR Tools Landing Page
 * ---
 * bwr-tools/frontend/src/app/page.tsx
 * ---
 * Root landing page
 */

'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    // Redirect to the plots tool
    router.push('/plots');
  }, [router]);

  // Show loading state while redirecting
  return (
    <div className="h-screen flex items-center justify-center bg-[var(--color-bg-primary)]">
      <div className="text-center">
        <div className="w-16 h-16 bg-[var(--color-primary)] rounded-lg flex items-center justify-center mx-auto mb-4">
          <span className="text-white font-bold text-2xl">BW</span>
        </div>
        <h1 className="text-xl font-semibold text-[var(--color-text-primary)]">BWR Tools</h1>
        <p className="text-sm text-[var(--color-text-muted)] mt-2">Loading...</p>
      </div>
    </div>
  );
}