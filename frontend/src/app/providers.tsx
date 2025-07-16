'use client';

import { QueryClient, QueryClientProvider as RQQueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';
import { queryClient } from '@/lib/queryClient';
import { SessionProvider } from '@/contexts/SessionContext';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { ViewportProvider } from '@/components/layout/primitives/ViewportProvider';

export function QueryClientProvider({ children }: { children: React.ReactNode }) {
  const [client] = useState(() => queryClient);

  return (
    <RQQueryClientProvider client={client}>
      <ThemeProvider>
        <ViewportProvider>
          <SessionProvider>
            {children}
          </SessionProvider>
        </ViewportProvider>
      </ThemeProvider>
    </RQQueryClientProvider>
  );
} 