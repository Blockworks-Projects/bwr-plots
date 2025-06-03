'use client';

import { QueryClient, QueryClientProvider as RQQueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';
import { queryClient } from '@/lib/queryClient';

export function QueryClientProvider({ children }: { children: React.ReactNode }) {
  const [client] = useState(() => queryClient);

  return (
    <RQQueryClientProvider client={client}>
      {children}
    </RQQueryClientProvider>
  );
} 