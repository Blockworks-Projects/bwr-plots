import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

interface DataOperation {
  type: 'drop_columns' | 'rename_columns' | 'pivot_data';
  columns?: string[];
  column_mapping?: Record<string, string>;
  pivot_config?: {
    index_column: string;
    value_column: string;
    pivot_column: string;
  };
}

interface ProcessingResponse {
  success: boolean;
  preview_data: Record<string, any>[];
  row_count: number;
  column_count: number;
  columns: string[];
}

export function useDataProcessing(sessionId: string) {
  const queryClient = useQueryClient();

  // Get current data preview
  const { data: previewData, isLoading, error } = useQuery({
    queryKey: ['dataPreview', sessionId],
    queryFn: async () => {
      console.log(`[useDataProcessing] Fetching preview for session: ${sessionId}`);
      
      if (!sessionId) {
        throw new Error('Session ID is required');
      }
      
      const response = await fetch(`/api/data/preview/${sessionId}`);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMsg = errorData.error || `HTTP ${response.status}: ${response.statusText}`;
        console.error(`[useDataProcessing] Preview fetch failed:`, errorMsg);
        throw new Error(errorMsg);
      }
      
      const result = await response.json();
      console.log(`[useDataProcessing] Preview data received:`, {
        rows: result.row_count || 0,
        columns: result.column_count || 0,
        sessionId
      });
      
      return result;
    },
    enabled: !!sessionId,
    retry: (failureCount, error) => {
      console.log(`[useDataProcessing] Query retry ${failureCount}, error:`, error?.message);
      return failureCount < 2;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
  });

  const processingMutation = useMutation<ProcessingResponse, Error, DataOperation[]>({
    mutationFn: async (operations: DataOperation[]) => {
      console.log(`[useDataProcessing] Starting data manipulation for session: ${sessionId}`);
      console.log(`[useDataProcessing] Operations:`, operations);
      
      const response = await fetch('/api/data/manipulate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          operations,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMsg = errorData.error || 'Data manipulation failed';
        console.error(`[useDataProcessing] Manipulation failed:`, errorMsg);
        throw new Error(errorMsg);
      }

      const result = await response.json();
      console.log(`[useDataProcessing] Manipulation completed successfully`);
      return result;
    },
    onSuccess: () => {
      console.log(`[useDataProcessing] Invalidating preview cache for session: ${sessionId}`);
      // Invalidate and refetch the preview data
      queryClient.invalidateQueries({ queryKey: ['dataPreview', sessionId] });
    },
    onError: (error) => {
      console.error(`[useDataProcessing] Mutation error:`, error);
    },
  });

  const dropColumns = (columns: string[]) => {
    console.log(`[useDataProcessing] Dropping columns:`, columns);
    const operation: DataOperation = {
      type: 'drop_columns',
      columns,
    };
    return processingMutation.mutate([operation]);
  };

  const renameColumns = (columnMapping: Record<string, string>) => {
    console.log(`[useDataProcessing] Renaming columns:`, columnMapping);
    const operation: DataOperation = {
      type: 'rename_columns',
      column_mapping: columnMapping,
    };
    return processingMutation.mutate([operation]);
  };

  const pivotData = (config: {
    index_column: string;
    value_column: string;
    pivot_column: string;
  }) => {
    console.log(`[useDataProcessing] Pivoting data:`, config);
    const operation: DataOperation = {
      type: 'pivot_data',
      pivot_config: config,
    };
    return processingMutation.mutate([operation]);
  };

  const applyOperations = (operations: DataOperation[]) => {
    console.log(`[useDataProcessing] Applying ${operations.length} operations`);
    return processingMutation.mutate(operations);
  };

  return {
    // Data
    previewData,
    isLoadingPreview: isLoading,
    previewError: error,
    
    // Processing
    dropColumns,
    renameColumns,
    pivotData,
    applyOperations,
    
    // Status
    isProcessing: processingMutation.isPending,
    processingError: processingMutation.error,
    processingSuccess: processingMutation.isSuccess,
    
    // Control
    reset: processingMutation.reset,
  };
} 