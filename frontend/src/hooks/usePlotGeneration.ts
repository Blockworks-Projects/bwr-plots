import { useState, useCallback } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { api, handleApiError } from '@/lib/api';
import { PlotType } from '@/types/plots';

interface PlotGenerationConfig {
  plotType: PlotType;
  sessionId: string;
  configuration: any;
}

interface PlotGenerationResult {
  plotData?: any;
  plotHtml?: string;
  success: boolean;
  error?: string;
  debugInfo?: {
    requestData?: any;
    responseData?: any;
    duration?: number;
    timestamp?: string;
    pythonStderr?: string;
    exitCode?: number;
    networkError?: string;
  };
}

interface UsePlotGenerationReturn {
  generatePlot: (config: PlotGenerationConfig) => Promise<PlotGenerationResult>;
  exportPlot: (format: 'html' | 'png' | 'svg' | 'pdf') => Promise<void>;
  isGenerating: boolean;
  isExporting: boolean;
  error: string | null;
  lastGeneratedPlot: PlotGenerationResult | null;
  clearError: () => void;
  regenerate: () => Promise<void>;
  debugInfo: any[];
}

export function usePlotGeneration(): UsePlotGenerationReturn {
  const [error, setError] = useState<string | null>(null);
  const [lastGeneratedPlot, setLastGeneratedPlot] = useState<PlotGenerationResult | null>(null);
  const [lastConfig, setLastConfig] = useState<PlotGenerationConfig | null>(null);
  const [debugInfo, setDebugInfo] = useState<any[]>([]);
  const queryClient = useQueryClient();

  // Function to add debug information
  const addDebugInfo = (info: any) => {
    setDebugInfo(prev => {
      const newInfo = [...prev, { ...info, timestamp: new Date().toISOString() }];
      // Keep only last 10 debug entries
      return newInfo.slice(-10);
    });
  };

  // Plot generation mutation
  const generateMutation = useMutation({
    mutationFn: async (config: PlotGenerationConfig): Promise<PlotGenerationResult> => {
      const startTime = performance.now();
      
      addDebugInfo({
        type: 'plot_generation_start',
        config,
        sessionId: config.sessionId,
        plotType: config.plotType
      });

      try {
        const plotRequest = {
          session_id: config.sessionId,
          plot_type: config.plotType,
          configuration: {
            ...config.configuration,
            plot_type: config.plotType
          }
        };

        addDebugInfo({
          type: 'request_prepared',
          plotRequest,
          requestSize: JSON.stringify(plotRequest).length
        });

        console.log('[PLOT_GENERATION] Sending request:', plotRequest);

        const response = await api.plots.generate(plotRequest);
        const duration = Math.round(performance.now() - startTime);
        
        addDebugInfo({
          type: 'response_received',
          duration,
          responseKeys: Object.keys(response),
          success: response.success,
          hasError: !!response.error,
          hasPlotData: !!response.plot_json,
          hasPlotHtml: !!response.plot_html
        });

        console.log('[PLOT_GENERATION] Response received:', {
          success: response.success,
          hasPlotData: !!response.plot_json,
          hasPlotHtml: !!response.plot_html,
          error: response.error,
          duration
        });
        
        if (response.success) {
          return {
            plotData: response.plot_json,
            plotHtml: response.plot_html,
            success: true,
            debugInfo: {
              requestData: plotRequest,
              responseData: response,
              duration,
              timestamp: new Date().toISOString()
            }
          };
        } else {
          // Enhanced error handling for different types of errors
          let errorMessage = response.error || 'Plot generation failed';
          
          // Check if this is a Python script error (code 2)
          if (errorMessage.includes('code 2')) {
            errorMessage = `Python script error (exit code 2). This usually indicates:
- Missing required Python packages
- Data format issues
- Column name mismatches
- Invalid plot configuration
- Python environment problems`;
          }

          const exitCodeMatch = response.error?.match(/code (\d+)/);
          const exitCode = exitCodeMatch ? parseInt(exitCodeMatch[1], 10) : undefined;

          addDebugInfo({
            type: 'plot_generation_failed',
            error: response.error,
            duration,
            exitCode
          });

          return {
            success: false,
            error: errorMessage,
            debugInfo: {
              requestData: plotRequest,
              responseData: response,
              duration,
              timestamp: new Date().toISOString(),
              exitCode
            }
          };
        }
      } catch (err) {
        const duration = Math.round(performance.now() - startTime);
        const errorMessage = handleApiError(err);
        
        addDebugInfo({
          type: 'network_error',
          error: errorMessage,
          duration,
          errorType: err instanceof Error ? err.constructor.name : 'Unknown'
        });

        console.error('[PLOT_GENERATION] Network error:', err);

        return {
          success: false,
          error: errorMessage,
          debugInfo: {
            duration,
            timestamp: new Date().toISOString(),
            networkError: errorMessage
          }
        };
      }
    },
    onSuccess: (result) => {
      setLastGeneratedPlot(result);
      setError(result.success ? null : result.error || 'Unknown error');
      
      addDebugInfo({
        type: 'mutation_success',
        success: result.success,
        hasError: !!result.error
      });

      // Invalidate related queries to refresh data
      queryClient.invalidateQueries({ queryKey: ['plots'] });
    },
    onError: (err) => {
      const errorMessage = handleApiError(err);
      setError(errorMessage);
      setLastGeneratedPlot({
        success: false,
        error: errorMessage
      });

      addDebugInfo({
        type: 'mutation_error',
        error: errorMessage
      });

      console.error('[PLOT_GENERATION] Mutation error:', err);
    }
  });

  // Plot export mutation
  const exportMutation = useMutation({
    mutationFn: async ({ format, plotData }: { format: string; plotData: any }) => {
      try {
        const exportRequest = {
          format,
          plot_data: plotData,
          session_id: lastConfig?.sessionId || ''
        };

        addDebugInfo({
          type: 'export_start',
          format,
          hasPlotData: !!plotData,
          sessionId: lastConfig?.sessionId
        });

        const response = await api.plots.export(exportRequest);
        
        if (response.success && response.download_url) {
          // Trigger download
          const link = document.createElement('a');
          link.href = response.download_url;
          link.download = `plot.${format}`;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);

          addDebugInfo({
            type: 'export_success',
            format,
            downloadUrl: response.download_url
          });
        } else {
          throw new Error(response.error || 'Export failed');
        }
      } catch (err) {
        addDebugInfo({
          type: 'export_error',
          error: handleApiError(err)
        });
        throw new Error(handleApiError(err));
      }
    },
    onError: (err) => {
      setError(handleApiError(err));
    }
  });

  // Main generate function
  const generatePlot = useCallback(async (config: PlotGenerationConfig): Promise<PlotGenerationResult> => {
    setLastConfig(config);
    setError(null);
    
    addDebugInfo({
      type: 'generate_plot_called',
      config: {
        plotType: config.plotType,
        sessionId: config.sessionId,
        configurationKeys: Object.keys(config.configuration || {}),
        hasXColumn: !!config.configuration?.x_column,
        hasYColumn: !!config.configuration?.y_column
      }
    });

    // Validate configuration
    if (!config.sessionId) {
      const error = 'Session ID is required';
      setError(error);
      addDebugInfo({ type: 'validation_error', field: 'sessionId' });
      return { success: false, error };
    }

    if (!config.plotType) {
      const error = 'Plot type is required';
      setError(error);
      addDebugInfo({ type: 'validation_error', field: 'plotType' });
      return { success: false, error };
    }

    // Check for required columns based on plot type
    const requiredFields = getRequiredFields(config.plotType);
    for (const field of requiredFields) {
      if (!config.configuration[field]) {
        const error = `${field.replace('_', ' ')} is required for ${config.plotType} plots`;
        setError(error);
        addDebugInfo({ 
          type: 'validation_error', 
          field, 
          plotType: config.plotType,
          availableFields: Object.keys(config.configuration || {})
        });
        return { success: false, error };
      }
    }

    addDebugInfo({
      type: 'validation_passed',
      requiredFields,
      providedFields: Object.keys(config.configuration || {})
    });

    try {
      const result = await generateMutation.mutateAsync(config);
      return result;
    } catch (err) {
      const errorMessage = handleApiError(err);
      setError(errorMessage);
      addDebugInfo({
        type: 'generate_plot_exception',
        error: errorMessage
      });
      return { success: false, error: errorMessage };
    }
  }, [generateMutation]);

  // Export function
  const exportPlot = useCallback(async (format: 'html' | 'png' | 'svg' | 'pdf') => {
    if (!lastGeneratedPlot?.success || !lastGeneratedPlot.plotData) {
      setError('No plot data available for export');
      addDebugInfo({
        type: 'export_validation_error',
        hasLastPlot: !!lastGeneratedPlot,
        plotSuccess: lastGeneratedPlot?.success,
        hasPlotData: !!lastGeneratedPlot?.plotData
      });
      return;
    }

    try {
      await exportMutation.mutateAsync({ 
        format, 
        plotData: lastGeneratedPlot.plotData 
      });
    } catch (err) {
      // Error is handled by the mutation
    }
  }, [lastGeneratedPlot, exportMutation]);

  // Regenerate function
  const regenerate = useCallback(async () => {
    if (!lastConfig) {
      setError('No previous configuration available');
      addDebugInfo({ type: 'regenerate_error', reason: 'no_last_config' });
      return;
    }

    addDebugInfo({
      type: 'regenerate_start',
      lastConfig: {
        plotType: lastConfig.plotType,
        sessionId: lastConfig.sessionId,
        configKeys: Object.keys(lastConfig.configuration || {})
      }
    });

    await generatePlot(lastConfig);
  }, [lastConfig, generatePlot]);

  // Clear error function
  const clearError = useCallback(() => {
    setError(null);
    addDebugInfo({ type: 'error_cleared' });
  }, []);

  return {
    generatePlot,
    exportPlot,
    isGenerating: generateMutation.isPending,
    isExporting: exportMutation.isPending,
    error,
    lastGeneratedPlot,
    clearError,
    regenerate,
    debugInfo
  };
}

// Helper function to get required fields for each plot type
function getRequiredFields(plotType: PlotType): string[] {
  switch (plotType) {
    case 'time_series':
      return ['x_column'];  // y_column is optional - will plot all numeric columns
    case 'scatter':
      return ['x_column'];  // y_column is optional - will plot all numeric columns
    case 'bar':
    case 'horizontal_bar':
      return ['x_column', 'y_column'];
    case 'line':
      return ['x_column'];  // y_column is optional - will plot all numeric columns
    case 'area':
      return ['x_column'];  // y_column is optional - will plot all numeric columns
    case 'histogram':
      return ['x_column'];
    default:
      return ['x_column'];
  }
} 