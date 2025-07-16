/**
 * Live Plot Hook
 * ---
 * bwr-plots/frontend/src/hooks/useLivePlot.ts
 * ---
 * Hook for managing live plot generation with debouncing
 */

import { useCallback, useEffect, useRef } from 'react';
import { useSession } from './useSession';
import { usePlotGeneration } from './usePlotGeneration';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Hook Configuration                                                                 │
// └────────────────────────────────────────────────────────────────────────────────────┘

interface UseLivePlotOptions {
  /** Debounce delay in milliseconds */
  debounceDelay?: number;
  /** Auto-generate on configuration changes */
  autoGenerate?: boolean;
  /** Callback when plot is generated */
  onPlotGenerated?: (plotHtml: string) => void;
  /** Callback when generation fails */
  onError?: (error: any) => void;
}

interface LivePlotState {
  /** Generate plot manually */
  generatePlot: () => Promise<void>;
  /** Whether plot is currently generating */
  isGenerating: boolean;
  /** Last generated plot HTML */
  plotHtml: string | null;
  /** Generation error if any */
  error: any;
  /** Whether conditions are met for generation */
  canGenerate: boolean;
  /** Pending generation (debounced) */
  isPending: boolean;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Live Plot Hook                                                                     │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function useLivePlot({
  debounceDelay = 1500,
  autoGenerate = true,
  onPlotGenerated,
  onError
}: UseLivePlotOptions = {}): LivePlotState {
  
  const { session } = useSession();
  const { generatePlot: apiGeneratePlot, isGenerating, error, lastGeneratedPlot } = usePlotGeneration();
  
  const plotHtmlRef = useRef<string | null>(null);
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const isPendingRef = useRef(false);

  // ┌────────────────────────────────────────────────────────────────────────────────────┐
  // │ Generation Conditions                                                              │
  // └────────────────────────────────────────────────────────────────────────────────────┘

  const canGenerate = Boolean(
    session?.plotType &&
    session?.uploadedData?.data?.length &&
    session?.sessionId &&
    session?.plotConfig?.x_column &&
    session?.plotConfig?.y_column
  );

  // ┌────────────────────────────────────────────────────────────────────────────────────┐
  // │ Plot Generation Function                                                           │
  // └────────────────────────────────────────────────────────────────────────────────────┘

  const generatePlot = useCallback(async () => {
    if (!canGenerate) {
      console.warn('Cannot generate plot: missing required data');
      return;
    }

    try {
      const result = await apiGeneratePlot({
        plotType: session!.plotType as any,
        sessionId: session!.sessionId,
        configuration: {
          ...session!.plotConfig,
          x_column: session!.plotConfig?.x_column,
          y_column: session!.plotConfig?.y_column,
          title: session!.plotConfig?.title || '',
          subtitle: session!.plotConfig?.subtitle || '',
          source: session!.plotConfig?.source || ''
        }
      });
      
      if (result.success && result.plotHtml) {
        plotHtmlRef.current = result.plotHtml;
        onPlotGenerated?.(result.plotHtml);
      }
    } catch (err) {
      console.error('Plot generation error:', err);
      onError?.(err);
    }
  }, [session, apiGeneratePlot, canGenerate, onPlotGenerated, onError]);

  // ┌────────────────────────────────────────────────────────────────────────────────────┐
  // │ Debounced Auto-Generation                                                          │
  // └────────────────────────────────────────────────────────────────────────────────────┘

  useEffect(() => {
    if (!autoGenerate || !canGenerate) {
      isPendingRef.current = false;
      return;
    }

    // Clear existing timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    // Set pending state
    isPendingRef.current = true;

    // Set up debounced generation
    debounceTimerRef.current = setTimeout(() => {
      isPendingRef.current = false;
      generatePlot();
    }, debounceDelay);

    // Cleanup
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
        isPendingRef.current = false;
      }
    };
  }, [
    session?.plotType,
    session?.plotConfig,
    autoGenerate,
    canGenerate,
    debounceDelay,
    generatePlot
  ]);

  // ┌────────────────────────────────────────────────────────────────────────────────────┐
  // │ Cleanup on Unmount                                                                │
  // └────────────────────────────────────────────────────────────────────────────────────┘

  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);

  // ┌────────────────────────────────────────────────────────────────────────────────────┐
  // │ Return State                                                                       │
  // └────────────────────────────────────────────────────────────────────────────────────┘

  return {
    generatePlot,
    isGenerating,
    plotHtml: plotHtmlRef.current,
    error,
    canGenerate,
    isPending: isPendingRef.current
  };
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Plot Suggestions Hook                                                             │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function usePlotSuggestions() {
  const { session, updateSession } = useSession();

  const generateSuggestions = useCallback(() => {
    if (!session?.uploadedData?.columns || !session?.uploadedData?.dataTypes) {
      return [];
    }

    const { columns, dataTypes } = session.uploadedData;
    const suggestions = [];

    // Count column types
    const numericColumns = columns.filter(col => 
      ['number', 'numeric', 'integer', 'float'].includes(dataTypes[col]?.toLowerCase())
    );
    const dateColumns = columns.filter(col => 
      ['date', 'datetime', 'timestamp'].includes(dataTypes[col]?.toLowerCase())
    );
    const categoricalColumns = columns.filter(col => 
      ['string', 'text', 'object'].includes(dataTypes[col]?.toLowerCase())
    );

    // Suggest plot types based on data structure
    if (dateColumns.length > 0 && numericColumns.length > 0) {
      suggestions.push({
        type: 'time_series',
        reason: 'Date/time data detected with numeric values',
        confidence: 0.9,
        suggestedConfig: {
          x_column: dateColumns[0],
          y_column: numericColumns[0]
        }
      });
    }

    if (categoricalColumns.length > 0 && numericColumns.length > 0) {
      suggestions.push({
        type: 'bar',
        reason: 'Categorical data with numeric values',
        confidence: 0.8,
        suggestedConfig: {
          x_column: categoricalColumns[0],
          y_column: numericColumns[0]
        }
      });
    }

    if (numericColumns.length >= 2) {
      suggestions.push({
        type: 'scatter',
        reason: 'Multiple numeric columns for correlation analysis',
        confidence: 0.7,
        suggestedConfig: {
          x_column: numericColumns[0],
          y_column: numericColumns[1]
        }
      });
    }

    return suggestions.sort((a, b) => b.confidence - a.confidence);
  }, [session]);

  const applySuggestion = useCallback((suggestion: any) => {
    updateSession({
      plotType: suggestion.type,
      plotConfig: {
        ...session?.plotConfig,
        ...suggestion.suggestedConfig
      }
    });
  }, [session, updateSession]);

  return {
    suggestions: generateSuggestions(),
    applySuggestion
  };
}