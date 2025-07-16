'use client';

import React, { useState } from 'react';

interface DebugInfo {
  plotType?: string;
  dataLength?: number;
  columns?: string[];
  plotHtml?: string;
  error?: string;
  config?: any;
  isGenerating?: boolean;
  lastGeneratedPlot?: any;
  bwrDebugInfo?: any[];
}

interface PlotDebugPanelProps {
  debugInfo: DebugInfo;
}

export function PlotDebugPanel({ debugInfo }: PlotDebugPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  return (
    <div className="fixed bottom-20 right-4 z-50 max-w-md">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="bg-gray-800 text-white px-4 py-2 rounded-lg shadow-lg hover:bg-gray-700 transition-colors text-sm"
      >
        {isExpanded ? 'Hide' : 'Show'} Plot Debug
      </button>

      {isExpanded && (
        <div className="absolute bottom-12 right-0 bg-gray-800 border border-gray-600 rounded-lg shadow-xl p-4 w-96 max-h-96 overflow-auto">
          <h3 className="font-semibold text-gray-100 mb-2">Plot Debug Info</h3>
          
          <div className="space-y-2 text-sm">
            <div>
              <span className="font-medium text-gray-300">Plot Type:</span>{' '}
              <span className="font-mono text-gray-100">{debugInfo.plotType || 'None'}</span>
            </div>
            
            <div>
              <span className="font-medium text-gray-300">Data Points:</span>{' '}
              <span className="font-mono text-gray-100">{debugInfo.dataLength || 0}</span>
            </div>
            
            <div>
              <span className="font-medium text-gray-300">Columns:</span>{' '}
              <span className="font-mono text-gray-100">
                {debugInfo.columns ? debugInfo.columns.join(', ') : 'None'}
              </span>
            </div>
            
            <div>
              <span className="font-medium text-gray-300">BWR Generation:</span>{' '}
              <span className={`font-mono ${debugInfo.isGenerating ? 'text-yellow-300' : 'text-gray-100'}`}>
                {debugInfo.isGenerating ? 'In Progress...' : 'Idle'}
              </span>
            </div>
            
            <div>
              <span className="font-medium text-gray-300">HTML Length:</span>{' '}
              <span className="font-mono text-gray-100">
                {debugInfo.plotHtml ? debugInfo.plotHtml.length : 0} chars
              </span>
            </div>
            
            {debugInfo.error && (
              <div>
                <span className="font-medium text-red-400">Error:</span>{' '}
                <span className="text-red-300">{debugInfo.error}</span>
              </div>
            )}
            
            {debugInfo.lastGeneratedPlot && (
              <div>
                <span className="font-medium text-gray-300">Last BWR Result:</span>
                <pre className="mt-1 p-2 bg-gray-900 rounded text-xs overflow-auto text-gray-100">
                  {JSON.stringify({
                    success: debugInfo.lastGeneratedPlot.success,
                    hasPlotData: !!debugInfo.lastGeneratedPlot.plotData,
                    hasPlotHtml: !!debugInfo.lastGeneratedPlot.plotHtml,
                    error: debugInfo.lastGeneratedPlot.error
                  }, null, 2)}
                </pre>
              </div>
            )}
            
            <div>
              <span className="font-medium text-gray-300">Config:</span>
              <pre className="mt-1 p-2 bg-gray-900 rounded text-xs overflow-auto text-gray-100">
                {JSON.stringify(debugInfo.config || {}, null, 2)}
              </pre>
            </div>

            {debugInfo.bwrDebugInfo && debugInfo.bwrDebugInfo.length > 0 && (
              <div>
                <span className="font-medium text-gray-300">BWR Debug Log (last 3):</span>
                <div className="mt-1 space-y-1">
                  {debugInfo.bwrDebugInfo.slice(-3).map((entry, index) => (
                    <div key={index} className="p-1 bg-gray-900 rounded text-xs text-gray-100">
                      <span className="text-zinc-300">{entry.type}:</span> {JSON.stringify(entry)}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {debugInfo.plotHtml && (
              <div>
                <span className="font-medium text-gray-300">HTML Preview (first 500 chars):</span>
                <pre className="mt-1 p-2 bg-gray-900 rounded text-xs overflow-auto text-gray-100">
                  {debugInfo.plotHtml.substring(0, 500)}...
                </pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}