/**
 * Data Panel
 * ---
 * bwr-plots/frontend/src/components/dashboard/DataPanel.tsx
 * ---
 * Left sidebar component for data upload, preview, and manipulation
 */

'use client';

import React, { useState } from 'react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { FileUpload } from '@/components/data/FileUpload';
import { useSession } from '@/hooks/useSession';
import { useDataUpload } from '@/hooks/useDataUpload';
import { getAllDummyDatasets } from '@/lib/dummyData';

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Type Definitions                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

interface DataPanelProps {
  className?: string;
}

interface DataStats {
  rows: number;
  columns: number;
  fileName?: string;
  fileSize?: string;
  dataTypes: Record<string, string>;
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Data Statistics Component                                                          │
// └────────────────────────────────────────────────────────────────────────────────────┘

function DataStatistics({ stats }: { stats: DataStats }) {
  return (
    <Card className="mb-4">
      <div className="p-4">
        <h3 className="text-sm font-semibold text-[var(--color-text-primary)] mb-3">
          Data Overview
        </h3>
        
        <div className="space-y-2 text-sm">
          {stats.fileName && (
            <div className="flex justify-between">
              <span className="text-[var(--color-text-muted)]">File:</span>
              <span className="text-[var(--color-text-secondary)] truncate ml-2">
                {stats.fileName}
              </span>
            </div>
          )}
          
          <div className="flex justify-between">
            <span className="text-[var(--color-text-muted)]">Rows:</span>
            <span className="text-[var(--color-text-secondary)]">
              {stats.rows.toLocaleString()}
            </span>
          </div>
          
          <div className="flex justify-between">
            <span className="text-[var(--color-text-muted)]">Columns:</span>
            <span className="text-[var(--color-text-secondary)]">
              {stats.columns}
            </span>
          </div>
          
          {stats.fileSize && (
            <div className="flex justify-between">
              <span className="text-[var(--color-text-muted)]">Size:</span>
              <span className="text-[var(--color-text-secondary)]">
                {stats.fileSize}
              </span>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Column Types Component                                                             │
// └────────────────────────────────────────────────────────────────────────────────────┘

function ColumnTypes({ dataTypes }: { dataTypes: Record<string, string> }) {
  const getTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'number':
      case 'numeric':
      case 'integer':
      case 'float':
        return '🔢';
      case 'string':
      case 'text':
      case 'object':
        return '📝';
      case 'date':
      case 'datetime':
      case 'timestamp':
        return '📅';
      case 'boolean':
        return '✅';
      default:
        return '❓';
    }
  };

  const getTypeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case 'number':
      case 'numeric':
      case 'integer':
      case 'float':
        return 'text-[var(--color-info)]';
      case 'string':
      case 'text':
      case 'object':
        return 'text-[var(--color-success)]';
      case 'date':
      case 'datetime':
      case 'timestamp':
        return 'text-[var(--color-primary)]';
      case 'boolean':
        return 'text-[var(--color-warning)]';
      default:
        return 'text-[var(--color-text-muted)]';
    }
  };

  return (
    <Card className="mb-4">
      <div className="p-4">
        <h3 className="text-sm font-semibold text-[var(--color-text-primary)] mb-3">
          Column Types
        </h3>
        
        <div className="space-y-2 max-h-40 overflow-auto">
          {Object.entries(dataTypes).map(([column, type]) => (
            <div key={column} className="flex items-center justify-between text-xs">
              <div className="flex items-center space-x-2 flex-1 min-w-0">
                <span className="text-sm">{getTypeIcon(type)}</span>
                <span className="text-[var(--color-text-secondary)] truncate">
                  {column}
                </span>
              </div>
              <span className={`ml-2 px-2 py-1 rounded text-xs bg-[var(--color-bg-elevated)] ${getTypeColor(type)}`}>
                {type}
              </span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Sample Data Selector                                                               │
// └────────────────────────────────────────────────────────────────────────────────────┘

function SampleDataSelector() {
  const [showSampleData, setShowSampleData] = useState(false);
  const sampleDatasets = getAllDummyDatasets();
  const { uploadDummyData } = useDataUpload();
  
  return (
    <Card className="mb-4">
      <div className="p-4">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowSampleData(!showSampleData)}
          className="w-full"
        >
          {showSampleData ? 'Hide' : 'Show'} Sample Datasets
        </Button>
        
        {showSampleData && (
          <div className="mt-3 space-y-2">
            {sampleDatasets.map((dataset) => (
              <Button
                key={dataset.id}
                variant="ghost"
                size="sm"
                className="w-full justify-start text-left h-auto p-2"
                onClick={() => {
                  uploadDummyData(dataset);
                  setShowSampleData(false);
                }}
              >
                <div>
                  <div className="text-sm font-medium text-[var(--color-text-primary)]">
                    {dataset.name}
                  </div>
                  <div className="text-xs text-[var(--color-text-muted)]">
                    {dataset.description}
                  </div>
                </div>
              </Button>
            ))}
          </div>
        )}
      </div>
    </Card>
  );
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Data Panel Component                                                               │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function DataPanel({ className = '' }: DataPanelProps) {
  const { session } = useSession();
  const [showDataPreview, setShowDataPreview] = useState(false);

  // Calculate data statistics
  const dataStats: DataStats | null = session?.uploadedData ? {
    rows: session.uploadedData.rowCount,
    columns: session.uploadedData.columns.length,
    fileName: session.uploadedData.originalFileName,
    dataTypes: session.uploadedData.dataTypes
  } : null;

  return (
    <div className={`h-full overflow-y-auto p-4 space-y-4 ${className}`}>
      {/* Upload Section */}
      <div>
        <h3 className="text-sm font-semibold text-[var(--color-text-primary)] mb-3">
          Upload Data
        </h3>
        <FileUpload />
      </div>

      {/* Sample Datasets */}
      <SampleDataSelector />

      {/* Data Statistics */}
      {dataStats && (
        <>
          <DataStatistics stats={dataStats} />
          <ColumnTypes dataTypes={dataStats.dataTypes} />
        </>
      )}

      {/* Data Preview Toggle */}
      {session?.uploadedData && (
        <Card>
          <div className="p-4">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowDataPreview(!showDataPreview)}
              className="w-full"
            >
              {showDataPreview ? 'Hide' : 'Show'} Data Preview
            </Button>
            
            {showDataPreview && (
              <div className="mt-3 max-h-60 overflow-auto">
                <div className="text-xs">
                  <div className="grid grid-cols-1 gap-1">
                    {/* Header Row */}
                    <div className="font-semibold text-[var(--color-text-primary)] border-b border-[var(--color-border)] pb-1">
                      {session.uploadedData.columns.join(' | ')}
                    </div>
                    {/* Data Rows (first 5) */}
                    {session.uploadedData.data.slice(0, 5).map((row, index) => (
                      <div key={index} className="text-[var(--color-text-muted)] py-1">
                        {session.uploadedData!.columns.map(col => String(row[col] || '')).join(' | ')}
                      </div>
                    ))}
                    {session.uploadedData.data.length > 5 && (
                      <div className="text-[var(--color-text-muted)] italic">
                        ... and {session.uploadedData.data.length - 5} more rows
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </Card>
      )}

      {/* Data Manipulation Tools */}
      {session?.uploadedData && (
        <Card>
          <div className="p-4">
            <h3 className="text-sm font-semibold text-[var(--color-text-primary)] mb-3">
              Data Tools
            </h3>
            <div className="space-y-2">
              <Button variant="outline" size="sm" className="w-full">
                🔧 Transform Data
              </Button>
              <Button variant="outline" size="sm" className="w-full">
                🧹 Clean Data
              </Button>
              <Button variant="outline" size="sm" className="w-full">
                📊 Calculate Stats
              </Button>
            </div>
          </div>
        </Card>
      )}

      {/* Empty State */}
      {!session?.uploadedData && (
        <div className="text-center py-8">
          <div className="text-4xl mb-4">📁</div>
          <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-2">
            No Data Loaded
          </h3>
          <p className="text-sm text-[var(--color-text-muted)]">
            Upload a CSV or Excel file, or choose a sample dataset to get started.
          </p>
        </div>
      )}
    </div>
  );
}