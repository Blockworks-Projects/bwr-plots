npm# BWR Plots Frontend Refactor Plan

## Executive Summary

This document outlines the comprehensive migration strategy from the current Streamlit-based BWR Plots application to a modern Next.js frontend deployed on Vercel. The goal is to preserve all existing Python plotting functionality while providing a superior user experience and improved scalability using Vercel's serverless platform.

## Current State Analysis

### Existing Architecture
- **Frontend**: Streamlit app (`app.py` - 1,275 lines)
- **Backend**: Embedded Python logic with BWR Plots library
- **Core Library**: `src/bwr_plots/` - Robust plotting functionality
- **Data Flow**: Monolithic Streamlit app handling UI, data processing, and visualization

### Current Features Inventory
1. **File Upload**: CSV/XLSX support with validation
2. **Data Manipulation**: 
   - Column dropping
   - Column renaming
   - Data pivoting
3. **Plot Types**: 7 different visualization types
4. **Data Processing**:
   - Date parsing and handling
   - Filtering (lookback/date window)
   - Resampling
   - Smoothing
5. **Plot Configuration**:
   - Titles, subtitles, sources
   - Axis customization
   - Watermark selection
   - Prefix/suffix formatting
6. **Export**: HTML plot generation

### Current Limitations
- Single-user application
- Limited UI customization
- No real-time collaboration
- Streamlit-specific constraints
- Monolithic architecture

## Target Architecture

### High-Level Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Next.js       │    │   Vercel API    │    │   BWR Plots     │
│   Frontend      │◄──►│   Routes        │◄──►│   Library       │
│   (Vercel)      │    │   (Serverless)  │    │   (Python)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack
- **Frontend**: Next.js 15.3.3 + TypeScript + Tailwind CSS (Vercel)
- **Backend**: Vercel API Routes (Serverless Functions)
- **Plotting**: Existing BWR Plots library (unchanged)
- **State Management**: React Query/TanStack Query
- **File Storage**: Vercel `/tmp` directory (temporary)
- **Session Management**: In-memory with Redis fallback (Vercel KV)

### Vercel-Specific Considerations
- **Function Limits**: 10-second timeout (Hobby), 60-second (Pro)
- **Payload Limits**: 4.5MB request body, 10MB function payload
- **Cold Starts**: Serverless function initialization delays
- **Edge Runtime**: Available for faster cold starts
- **File Storage**: Temporary `/tmp` directory only

## Detailed Implementation Plan

## Phase 1: Next.js Setup & API Routes (Weeks 1-3)

### 1.1 Project Structure Setup
```
frontend/
├── app/
│   ├── layout.tsx             # Root layout
│   ├── page.tsx               # Main plotting interface
│   ├── globals.css            # Global styles
│   └── api/                   # Vercel API routes
│       ├── data/
│       │   ├── upload/
│       │   │   └── route.ts   # File upload endpoint
│       │   ├── preview/
│       │   │   └── [sessionId]/
│       │   │       └── route.ts # Data preview endpoint
│       │   └── manipulate/
│       │       └── route.ts   # Data manipulation endpoint
│       ├── plots/
│       │   ├── generate/
│       │   │   └── route.ts   # Plot generation endpoint
│       │   ├── export/
│       │   │   └── route.ts   # Plot export endpoint
│       │   └── types/
│       │       └── route.ts   # Available plot types
│       ├── config/
│       │   ├── watermarks/
│       │   │   └── route.ts   # Watermark configuration
│       │   └── plot-defaults/
│       │       └── [plotType]/
│       │           └── route.ts # Plot defaults
│       └── health/
│           └── route.ts       # Health check
├── components/
│   ├── ui/                    # Reusable UI components
│   ├── layout/                # Layout components
│   ├── data/                  # Data management components
│   ├── plotting/              # Plot-related components
│   └── forms/                 # Form components
├── hooks/                     # Custom React hooks
├── lib/                       # Utility libraries
├── types/                     # TypeScript definitions
├── styles/                    # Additional styles
├── utils/                     # Python utilities (for API routes)
│   ├── data_processor.py      # Data manipulation logic
│   ├── plot_generator.py      # Plot generation service
│   ├── file_handler.py        # File upload/validation
│   └── session_manager.py     # Session state management
├── next.config.js             # Next.js configuration
├── vercel.json                # Vercel deployment config
├── requirements.txt           # Python dependencies for API routes
└── package.json               # Node.js dependencies
```

### 1.2 Vercel Configuration

#### vercel.json
```json
{
  "functions": {
    "app/api/**/*.ts": {
      "runtime": "nodejs18.x",
      "maxDuration": 30
    }
  },
  "build": {
    "env": {
      "PYTHON_VERSION": "3.10"
    }
  }
}
```

#### next.config.js
```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverComponentsExternalPackages: ['pandas', 'numpy', 'plotly']
  },
  api: {
    bodyParser: {
      sizeLimit: '4.5mb',
    },
  }
}

module.exports = nextConfig
```

### 1.3 API Routes Implementation

#### Data Upload Route (`app/api/data/upload/route.ts`)
```typescript
import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import { writeFile } from 'fs/promises';
import { join } from 'path';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }
    
    // Validate file size (4.5MB limit)
    if (file.size > 4.5 * 1024 * 1024) {
      return NextResponse.json({ error: 'File too large' }, { status: 413 });
    }
    
    // Save to temporary directory
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const tempPath = join('/tmp', `upload_${Date.now()}_${file.name}`);
    await writeFile(tempPath, buffer);
    
    // Process with Python
    const result = await processPythonScript('load_data', {
      file_path: tempPath,
      file_name: file.name
    });
    
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json({ error: 'Upload failed' }, { status: 500 });
  }
}

async function processPythonScript(script: string, args: any): Promise<any> {
  return new Promise((resolve, reject) => {
    const python = spawn('python3', ['-c', `
import sys
import json
sys.path.append('/var/task/utils')
from ${script} import main
result = main(${JSON.stringify(args)})
print(json.dumps(result))
    `]);
    
    let output = '';
    python.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    python.on('close', (code) => {
      if (code === 0) {
        try {
          resolve(JSON.parse(output));
        } catch (e) {
          reject(new Error('Invalid JSON response'));
        }
      } else {
        reject(new Error(`Python script failed with code ${code}`));
      }
    });
  });
}
```

#### Plot Generation Route (`app/api/plots/generate/route.ts`)
```typescript
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Validate request
    if (!body.session_id || !body.plot_type || !body.configuration) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }
    
    // Process with Python plot generator
    const result = await processPythonScript('plot_generator', {
      session_id: body.session_id,
      plot_type: body.plot_type,
      configuration: body.configuration,
      data_processing: body.data_processing
    });
    
    return NextResponse.json(result);
  } catch (error) {
    console.error('Plot generation error:', error);
    return NextResponse.json({ error: 'Plot generation failed' }, { status: 500 });
  }
}

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '4.5mb',
    },
  },
  maxDuration: 30, // 30 seconds for plot generation
}
```

### 1.4 Python Utilities for API Routes

#### utils/data_processor.py
```python
import pandas as pd
import json
import os
from typing import Dict, Any, List

def main(args: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point for data processing operations"""
    operation = args.get('operation', 'load')
    
    if operation == 'load':
        return load_data(args)
    elif operation == 'manipulate':
        return manipulate_data(args)
    else:
        raise ValueError(f"Unknown operation: {operation}")

def load_data(args: Dict[str, Any]) -> Dict[str, Any]:
    """Load and analyze data file"""
    file_path = args['file_path']
    file_name = args['file_name']
    
    try:
        # Load data based on file extension
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Generate session ID
        session_id = f"session_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Save session data
        session_data = {
            'original_data': df.to_json(orient='records'),
            'current_data': df.to_json(orient='records'),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict()
        }
        
        # Store in temporary session storage
        session_path = f"/tmp/session_{session_id}.json"
        with open(session_path, 'w') as f:
            json.dump(session_data, f)
        
        # Generate preview
        preview_data = df.head(100).to_dict('records')
        
        return {
            'session_id': session_id,
            'columns': [{'name': col, 'type': str(df[col].dtype)} for col in df.columns],
            'preview_data': preview_data,
            'row_count': len(df),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
    except Exception as e:
        return {'error': str(e)}
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
```

#### utils/plot_generator.py
```python
import sys
import os
import json
from typing import Dict, Any

# Add BWR plots to path
sys.path.append('/var/task/src')
from bwr_plots import BWRPlots

def main(args: Dict[str, Any]) -> Dict[str, Any]:
    """Generate plot using BWR Plots library"""
    try:
        session_id = args['session_id']
        plot_type = args['plot_type']
        configuration = args['configuration']
        data_processing = args.get('data_processing', {})
        
        # Load session data
        session_path = f"/tmp/session_{session_id}.json"
        if not os.path.exists(session_path):
            return {'error': 'Session not found'}
        
        with open(session_path, 'r') as f:
            session_data = json.load(f)
        
        # Convert back to DataFrame
        df = pd.read_json(session_data['current_data'], orient='records')
        
        # Apply data processing
        processed_df = apply_data_processing(df, data_processing)
        
        # Generate plot using BWR Plots
        bwr_plots = BWRPlots()
        plot_config = convert_config_to_bwr_format(configuration)
        
        # Generate plot based on type
        if plot_type == 'line':
            fig = bwr_plots.line_plot(processed_df, **plot_config)
        elif plot_type == 'bar':
            fig = bwr_plots.bar_plot(processed_df, **plot_config)
        # ... other plot types
        
        # Convert to JSON for frontend
        plot_json = fig.to_json()
        
        # Generate HTML for export
        plot_html = fig.to_html(include_plotlyjs='cdn')
        
        return {
            'success': True,
            'plot_json': json.loads(plot_json),
            'plot_html': plot_html
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def apply_data_processing(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Apply filtering, resampling, and smoothing"""
    # Extract processing logic from original app.py
    # ... implementation details
    return df

def convert_config_to_bwr_format(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert frontend config to BWR Plots format"""
    # ... conversion logic
    return config
```

## Phase 2: Frontend Development (Weeks 4-7)

### 2.1 Component Architecture

#### Main Application Flow
```typescript
// app/page.tsx
'use client';

import { useState } from 'react';
import { FileUpload } from '@/components/data/FileUpload';
import { DataPreview } from '@/components/data/DataPreview';
import { DataManipulation } from '@/components/data/DataManipulation';
import { PlotConfiguration } from '@/components/plotting/PlotConfiguration';
import { PlotDisplay } from '@/components/plotting/PlotDisplay';

export default function Home() {
  const [currentStep, setCurrentStep] = useState(1);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [plotData, setPlotData] = useState(null);

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          {/* Step indicator */}
          <div className="mb-8">
            <StepIndicator currentStep={currentStep} />
          </div>

          {/* Step content */}
          {currentStep === 1 && (
            <FileUpload
              onUploadSuccess={(sessionId) => {
                setSessionId(sessionId);
                setCurrentStep(2);
              }}
            />
          )}

          {currentStep === 2 && sessionId && (
            <DataPreview
              sessionId={sessionId}
              onNext={() => setCurrentStep(3)}
            />
          )}

          {currentStep === 3 && sessionId && (
            <DataManipulation
              sessionId={sessionId}
              onNext={() => setCurrentStep(4)}
            />
          )}

          {currentStep === 4 && sessionId && (
            <PlotConfiguration
              sessionId={sessionId}
              onPlotGenerated={(data) => {
                setPlotData(data);
                setCurrentStep(5);
              }}
            />
          )}

          {currentStep === 5 && plotData && (
            <PlotDisplay
              plotData={plotData}
              onExport={(format) => handleExport(format)}
            />
          )}
        </div>
      </div>
    </div>
  );
}
```

#### File Upload Component
```typescript
// components/data/FileUpload.tsx
'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { useMutation } from '@tanstack/react-query';

interface FileUploadProps {
  onUploadSuccess: (sessionId: string) => void;
}

export function FileUpload({ onUploadSuccess }: FileUploadProps) {
  const [uploadProgress, setUploadProgress] = useState(0);

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/data/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      return response.json();
    },
    onSuccess: (data) => {
      if (data.session_id) {
        onUploadSuccess(data.session_id);
      }
    },
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      // Validate file size (4.5MB limit for Vercel)
      if (file.size > 4.5 * 1024 * 1024) {
        alert('File size must be less than 4.5MB');
        return;
      }
      uploadMutation.mutate(file);
    }
  }, [uploadMutation]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls']
    },
    maxFiles: 1,
    maxSize: 4.5 * 1024 * 1024 // 4.5MB limit
  });

  return (
    <div className="max-w-2xl mx-auto">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
          isDragActive
            ? 'border-blue-400 bg-blue-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
      >
        <input {...getInputProps()} />
        
        {uploadMutation.isLoading ? (
          <div className="space-y-4">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <p className="text-gray-600">Uploading and processing file...</p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="mx-auto h-12 w-12 text-gray-400">
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <div>
              <p className="text-lg font-medium text-gray-900">
                {isDragActive ? 'Drop your file here' : 'Upload your data file'}
              </p>
              <p className="text-gray-500 mt-2">
                Drag and drop a CSV or Excel file, or click to browse
              </p>
              <p className="text-sm text-gray-400 mt-1">
                Maximum file size: 4.5MB
              </p>
            </div>
          </div>
        )}
      </div>

      {uploadMutation.error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-md">
          <p className="text-red-800">
            Upload failed: {uploadMutation.error.message}
          </p>
        </div>
      )}
    </div>
  );
}
```

### 2.2 API Client Setup

#### lib/api.ts
```typescript
import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
      retry: (failureCount, error: any) => {
        // Don't retry on 4xx errors
        if (error?.status >= 400 && error?.status < 500) {
          return false;
        }
        return failureCount < 2;
      },
    },
    mutations: {
      retry: 1,
    },
  },
});

// API client functions
export const api = {
  data: {
    upload: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('/api/data/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }
      
      return response.json();
    },
    
    preview: async (sessionId: string) => {
      const response = await fetch(`/api/data/preview/${sessionId}`);
      
      if (!response.ok) {
        throw new Error(`Preview failed: ${response.statusText}`);
      }
      
      return response.json();
    },
    
    manipulate: async (sessionId: string, operations: any[]) => {
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
        throw new Error(`Manipulation failed: ${response.statusText}`);
      }
      
      return response.json();
    },
  },
  
  plots: {
    generate: async (request: any) => {
      const response = await fetch('/api/plots/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });
      
      if (!response.ok) {
        throw new Error(`Plot generation failed: ${response.statusText}`);
      }
      
      return response.json();
    },
    
    getTypes: async () => {
      const response = await fetch('/api/plots/types');
      
      if (!response.ok) {
        throw new Error(`Failed to get plot types: ${response.statusText}`);
      }
      
      return response.json();
    },
  },
  
  config: {
    getWatermarks: async () => {
      const response = await fetch('/api/config/watermarks');
      
      if (!response.ok) {
        throw new Error(`Failed to get watermarks: ${response.statusText}`);
      }
      
      return response.json();
    },
  },
};
```

## Phase 3: Integration & Testing (Weeks 8-9)

### 3.1 Vercel-Specific Testing

#### Performance Testing
```typescript
// tests/performance.test.ts
import { test, expect } from '@playwright/test';

test.describe('Vercel Performance Tests', () => {
  test('API route cold start time', async ({ page }) => {
    const startTime = Date.now();
    
    const response = await page.request.get('/api/health');
    
    const endTime = Date.now();
    const responseTime = endTime - startTime;
    
    expect(response.status()).toBe(200);
    expect(responseTime).toBeLessThan(5000); // 5 second cold start limit
  });
  
  test('File upload within size limits', async ({ page }) => {
    // Test with 4MB file (under 4.5MB limit)
    const file = await generateTestFile(4 * 1024 * 1024);
    
    await page.goto('/');
    await page.setInputFiles('input[type="file"]', file);
    
    await expect(page.locator('[data-testid="upload-success"]')).toBeVisible({
      timeout: 30000 // Allow for serverless processing time
    });
  });
  
  test('Plot generation within timeout', async ({ page }) => {
    // Upload test data
    await uploadTestData(page);
    
    // Generate plot
    const startTime = Date.now();
    await page.click('[data-testid="generate-plot"]');
    
    await expect(page.locator('[data-testid="plot-display"]')).toBeVisible({
      timeout: 30000 // Vercel function timeout
    });
    
    const endTime = Date.now();
    const generationTime = endTime - startTime;
    
    expect(generationTime).toBeLessThan(30000); // 30 second limit
  });
});
```

### 3.2 Error Handling for Serverless

```typescript
// lib/error-handling.ts
export class VercelError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public code?: string
  ) {
    super(message);
    this.name = 'VercelError';
  }
}

export function handleVercelErrors(error: any): VercelError {
  if (error.code === 'FUNCTION_INVOCATION_TIMEOUT') {
    return new VercelError(
      'The operation took too long to complete. Please try with a smaller file or simpler configuration.',
      408,
      'TIMEOUT'
    );
  }
  
  if (error.code === 'FUNCTION_PAYLOAD_TOO_LARGE') {
    return new VercelError(
      'The file is too large. Please use a file smaller than 4.5MB.',
      413,
      'PAYLOAD_TOO_LARGE'
    );
  }
  
  if (error.code === 'FUNCTION_INVOCATION_FAILED') {
    return new VercelError(
      'A server error occurred. Please try again.',
      500,
      'INVOCATION_FAILED'
    );
  }
  
  return new VercelError(
    error.message || 'An unexpected error occurred',
    500,
    'UNKNOWN'
  );
}
```

## Phase 4: Deployment & Production (Weeks 10-11)

### 4.1 Vercel Deployment Configuration

#### Environment Variables Setup
```bash
# Vercel CLI commands for environment setup
vercel env add PYTHON_PATH
vercel env add BWR_PLOTS_CONFIG
vercel env add SESSION_SECRET
vercel env add REDIS_URL # For production session storage
```

#### Production Optimizations
```javascript
// next.config.js (production)
/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverComponentsExternalPackages: ['pandas', 'numpy', 'plotly'],
  },
  api: {
    bodyParser: {
      sizeLimit: '4.5mb',
    },
  },
  // Enable compression
  compress: true,
  // Optimize images
  images: {
    domains: ['your-domain.com'],
    formats: ['image/webp', 'image/avif'],
  },
  // Security headers
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin',
          },
        ],
      },
    ];
  },
}

module.exports = nextConfig
```

### 4.2 Monitoring & Analytics

#### Vercel Analytics Setup
```typescript
// app/layout.tsx
import { Analytics } from '@vercel/analytics/react';
import { SpeedInsights } from '@vercel/speed-insights/next';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        {children}
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  );
}
```

#### Custom Monitoring
```typescript
// lib/monitoring.ts
export function trackEvent(eventName: string, properties?: Record<string, any>) {
  if (typeof window !== 'undefined' && window.va) {
    window.va('track', eventName, properties);
  }
}

export function trackError(error: Error, context?: Record<string, any>) {
  console.error('Application Error:', error, context);
  
  // Track in Vercel Analytics
  trackEvent('error', {
    message: error.message,
    stack: error.stack,
    ...context,
  });
}

// Usage in components
export function useErrorTracking() {
  return {
    trackError: (error: Error, context?: Record<string, any>) => {
      trackError(error, context);
    },
  };
}
```

## Migration Strategy

### 4.1 Deployment Strategy
1. **Preview Deployments**: Use Vercel's preview deployments for testing
2. **Gradual Rollout**: Use Vercel's edge config for feature flags
3. **Instant Rollback**: Leverage Vercel's instant rollback capability
4. **Domain Migration**: Gradual DNS migration to new Vercel deployment

### 4.2 Data Migration
- No persistent data migration needed (session-based)
- Configuration migration from Streamlit to Vercel environment variables
- User preference migration via local storage

### 4.3 Performance Considerations
- **Cold Starts**: Implement warming strategies for critical functions
- **File Size Limits**: Clear user communication about 4.5MB limit
- **Function Timeouts**: Optimize processing for 10-30 second limits
- **Edge Caching**: Leverage Vercel's edge network for static assets

## Risk Assessment & Mitigation

### 4.1 Vercel-Specific Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Function timeout on large files | High | Medium | File size validation, processing optimization |
| Cold start delays | Medium | High | Function warming, edge runtime usage |
| Payload size limits | Medium | Low | Clear user guidance, validation |
| Python dependency issues | High | Low | Thorough testing, dependency optimization |

### 4.2 Performance Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Slow plot generation | Medium | Medium | Optimization, progress indicators |
| Memory limits exceeded | High | Low | Data processing optimization |
| Concurrent user limits | Medium | Low | Serverless scaling, monitoring |

## Success Metrics

### 4.1 Technical Metrics (Adjusted for Vercel)
- API response times < 5 seconds (including cold starts)
- Frontend load times < 3 seconds
- 99.9% uptime via Vercel SLA
- Zero data loss incidents
- Function cold start times < 2 seconds

### 4.2 User Experience Metrics
- User satisfaction score > 8/10
- Task completion rate > 95%
- Error rate < 1%
- Mobile usability score > 80%

### 4.3 Performance Metrics (Vercel-Optimized)
- Support for concurrent serverless executions
- Handle files up to 4.5MB efficiently
- Generate plots within function timeout limits
- Optimal edge network utilization

## Future Enhancements

### 4.1 Vercel-Specific Enhancements
- **Edge Runtime**: Migrate suitable functions to Edge Runtime
- **Vercel KV**: Implement Redis-compatible session storage
- **Vercel Blob**: Large file storage for exports
- **Vercel Cron**: Automated cleanup jobs

### 4.2 Advanced Features
- Real-time collaboration via Vercel's WebSocket support
- Advanced analytics with Vercel Analytics
- A/B testing with Vercel's edge config
- Multi-region deployment optimization

## Conclusion

This updated refactor plan leverages Vercel's serverless platform to provide a scalable, performant solution while maintaining all existing BWR Plots functionality. The key advantages of the Vercel approach include:

1. **Zero Infrastructure Management**: No Docker containers or server management
2. **Automatic Scaling**: Serverless functions scale automatically
3. **Global Edge Network**: Fast content delivery worldwide
4. **Instant Deployments**: Quick iterations and rollbacks
5. **Built-in Analytics**: Performance monitoring out of the box

**Timeline**: 11 weeks total
**Effort**: 2-3 full-time developers
**Budget**: Development time + Vercel Pro plan costs

The plan maintains the same development phases while optimizing for Vercel's serverless architecture and deployment model. 