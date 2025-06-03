import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import { join } from 'path';

export async function GET(request: NextRequest) {
  try {
    // Check Python availability and version
    const pythonInfo = await checkPythonEnvironment();
    
    // Check if plot generator script exists
    const scriptPath = join(process.cwd(), 'frontend', 'utils', 'plot_generator.py');
    const scriptExists = await checkFileExists(scriptPath);
    
    // Check if backend directory exists
    const backendPath = join(process.cwd(), 'backend');
    const backendExists = await checkFileExists(backendPath);
    
    return NextResponse.json({
      success: true,
      environment: {
        python: pythonInfo,
        script: {
          path: scriptPath,
          exists: scriptExists
        },
        backend: {
          path: backendPath,
          exists: backendExists
        },
        nodeEnv: process.env.NODE_ENV,
        platform: process.platform,
        pythonPath: process.env.PYTHON_PATH || 'python',
        workingDirectory: process.cwd(),
        tempDir: process.env.TEMP || process.env.TMP || (process.platform === 'win32' ? 'C:\\temp' : '/tmp')
      }
    });
  } catch (error) {
    console.error('Python environment check error:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
      environment: {
        nodeEnv: process.env.NODE_ENV,
        platform: process.platform,
        pythonPath: process.env.PYTHON_PATH || 'python',
        workingDirectory: process.cwd()
      }
    });
  }
}

async function checkPythonEnvironment(): Promise<any> {
  return new Promise((resolve) => {
    const pythonPath = process.env.PYTHON_PATH || 'python';
    
    const python = spawn(pythonPath, ['--version'], {
      stdio: ['pipe', 'pipe', 'pipe'],
      timeout: 5000
    });
    
    let output = '';
    let errorOutput = '';
    
    python.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    python.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });
    
    python.on('close', (code) => {
      resolve({
        available: code === 0,
        version: output.trim() || errorOutput.trim(),
        exitCode: code,
        command: `${pythonPath} --version`
      });
    });
    
    python.on('error', (error) => {
      resolve({
        available: false,
        error: error.message,
        command: `${pythonPath} --version`
      });
    });
    
    python.on('timeout', () => {
      python.kill();
      resolve({
        available: false,
        error: 'Python version check timed out',
        command: `${pythonPath} --version`
      });
    });
  });
}

async function checkFileExists(filePath: string): Promise<boolean> {
  try {
    const fs = require('fs').promises;
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
} 