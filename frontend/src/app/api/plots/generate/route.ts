import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import { join } from 'path';
import { writeFileSync, unlinkSync } from 'fs';
import { randomUUID } from 'crypto';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Validate request
    if (!body.session_id || !body.plot_type || !body.configuration) {
      return NextResponse.json({ 
        error: 'Missing required fields: session_id, plot_type, and configuration' 
      }, { status: 400 });
    }
    
    // Validate session ID format
    if (!body.session_id.match(/^session_\d+_\d+$/)) {
      return NextResponse.json({ error: 'Invalid session ID format' }, { status: 400 });
    }
    
    // Validate plot type
    const validPlotTypes = ['line', 'bar', 'scatter', 'area', 'histogram', 'box', 'heatmap'];
    if (!validPlotTypes.includes(body.plot_type)) {
      return NextResponse.json({ 
        error: `Invalid plot type. Must be one of: ${validPlotTypes.join(', ')}` 
      }, { status: 400 });
    }
    
    // Process with Python plot generator
    const result = await processPythonScript('plot_generator', {
      session_id: body.session_id,
      plot_type: body.plot_type,
      configuration: body.configuration,
      data_processing: body.data_processing || {}
    });
    
    if (result.error) {
      return NextResponse.json({ error: result.error }, { status: 400 });
    }
    
    return NextResponse.json(result);
    
  } catch (error) {
    console.error('Plot generation error:', error);
    return NextResponse.json({ 
      error: 'Plot generation failed. Please try again.' 
    }, { status: 500 });
  }
}

async function processPythonScript(script: string, args: any): Promise<any> {
  return new Promise((resolve, reject) => {
    const pythonPath = process.env.PYTHON_PATH || 'python';
    const utilsPath = join(process.cwd(), 'utils', `${script}.py`);
    
    console.log(`[PLOT_GENERATION] Executing: ${pythonPath} ${utilsPath}`);
    console.log(`[PLOT_GENERATION] Arguments:`, JSON.stringify(args));
    
    // Write arguments to a temporary file to avoid Windows command line issues
    const tempDir = process.env.TEMP || process.env.TMP || (process.platform === 'win32' ? 'C:\\temp' : '/tmp');
    const argsFile = join(tempDir, `args_${randomUUID()}.json`);
    
    try {
      writeFileSync(argsFile, JSON.stringify(args), 'utf8');
      console.log(`[PLOT_GENERATION] Arguments written to: ${argsFile}`);
      
      // Pass the file path instead of the JSON directly
      const python = spawn(pythonPath, [utilsPath, argsFile], {
        stdio: ['pipe', 'pipe', 'pipe'],
        timeout: 30000 // 30 seconds timeout for plot generation
      });
      
      let output = '';
      let errorOutput = '';
      
      python.stdout.on('data', (data) => {
        const chunk = data.toString();
        output += chunk;
        console.log(`[PLOT_STDOUT] ${chunk.trim()}`);
      });
      
      python.stderr.on('data', (data) => {
        const chunk = data.toString();
        errorOutput += chunk;
        console.error(`[PLOT_STDERR] ${chunk.trim()}`);
        
        // Log individual lines for better debugging
        chunk.split('\n').forEach((line: string) => {
          if (line.trim()) {
            console.error(`[PLOT_STDERR_LINE] ${line.trim()}`);
          }
        });
      });
      
      python.on('close', (code) => {
        console.log(`[PLOT_GENERATION] Process exited with code: ${code}`);
        
        // Clean up the temporary file
        try {
          unlinkSync(argsFile);
          console.log(`[PLOT_GENERATION] Cleaned up args file: ${argsFile}`);
        } catch (cleanupError) {
          console.warn(`[PLOT_GENERATION] Failed to cleanup args file: ${cleanupError}`);
        }
        
        if (code === 0) {
          try {
            const result = JSON.parse(output.trim());
            console.log(`[PLOT_GENERATION] Successfully parsed result`);
            resolve(result);
          } catch (e) {
            console.error('[PLOT_GENERATION] Python output parsing error:', e);
            console.error('[PLOT_GENERATION] Raw output:', output);
            console.error('[PLOT_GENERATION] Error output:', errorOutput);
            resolve({ error: 'Invalid response from plot generator' });
          }
        } else {
          console.error(`[PLOT_GENERATION] Python script failed with code ${code}`);
          console.error('[PLOT_GENERATION] Error output:', errorOutput);
          resolve({ error: `Plot generation failed (code ${code})`, stderr: errorOutput });
        }
      });
      
      python.on('error', (error) => {
        console.error('[PLOT_GENERATION] Python spawn error:', error);
        
        // Clean up the temporary file on error
        try {
          unlinkSync(argsFile);
        } catch (cleanupError) {
          console.warn(`[PLOT_GENERATION] Failed to cleanup args file on error: ${cleanupError}`);
        }
        
        resolve({ error: 'Failed to start plot generator', spawnError: error.message });
      });
      
      python.on('timeout', () => {
        python.kill();
        
        // Clean up the temporary file on timeout
        try {
          unlinkSync(argsFile);
        } catch (cleanupError) {
          console.warn(`[PLOT_GENERATION] Failed to cleanup args file on timeout: ${cleanupError}`);
        }
        
        resolve({ error: 'Plot generation timed out. Please try with a smaller dataset or simpler configuration.' });
      });
      
    } catch (fileError) {
      console.error('[PLOT_GENERATION] Failed to write arguments file:', fileError);
      resolve({ 
        error: 'Failed to prepare plot generator arguments', 
        fileError: fileError instanceof Error ? fileError.message : 'Unknown file error' 
      });
    }
  });
}

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '4.5mb',
    },
  },
  maxDuration: 30, // 30 seconds for plot generation
}; 