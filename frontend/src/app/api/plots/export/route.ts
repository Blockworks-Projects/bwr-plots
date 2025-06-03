import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import { join } from 'path';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Validate request
    if (!body.session_id || !body.plot_type || !body.configuration || !body.format) {
      return NextResponse.json({ 
        error: 'Missing required fields: session_id, plot_type, configuration, and format' 
      }, { status: 400 });
    }
    
    // Validate session ID format
    if (!body.session_id.match(/^session_\d+_\d+$/)) {
      return NextResponse.json({ error: 'Invalid session ID format' }, { status: 400 });
    }
    
    // Validate export format
    const validFormats = ['html', 'png', 'pdf', 'svg', 'json'];
    if (!validFormats.includes(body.format)) {
      return NextResponse.json({ 
        error: `Invalid format. Must be one of: ${validFormats.join(', ')}` 
      }, { status: 400 });
    }
    
    // Generate the plot first
    const plotResult = await processPythonScript('plot_generator', {
      session_id: body.session_id,
      plot_type: body.plot_type,
      configuration: body.configuration,
      data_processing: body.data_processing || {}
    });
    
    if (plotResult.error) {
      return NextResponse.json({ error: plotResult.error }, { status: 400 });
    }
    
    // Handle different export formats
    let exportData;
    let contentType;
    let filename;
    
    switch (body.format) {
      case 'html':
        exportData = plotResult.plot_html;
        contentType = 'text/html';
        filename = `plot_${body.session_id}.html`;
        break;
        
      case 'json':
        exportData = JSON.stringify(plotResult.plot_json, null, 2);
        contentType = 'application/json';
        filename = `plot_${body.session_id}.json`;
        break;
        
      default:
        // For PNG, PDF, SVG - would need additional processing
        return NextResponse.json({ 
          error: `Export format '${body.format}' not yet implemented` 
        }, { status: 501 });
    }
    
    // Return the export data
    return new NextResponse(exportData, {
      headers: {
        'Content-Type': contentType,
        'Content-Disposition': `attachment; filename="${filename}"`,
        'Cache-Control': 'no-cache'
      }
    });
    
  } catch (error) {
    console.error('Export error:', error);
    return NextResponse.json({ 
      error: 'Plot export failed. Please try again.' 
    }, { status: 500 });
  }
}

async function processPythonScript(script: string, args: any): Promise<any> {
  return new Promise((resolve, reject) => {
    const pythonPath = process.env.PYTHON_PATH || 'python';
    const utilsPath = join(process.cwd(), 'utils', `${script}.py`);
    
    // Pass arguments as command line argument
    const python = spawn(pythonPath, [utilsPath, JSON.stringify(args)], {
      stdio: ['pipe', 'pipe', 'pipe'],
      timeout: 30000 // 30 seconds timeout
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
      if (code === 0) {
        try {
          const result = JSON.parse(output.trim());
          resolve(result);
        } catch (e) {
          console.error('Python output parsing error:', e);
          console.error('Raw output:', output);
          console.error('Error output:', errorOutput);
          resolve({ error: 'Invalid response from plot generator' });
        }
      } else {
        console.error('Python script error:', errorOutput);
        resolve({ error: `Plot generation failed (code ${code})` });
      }
    });
    
    python.on('error', (error) => {
      console.error('Python spawn error:', error);
      resolve({ error: 'Failed to start plot generator' });
    });
    
    python.on('timeout', () => {
      python.kill();
      resolve({ error: 'Plot generation timed out' });
    });
  });
} 