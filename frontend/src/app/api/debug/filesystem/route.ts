import { NextRequest, NextResponse } from 'next/server';
import { readdirSync, existsSync, statSync } from 'fs';
import { join } from 'path';

export async function GET(request: NextRequest) {
  try {
    const cwd = process.cwd();
    
    // Function to safely list directory contents
    const listDir = (path: string, depth = 0) => {
      try {
        if (!existsSync(path) || depth > 2) return null;
        
        const items = readdirSync(path);
        const result: any = {};
        
        items.forEach(item => {
          const itemPath = join(path, item);
          try {
            const stat = statSync(itemPath);
            if (stat.isDirectory()) {
              result[`${item}/`] = depth < 2 ? listDir(itemPath, depth + 1) : '[directory]';
            } else {
              result[item] = `[file - ${stat.size} bytes]`;
            }
          } catch (e) {
            result[item] = '[error reading]';
          }
        });
        
        return result;
      } catch (e) {
        return `[error: ${e instanceof Error ? e.message : 'unknown'}]`;
      }
    };
    
    // Check various important paths
    const pathsToCheck = [
      cwd,
      join(cwd, 'utils'),
      join(cwd, 'frontend'),
      join(cwd, 'frontend', 'utils'),
      join(cwd, 'src'),
      join(cwd, '..'),
      join(cwd, '..', 'frontend'),
      join(cwd, '..', 'frontend', 'utils'),
    ];
    
    const pathStatus: any = {};
    pathsToCheck.forEach(path => {
      pathStatus[path] = {
        exists: existsSync(path),
        isDirectory: existsSync(path) ? (() => {
          try {
            return statSync(path).isDirectory();
          } catch {
            return false;
          }
        })() : false,
        contents: existsSync(path) ? listDir(path, 0) : null
      };
    });
    
    // Specific file checks
    const filesToCheck = [
      join(cwd, 'utils', 'plot_generator.py'),
      join(cwd, 'frontend', 'utils', 'plot_generator.py'),
      join(cwd, 'utils', 'data_processor.py'),
      join(cwd, 'frontend', 'utils', 'data_processor.py'),
    ];
    
    const fileStatus: any = {};
    filesToCheck.forEach(file => {
      fileStatus[file] = {
        exists: existsSync(file),
        size: existsSync(file) ? (() => {
          try {
            return statSync(file).size;
          } catch {
            return 'unknown';
          }
        })() : 'n/a'
      };
    });
    
    return NextResponse.json({
      success: true,
      environment: {
        cwd: cwd,
        platform: process.platform,
        nodeEnv: process.env.NODE_ENV,
        tempDir: process.env.TEMP || process.env.TMP || 'not set'
      },
      pathStatus,
      fileStatus,
      directoryStructure: {
        root: listDir(cwd, 0)
      }
    });
    
  } catch (error) {
    console.error('Filesystem debug error:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
      cwd: process.cwd()
    });
  }
} 