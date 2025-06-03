import { NextRequest, NextResponse } from 'next/server';
import { join } from 'path';
import { readdir, stat } from 'fs/promises';

export async function GET(request: NextRequest) {
  try {
    const sessionDirs = [
      join(process.cwd(), 'backend', 'storage', 'sessions'),
      join(process.cwd(), 'frontend', 'storage', 'sessions'),
      join(process.cwd(), 'storage', 'sessions'),
      'C:\\temp', // Windows temp directory
      '/tmp' // Unix temp directory
    ];

    const sessionInfo = [];

    for (const dirPath of sessionDirs) {
      try {
        const files = await readdir(dirPath);
        const sessionFiles = files.filter(file => file.startsWith('session_'));
        
        if (sessionFiles.length > 0) {
          const fileDetails = await Promise.all(
            sessionFiles.map(async (file) => {
              try {
                const filePath = join(dirPath, file);
                const stats = await stat(filePath);
                return {
                  name: file,
                  size: stats.size,
                  modified: stats.mtime.toISOString(),
                  isDirectory: stats.isDirectory()
                };
              } catch (error) {
                return {
                  name: file,
                  error: error instanceof Error ? error.message : 'Unknown error'
                };
              }
            })
          );

          sessionInfo.push({
            directory: dirPath,
            exists: true,
            files: fileDetails
          });
        } else {
          sessionInfo.push({
            directory: dirPath,
            exists: true,
            files: []
          });
        }
      } catch (error) {
        sessionInfo.push({
          directory: dirPath,
          exists: false,
          error: error instanceof Error ? error.message : 'Unknown error'
        });
      }
    }

    return NextResponse.json({
      success: true,
      sessionDirectories: sessionInfo,
      totalDirectories: sessionDirs.length,
      workingDirectory: process.cwd(),
      platform: process.platform
    });

  } catch (error) {
    console.error('Sessions debug error:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
      workingDirectory: process.cwd(),
      platform: process.platform
    });
  }
} 