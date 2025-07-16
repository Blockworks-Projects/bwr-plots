import { NextRequest, NextResponse } from 'next/server';
import { writeFileSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';

export async function POST(request: NextRequest) {
  try {
    const sessionData = await request.json();
    
    // Validate session data
    if (!sessionData.session_id || !sessionData.current_data) {
      return NextResponse.json(
        { error: 'Missing required session data' },
        { status: 400 }
      );
    }
    
    // Save to temp directory
    const tempDir = tmpdir();
    const sessionFile = join(tempDir, `${sessionData.session_id}.json`);
    
    // Write session data to file
    writeFileSync(sessionFile, JSON.stringify(sessionData, null, 2), 'utf8');
    
    console.log(`[SESSION] Saved session data to: ${sessionFile}`);
    console.log(`[SESSION] Session ID: ${sessionData.session_id}`);
    console.log(`[SESSION] Data length: ${sessionData.current_data.length} chars`);
    
    return NextResponse.json({ 
      success: true, 
      sessionFile,
      sessionId: sessionData.session_id
    });
    
  } catch (error) {
    console.error('Session save error:', error);
    return NextResponse.json(
      { error: 'Failed to save session' },
      { status: 500 }
    );
  }
}