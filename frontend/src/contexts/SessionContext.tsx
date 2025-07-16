'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { DummyDataType } from '@/lib/dummyData';

interface UploadedData {
  columns: string[];
  data: Record<string, any>[];
  originalFileName?: string;
  rowCount: number;
  dataTypes: Record<string, string>;
}

interface PlotConfig {
  title?: string;
  subtitle?: string;
  source?: string;
  x_column?: string;
  y_column?: string;
  xAxis?: string;
  yAxis?: string | string[];
  axis_config?: {
    x_title?: string;
    y_title?: string;
  };
  // Add more config options as needed
}

interface Session {
  id: string;
  sessionId: string;
  uploadedData?: UploadedData;
  plotType?: string;
  plotConfig?: PlotConfig;
  suggestedPlotType?: DummyDataType;
  lastGeneratedPlot?: any;
  // Panel visibility states for dashboard
  panelState?: {
    dataPanel: boolean;
    configPanel: boolean;
  };
}

interface SessionContextType {
  session: Session | null;
  updateSession: (updates: Partial<Session>) => void;
  clearSession: () => void;
  // Remove step-based navigation
  // nextStep: () => void;
  // previousStep: () => void;
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(() => {
    const sessionId = `session_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
    return {
      id: sessionId,
      sessionId: sessionId,
      panelState: {
        dataPanel: true,
        configPanel: true
      }
    };
  });

  // Save session to localStorage whenever it changes
  useEffect(() => {
    if (session) {
      localStorage.setItem('bwr-plots-session', JSON.stringify(session));
      
      // Also save session data for Python script if we have uploaded data
      if (session.uploadedData?.data && session.sessionId) {
        saveSessionForPython(session);
      }
    }
  }, [session]);

  // Save session data in format Python script expects
  const saveSessionForPython = async (session: Session) => {
    if (!session.uploadedData?.data || !session.sessionId) return;
    
    try {
      const sessionData = {
        session_id: session.sessionId,
        current_data: JSON.stringify(session.uploadedData.data),
        columns: session.uploadedData.columns,
        row_count: session.uploadedData.rowCount,
        data_types: session.uploadedData.dataTypes,
        created_at: new Date().toISOString()
      };

      // Save via API endpoint that will write to temp file
      await fetch('/api/session/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sessionData)
      });
    } catch (error) {
      console.error('Failed to save session for Python script:', error);
    }
  };

  // Load session from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem('bwr-plots-session');
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setSession(parsed);
      } catch (error) {
        console.error('Failed to parse stored session:', error);
      }
    }
  }, []);

  const updateSession = (updates: Partial<Session>) => {
    setSession(prev => prev ? { ...prev, ...updates } : null);
  };

  const clearSession = () => {
    localStorage.removeItem('bwr-plots-session');
    const sessionId = `session_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
    setSession({
      id: sessionId,
      sessionId: sessionId,
      panelState: {
        dataPanel: true,
        configPanel: true
      }
    });
  };

  return (
    <SessionContext.Provider value={{ session, updateSession, clearSession }}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const context = useContext(SessionContext);
  if (context === undefined) {
    throw new Error('useSession must be used within a SessionProvider');
  }
  return context;
}