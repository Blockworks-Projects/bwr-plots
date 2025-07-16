import { useState, useEffect } from 'react';

interface SessionData {
  sessionId: string | null;
  currentStep: number;
  lastActivity: number;
  autoLoadDisabled?: boolean; // New flag to prevent auto-loading
}

const SESSION_KEY = 'bwr-plots-session';
const SESSION_TIMEOUT = 30 * 60 * 1000; // 30 minutes

export function useSession() {
  const [sessionData, setSessionData] = useState<SessionData>({
    sessionId: null,
    currentStep: 1,
    lastActivity: Date.now(),
    autoLoadDisabled: false,
  });

  // Load session from localStorage on mount, but with explicit user control
  useEffect(() => {
    // Check if we're on the client side
    if (typeof window === 'undefined') return;
    
    console.log('[useSession] Checking for stored session data');
    const stored = localStorage.getItem(SESSION_KEY);
    if (stored) {
      try {
        const parsed = JSON.parse(stored) as SessionData;
        console.log(`[useSession] Found stored session: ${parsed.sessionId}, step: ${parsed.currentStep}`);
        
        // Check if session has expired
        if (Date.now() - parsed.lastActivity < SESSION_TIMEOUT) {
          console.log('[useSession] Session is still valid');
          // Don't auto-load the session - just mark it as available
          // The user will need to explicitly choose to restore it
          setSessionData(prev => ({
            ...prev,
            // Don't set sessionId here - require user action
            autoLoadDisabled: true // Flag that we have a session available but not auto-loaded
          }));
        } else {
          console.log('[useSession] Session expired, clearing');
          localStorage.removeItem(SESSION_KEY);
        }
      } catch (error) {
        console.warn('[useSession] Failed to parse stored session data:', error);
        localStorage.removeItem(SESSION_KEY);
      }
    } else {
      console.log('[useSession] No stored session found');
    }
  }, []);

  // Save to localStorage whenever sessionData changes
  useEffect(() => {
    // Check if we're on the client side
    if (typeof window === 'undefined') return;
    
    if (sessionData.sessionId) {
      const dataToStore = {
        ...sessionData,
        lastActivity: Date.now(),
      };
      console.log(`[useSession] Saving session data for: ${sessionData.sessionId}`);
      localStorage.setItem(SESSION_KEY, JSON.stringify(dataToStore));
    }
  }, [sessionData]);

  const setSessionId = (sessionId: string | null) => {
    console.log(`[useSession] Setting session ID: ${sessionId}`);
    setSessionData(prev => ({
      ...prev,
      sessionId,
      lastActivity: Date.now(),
      autoLoadDisabled: false, // Reset the flag when explicitly setting a session
    }));
  };

  const setCurrentStep = (step: number) => {
    console.log(`[useSession] Setting current step: ${step}`);
    setSessionData(prev => ({
      ...prev,
      currentStep: step,
      lastActivity: Date.now(),
    }));
  };

  const nextStep = () => {
    console.log(`[useSession] Moving to next step from: ${sessionData.currentStep}`);
    setSessionData(prev => ({
      ...prev,
      currentStep: prev.currentStep + 1,
      lastActivity: Date.now(),
    }));
  };

  const previousStep = () => {
    console.log(`[useSession] Moving to previous step from: ${sessionData.currentStep}`);
    setSessionData(prev => ({
      ...prev,
      currentStep: Math.max(1, prev.currentStep - 1),
      lastActivity: Date.now(),
    }));
  };

  const clearSession = () => {
    console.log('[useSession] Clearing session');
    setSessionData({
      sessionId: null,
      currentStep: 1,
      lastActivity: Date.now(),
      autoLoadDisabled: false,
    });
    if (typeof window !== 'undefined') {
      localStorage.removeItem(SESSION_KEY);
    }
  };

  // New method to restore a session from localStorage
  const restoreStoredSession = () => {
    console.log('[useSession] Attempting to restore stored session');
    if (typeof window === 'undefined') return false;
    
    const stored = localStorage.getItem(SESSION_KEY);
    if (stored) {
      try {
        const parsed = JSON.parse(stored) as SessionData;
        if (Date.now() - parsed.lastActivity < SESSION_TIMEOUT) {
          console.log(`[useSession] Restoring session: ${parsed.sessionId}`);
          setSessionData({
            ...parsed,
            lastActivity: Date.now(),
            autoLoadDisabled: false,
          });
          return true;
        }
      } catch (error) {
        console.error('[useSession] Failed to restore session:', error);
      }
    }
    return false;
  };

  // Check if there's a stored session available to restore
  const hasStoredSession = () => {
    if (typeof window === 'undefined') return false;
    
    const stored = localStorage.getItem(SESSION_KEY);
    if (stored) {
      try {
        const parsed = JSON.parse(stored) as SessionData;
        return Date.now() - parsed.lastActivity < SESSION_TIMEOUT;
      } catch {
        return false;
      }
    }
    return false;
  };

  const updateActivity = () => {
    setSessionData(prev => ({
      ...prev,
      lastActivity: Date.now(),
    }));
  };

  const isSessionExpired = () => {
    return Date.now() - sessionData.lastActivity > SESSION_TIMEOUT;
  };

  const getTimeRemaining = () => {
    const elapsed = Date.now() - sessionData.lastActivity;
    const remaining = SESSION_TIMEOUT - elapsed;
    return Math.max(0, remaining);
  };

  return {
    sessionId: sessionData.sessionId,
    currentStep: sessionData.currentStep,
    lastActivity: sessionData.lastActivity,
    autoLoadDisabled: sessionData.autoLoadDisabled,
    setSessionId,
    setCurrentStep,
    nextStep,
    previousStep,
    clearSession,
    restoreStoredSession,
    hasStoredSession,
    updateActivity,
    isSessionExpired,
    getTimeRemaining,
    hasActiveSession: !!sessionData.sessionId && !isSessionExpired(),
  };
} 