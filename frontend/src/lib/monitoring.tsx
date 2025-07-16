/**
 * Performance Monitoring
 * ---
 * bwr-tools/frontend/src/lib/monitoring.ts
 * ---
 * Performance monitoring and metrics collection for layout system
 */

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Type Definitions                                                                   │
// └────────────────────────────────────────────────────────────────────────────────────┘

export interface PerformanceMetric {
  name: string;
  value: number;
  unit: string;
  timestamp: number;
  metadata?: Record<string, any>;
}

export interface LayoutMetrics {
  renderTime: number;
  layoutShift: number;
  resizeEvents: number;
  scrollEvents: number;
  componentCount: number;
  memoryUsage?: number;
}

export interface WebVitals {
  FCP?: number; // First Contentful Paint
  LCP?: number; // Largest Contentful Paint
  FID?: number; // First Input Delay
  CLS?: number; // Cumulative Layout Shift
  TTFB?: number; // Time to First Byte
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Performance Monitor Class                                                          │
// └────────────────────────────────────────────────────────────────────────────────────┘

class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private metrics: PerformanceMetric[] = [];
  private observers: Map<string, PerformanceObserver> = new Map();
  private enabled: boolean = false;
  private sampleRate: number = 0.1;

  private constructor() {
    this.initialize();
  }

  static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }

  private initialize(): void {
    // Check if monitoring is enabled
    this.enabled = process.env.NEXT_PUBLIC_ENABLE_PERFORMANCE_MONITORING === 'true';
    
    // Set sample rate
    const envSampleRate = process.env.NEXT_PUBLIC_PERFORMANCE_SAMPLE_RATE;
    if (envSampleRate) {
      this.sampleRate = parseFloat(envSampleRate);
    }

    if (!this.enabled || typeof window === 'undefined') {
      return;
    }

    // Only monitor for sampled sessions
    if (Math.random() > this.sampleRate) {
      this.enabled = false;
      return;
    }

    this.setupObservers();
    this.measureInitialMetrics();
  }

  private setupObservers(): void {
    // Observe layout shifts
    if ('PerformanceObserver' in window) {
      try {
        const clsObserver = new PerformanceObserver((list) => {
          let cls = 0;
          for (const entry of list.getEntries()) {
            if (!(entry as any).hadRecentInput) {
              cls += (entry as any).value;
            }
          }
          this.recordMetric({
            name: 'cumulative-layout-shift',
            value: cls,
            unit: 'score',
            timestamp: Date.now(),
          });
        });
        clsObserver.observe({ type: 'layout-shift', buffered: true });
        this.observers.set('cls', clsObserver);
      } catch (e) {
        // Layout shift observer not supported
      }

      // Observe largest contentful paint
      try {
        const lcpObserver = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const lastEntry = entries[entries.length - 1];
          this.recordMetric({
            name: 'largest-contentful-paint',
            value: lastEntry.startTime,
            unit: 'ms',
            timestamp: Date.now(),
          });
        });
        lcpObserver.observe({ type: 'largest-contentful-paint', buffered: true });
        this.observers.set('lcp', lcpObserver);
      } catch (e) {
        // LCP observer not supported
      }

      // Observe first input delay
      try {
        const fidObserver = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          entries.forEach((entry) => {
            this.recordMetric({
              name: 'first-input-delay',
              value: (entry as any).processingStart - entry.startTime,
              unit: 'ms',
              timestamp: Date.now(),
            });
          });
        });
        fidObserver.observe({ type: 'first-input', buffered: true });
        this.observers.set('fid', fidObserver);
      } catch (e) {
        // FID observer not supported
      }
    }
  }

  private measureInitialMetrics(): void {
    if (!window.performance || !window.performance.timing) return;

    const timing = window.performance.timing;
    const navigation = window.performance.getEntriesByType('navigation')[0] as any;

    // Time to First Byte
    if (timing.responseStart && timing.requestStart) {
      this.recordMetric({
        name: 'time-to-first-byte',
        value: timing.responseStart - timing.requestStart,
        unit: 'ms',
        timestamp: Date.now(),
      });
    }

    // First Contentful Paint
    if (navigation && navigation.loadEventEnd) {
      const paintEntries = window.performance.getEntriesByType('paint');
      const fcpEntry = paintEntries.find(entry => entry.name === 'first-contentful-paint');
      
      if (fcpEntry) {
        this.recordMetric({
          name: 'first-contentful-paint',
          value: fcpEntry.startTime,
          unit: 'ms',
          timestamp: Date.now(),
        });
      }
    }
  }

  recordMetric(metric: PerformanceMetric): void {
    if (!this.enabled) return;

    this.metrics.push(metric);

    // Log in development
    if (process.env.NODE_ENV === 'development') {
      console.log(`[Performance] ${metric.name}: ${metric.value}${metric.unit}`);
    }

    // Send to analytics (implement based on your analytics provider)
    this.sendToAnalytics(metric);
  }

  private sendToAnalytics(metric: PerformanceMetric): void {
    // Example: Send to Google Analytics, Segment, or custom endpoint
    if (typeof window !== 'undefined' && (window as any).gtag) {
      (window as any).gtag('event', 'performance_metric', {
        metric_name: metric.name,
        metric_value: metric.value,
        metric_unit: metric.unit,
      });
    }

    // Or send to custom endpoint
    // fetch('/api/metrics', {
    //   method: 'POST',
    //   body: JSON.stringify(metric),
    //   headers: { 'Content-Type': 'application/json' },
    // });
  }

  measureLayoutRender(componentName: string, callback: () => void): void {
    if (!this.enabled) {
      callback();
      return;
    }

    const startTime = performance.now();
    callback();
    const endTime = performance.now();

    this.recordMetric({
      name: 'layout-render',
      value: endTime - startTime,
      unit: 'ms',
      timestamp: Date.now(),
      metadata: { component: componentName },
    });
  }

  startMeasure(name: string): () => void {
    if (!this.enabled) {
      return () => {};
    }

    const startTime = performance.now();
    
    return () => {
      const endTime = performance.now();
      this.recordMetric({
        name,
        value: endTime - startTime,
        unit: 'ms',
        timestamp: Date.now(),
      });
    };
  }

  getMetrics(): PerformanceMetric[] {
    return [...this.metrics];
  }

  getWebVitals(): WebVitals {
    const vitals: WebVitals = {};

    this.metrics.forEach(metric => {
      switch (metric.name) {
        case 'first-contentful-paint':
          vitals.FCP = metric.value;
          break;
        case 'largest-contentful-paint':
          vitals.LCP = metric.value;
          break;
        case 'first-input-delay':
          vitals.FID = metric.value;
          break;
        case 'cumulative-layout-shift':
          vitals.CLS = metric.value;
          break;
        case 'time-to-first-byte':
          vitals.TTFB = metric.value;
          break;
      }
    });

    return vitals;
  }

  destroy(): void {
    this.observers.forEach(observer => observer.disconnect());
    this.observers.clear();
    this.metrics = [];
  }
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Public API                                                                         │
// └────────────────────────────────────────────────────────────────────────────────────┘

export const performanceMonitor = PerformanceMonitor.getInstance();

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ React Hooks                                                                        │
// └────────────────────────────────────────────────────────────────────────────────────┘

import { useEffect, useRef } from 'react';

export function useLayoutPerformance(componentName: string) {
  const renderCount = useRef(0);

  useEffect(() => {
    renderCount.current += 1;
    
    const measure = performanceMonitor.startMeasure(`${componentName}-mount`);
    
    return () => {
      measure();
      
      // Track excessive re-renders
      if (renderCount.current > 10) {
        performanceMonitor.recordMetric({
          name: 'excessive-rerenders',
          value: renderCount.current,
          unit: 'count',
          timestamp: Date.now(),
          metadata: { component: componentName },
        });
      }
    };
  }, [componentName]);
}

export function useMeasure(name: string) {
  return {
    start: () => performanceMonitor.startMeasure(name),
    measure: (callback: () => void) => {
      performanceMonitor.measureLayoutRender(name, callback);
    },
  };
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Layout-specific Monitoring                                                         │
// └────────────────────────────────────────────────────────────────────────────────────┘

export class LayoutMonitor {
  private resizeCount = 0;
  private scrollCount = 0;
  private lastResize = 0;
  private lastScroll = 0;

  trackResize(): void {
    const now = Date.now();
    if (now - this.lastResize > 100) { // Debounce
      this.resizeCount++;
      this.lastResize = now;
      
      performanceMonitor.recordMetric({
        name: 'layout-resize',
        value: 1,
        unit: 'count',
        timestamp: now,
      });
    }
  }

  trackScroll(): void {
    const now = Date.now();
    if (now - this.lastScroll > 100) { // Debounce
      this.scrollCount++;
      this.lastScroll = now;
    }
  }

  getMetrics(): LayoutMetrics {
    return {
      renderTime: 0, // Set by component
      layoutShift: 0, // Set by observer
      resizeEvents: this.resizeCount,
      scrollEvents: this.scrollCount,
      componentCount: 0, // Set by component tree
      memoryUsage: (performance as any).memory?.usedJSHeapSize,
    };
  }

  reset(): void {
    this.resizeCount = 0;
    this.scrollCount = 0;
  }
}

// ┌────────────────────────────────────────────────────────────────────────────────────┐
// │ Debug Component                                                                    │
// └────────────────────────────────────────────────────────────────────────────────────┘

export function PerformanceDebugPanel() {
  const [metrics, setMetrics] = useState<WebVitals>({});
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (process.env.NODE_ENV !== 'development') return;

    const interval = setInterval(() => {
      setMetrics(performanceMonitor.getWebVitals());
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  if (process.env.NODE_ENV !== 'development' || !show) {
    return (
      <button
        onClick={() => setShow(true)}
        className="fixed bottom-4 left-4 bg-[var(--color-bg-elevated)] border border-[var(--color-border)] rounded-lg shadow-lg p-2 z-50"
        title="Show performance metrics"
      >
        📊
      </button>
    );
  }

  return (
    <div className="fixed bottom-4 left-4 bg-[var(--color-bg-elevated)] border border-[var(--color-border)] rounded-lg shadow-lg p-4 z-50 max-w-sm">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">
          Performance Metrics
        </h3>
        <button
          onClick={() => setShow(false)}
          className="text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]"
        >
          ✕
        </button>
      </div>
      
      <div className="space-y-2 text-xs">
        <div className="flex justify-between">
          <span className="text-[var(--color-text-muted)]">FCP:</span>
          <span className={`${metrics.FCP && metrics.FCP < 1800 ? 'text-green-500' : 'text-red-500'}`}>
            {metrics.FCP ? `${Math.round(metrics.FCP)}ms` : '-'}
          </span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-[var(--color-text-muted)]">LCP:</span>
          <span className={`${metrics.LCP && metrics.LCP < 2500 ? 'text-green-500' : 'text-red-500'}`}>
            {metrics.LCP ? `${Math.round(metrics.LCP)}ms` : '-'}
          </span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-[var(--color-text-muted)]">FID:</span>
          <span className={`${metrics.FID && metrics.FID < 100 ? 'text-green-500' : 'text-red-500'}`}>
            {metrics.FID ? `${Math.round(metrics.FID)}ms` : '-'}
          </span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-[var(--color-text-muted)]">CLS:</span>
          <span className={`${metrics.CLS && metrics.CLS < 0.1 ? 'text-green-500' : 'text-red-500'}`}>
            {metrics.CLS ? metrics.CLS.toFixed(3) : '-'}
          </span>
        </div>
      </div>
    </div>
  );
}

import { useState } from 'react';