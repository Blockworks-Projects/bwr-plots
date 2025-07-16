# BWR Plots Testing Strategy

## Overview
Comprehensive testing strategy for the new layout architecture, covering unit tests, integration tests, visual regression, performance, and accessibility.

## Testing Pyramid

```
         /\
        /  \  E2E Tests (10%)
       /────\
      /      \  Integration Tests (30%)
     /────────\
    /          \  Unit Tests (60%)
   /────────────\
```

## Test Categories

### 1. Unit Tests (60%)

#### Layout Primitives
```typescript
// Stack.test.tsx
import { render, screen } from '@testing-library/react';
import { Stack } from '@/components/layout/Stack';

describe('Stack Component', () => {
  describe('Layout Behavior', () => {
    it('renders children in correct order', () => {
      render(
        <Stack>
          <div>First</div>
          <div>Second</div>
          <div>Third</div>
        </Stack>
      );
      
      const elements = screen.getAllByText(/^(First|Second|Third)$/);
      expect(elements[0]).toHaveTextContent('First');
      expect(elements[1]).toHaveTextContent('Second');
      expect(elements[2]).toHaveTextContent('Third');
    });

    it('applies correct spacing between items', () => {
      const { container } = render(
        <Stack spacing="md" data-testid="stack">
          <div>Item 1</div>
          <div>Item 2</div>
        </Stack>
      );
      
      const stack = container.querySelector('[data-testid="stack"]');
      expect(stack).toHaveStyle({ gap: '1rem' });
    });

    it('changes direction based on prop', () => {
      const { rerender, container } = render(
        <Stack direction="horizontal" data-testid="stack">
          <div>Item</div>
        </Stack>
      );
      
      let stack = container.querySelector('[data-testid="stack"]');
      expect(stack).toHaveStyle({ flexDirection: 'row' });
      
      rerender(
        <Stack direction="vertical" data-testid="stack">
          <div>Item</div>
        </Stack>
      );
      
      stack = container.querySelector('[data-testid="stack"]');
      expect(stack).toHaveStyle({ flexDirection: 'column' });
    });
  });

  describe('Responsive Behavior', () => {
    it('applies responsive spacing', () => {
      render(
        <Stack spacing={{ base: 'sm', md: 'lg' }} data-testid="stack">
          <div>Item</div>
        </Stack>
      );
      
      const stack = screen.getByTestId('stack');
      
      // Test mobile viewport
      window.matchMedia = createMatchMedia(400);
      expect(stack).toHaveClass('gap-2'); // sm spacing
      
      // Test desktop viewport
      window.matchMedia = createMatchMedia(1024);
      expect(stack).toHaveClass('gap-8'); // lg spacing
    });
  });

  describe('Accessibility', () => {
    it('forwards ref correctly', () => {
      const ref = createRef<HTMLDivElement>();
      render(<Stack ref={ref}>Content</Stack>);
      
      expect(ref.current).toBeInstanceOf(HTMLDivElement);
    });

    it('spreads additional props', () => {
      render(
        <Stack 
          data-testid="custom-stack" 
          aria-label="Navigation stack"
        >
          Content
        </Stack>
      );
      
      const stack = screen.getByTestId('custom-stack');
      expect(stack).toHaveAttribute('aria-label', 'Navigation stack');
    });
  });
});
```

#### Hooks Testing
```typescript
// useResponsive.test.ts
import { renderHook, act } from '@testing-library/react-hooks';
import { useResponsive } from '@/hooks/useResponsive';

describe('useResponsive Hook', () => {
  beforeEach(() => {
    // Reset matchMedia mock
    window.matchMedia = jest.fn().mockImplementation(query => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: jest.fn(),
      removeListener: jest.fn(),
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }));
  });

  it('detects current breakpoint', () => {
    // Mock desktop viewport
    window.matchMedia = createMatchMedia(1200);
    
    const { result } = renderHook(() => useResponsive());
    
    expect(result.current.isDesktop).toBe(true);
    expect(result.current.isTablet).toBe(false);
    expect(result.current.isMobile).toBe(false);
  });

  it('updates on viewport change', () => {
    const { result, rerender } = renderHook(() => useResponsive());
    
    // Start with mobile
    act(() => {
      window.resizeTo(400, 800);
      window.dispatchEvent(new Event('resize'));
    });
    
    expect(result.current.isMobile).toBe(true);
    
    // Change to tablet
    act(() => {
      window.resizeTo(768, 1024);
      window.dispatchEvent(new Event('resize'));
    });
    
    expect(result.current.isTablet).toBe(true);
    expect(result.current.isMobile).toBe(false);
  });

  it('provides helper methods', () => {
    const { result } = renderHook(() => useResponsive());
    
    expect(result.current.up('md')).toBe(window.innerWidth >= 768);
    expect(result.current.down('md')).toBe(window.innerWidth < 768);
    expect(result.current.between('sm', 'lg')).toBe(
      window.innerWidth >= 640 && window.innerWidth < 1024
    );
  });
});
```

#### Utility Functions
```typescript
// layout-utils.test.ts
import { 
  calculateGridColumns, 
  getSpacingValue,
  parseResponsiveValue 
} from '@/utils/layout';

describe('Layout Utilities', () => {
  describe('calculateGridColumns', () => {
    it('returns correct column count for different viewports', () => {
      expect(calculateGridColumns(320)).toBe(1);  // Mobile
      expect(calculateGridColumns(768)).toBe(2);  // Tablet
      expect(calculateGridColumns(1200)).toBe(3); // Desktop
    });

    it('respects max columns setting', () => {
      expect(calculateGridColumns(1920, { maxCols: 4 })).toBe(4);
    });
  });

  describe('getSpacingValue', () => {
    it('converts spacing tokens to rem values', () => {
      expect(getSpacingValue('xs')).toBe('0.25rem');
      expect(getSpacingValue('sm')).toBe('0.5rem');
      expect(getSpacingValue('md')).toBe('1rem');
      expect(getSpacingValue('lg')).toBe('1.5rem');
      expect(getSpacingValue('xl')).toBe('2rem');
    });

    it('returns custom values unchanged', () => {
      expect(getSpacingValue('24px')).toBe('24px');
      expect(getSpacingValue('2em')).toBe('2em');
    });
  });

  describe('parseResponsiveValue', () => {
    it('handles single values', () => {
      expect(parseResponsiveValue('md')).toEqual({
        base: 'md',
        sm: 'md',
        md: 'md',
        lg: 'md',
        xl: 'md'
      });
    });

    it('handles responsive objects', () => {
      const input = { base: 'sm', md: 'lg' };
      expect(parseResponsiveValue(input)).toEqual({
        base: 'sm',
        sm: 'sm',
        md: 'lg',
        lg: 'lg',
        xl: 'lg'
      });
    });
  });
});
```

### 2. Integration Tests (30%)

#### Component Interactions
```typescript
// PlotWorkflow.integration.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { PlotWorkflow } from '@/components/PlotWorkflow';
import { mockServer } from '@/test/mocks/server';

describe('Plot Generation Workflow', () => {
  beforeAll(() => mockServer.listen());
  afterEach(() => mockServer.resetHandlers());
  afterAll(() => mockServer.close());

  it('completes full plot generation flow', async () => {
    const user = userEvent.setup();
    render(<PlotWorkflow />);

    // Step 1: Upload data
    const file = new File(
      ['x,y\n1,2\n3,4\n5,6'], 
      'data.csv',
      { type: 'text/csv' }
    );
    
    const input = screen.getByLabelText('Upload CSV File');
    await user.upload(input, file);
    
    await waitFor(() => {
      expect(screen.getByText('data.csv')).toBeInTheDocument();
      expect(screen.getByText('3 rows, 2 columns')).toBeInTheDocument();
    });

    // Step 2: Configure plot
    await user.click(screen.getByText('Next: Configure Plot'));
    
    await user.selectOptions(
      screen.getByLabelText('Plot Type'),
      'scatter'
    );
    
    await user.selectOptions(
      screen.getByLabelText('X Axis'),
      'x'
    );
    
    await user.selectOptions(
      screen.getByLabelText('Y Axis'),
      'y'
    );

    // Step 3: Generate plot
    await user.click(screen.getByText('Generate Plot'));
    
    // Wait for loading state
    expect(screen.getByText('Generating plot...')).toBeInTheDocument();
    
    // Wait for result
    await waitFor(() => {
      expect(screen.getByRole('img', { name: 'Generated plot' }))
        .toBeInTheDocument();
    });

    // Step 4: Export options
    expect(screen.getByText('Export as PNG')).toBeInTheDocument();
    expect(screen.getByText('Export as SVG')).toBeInTheDocument();
    expect(screen.getByText('Copy to Clipboard')).toBeInTheDocument();
  });

  it('handles errors gracefully', async () => {
    const user = userEvent.setup();
    
    // Mock server error
    mockServer.use(
      rest.post('/api/plots/generate', (req, res, ctx) => {
        return res(
          ctx.status(500),
          ctx.json({ error: 'Internal server error' })
        );
      })
    );
    
    render(<PlotWorkflow />);
    
    // Upload and configure...
    // (setup steps)
    
    await user.click(screen.getByText('Generate Plot'));
    
    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(
        'Failed to generate plot. Please try again.'
      );
    });
    
    // Can retry
    expect(screen.getByText('Retry')).toBeEnabled();
  });
});
```

#### API Integration
```typescript
// api.integration.test.ts
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { usePlotGeneration } from '@/hooks/usePlotGeneration';

describe('API Integration', () => {
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })}>
      {children}
    </QueryClientProvider>
  );

  it('generates plot with valid data', async () => {
    const { result } = renderHook(() => usePlotGeneration(), { wrapper });
    
    act(() => {
      result.current.generatePlot({
        data: [{ x: 1, y: 2 }, { x: 3, y: 4 }],
        config: { type: 'scatter', xAxis: 'x', yAxis: 'y' }
      });
    });
    
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });
    
    expect(result.current.data).toMatchObject({
      imageUrl: expect.stringContaining('http'),
      plotId: expect.any(String),
      metadata: expect.objectContaining({
        type: 'scatter',
        timestamp: expect.any(String)
      })
    });
  });

  it('handles validation errors', async () => {
    const { result } = renderHook(() => usePlotGeneration(), { wrapper });
    
    act(() => {
      result.current.generatePlot({
        data: [], // Empty data
        config: { type: 'scatter' }
      });
    });
    
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });
    
    expect(result.current.error).toMatchObject({
      message: 'Data cannot be empty',
      code: 'VALIDATION_ERROR'
    });
  });
});
```

### 3. Visual Regression Tests

#### Setup with Playwright
```typescript
// playwright.config.ts
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/visual',
  use: {
    baseURL: 'http://localhost:3000',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'] },
    },
  ],
});
```

#### Visual Test Examples
```typescript
// layout-visual.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Layout Visual Regression', () => {
  test('Stack component variations', async ({ page }) => {
    await page.goto('/storybook/stack');
    
    const variations = [
      'default',
      'horizontal-sm',
      'horizontal-md',
      'horizontal-lg',
      'vertical-sm',
      'vertical-md',
      'vertical-lg',
      'responsive-spacing',
      'with-dividers'
    ];
    
    for (const variation of variations) {
      await page.click(`[data-story="${variation}"]`);
      await page.waitForTimeout(100); // Wait for animation
      
      await expect(page.locator('.story-container')).toHaveScreenshot(
        `stack-${variation}.png`,
        { maxDiffPixels: 100 }
      );
    }
  });

  test('Grid responsive behavior', async ({ page }) => {
    await page.goto('/storybook/grid');
    
    // Test different viewport sizes
    const viewports = [
      { width: 375, height: 667, name: 'mobile' },
      { width: 768, height: 1024, name: 'tablet' },
      { width: 1440, height: 900, name: 'desktop' },
      { width: 1920, height: 1080, name: 'wide' }
    ];
    
    for (const viewport of viewports) {
      await page.setViewportSize(viewport);
      await page.waitForTimeout(100);
      
      await expect(page.locator('.grid-demo')).toHaveScreenshot(
        `grid-${viewport.name}.png`
      );
    }
  });

  test('Plot display states', async ({ page }) => {
    await page.goto('/demo/plot');
    
    // Loading state
    await page.click('[data-testid="generate-plot"]');
    await expect(page.locator('.plot-container')).toHaveScreenshot(
      'plot-loading.png'
    );
    
    // Success state
    await page.waitForSelector('.plot-image');
    await expect(page.locator('.plot-container')).toHaveScreenshot(
      'plot-success.png'
    );
    
    // Error state
    await page.click('[data-testid="trigger-error"]');
    await expect(page.locator('.plot-container')).toHaveScreenshot(
      'plot-error.png'
    );
  });

  test('Dark mode variations', async ({ page }) => {
    await page.goto('/');
    
    // Light mode
    await expect(page).toHaveScreenshot('app-light.png');
    
    // Dark mode
    await page.click('[data-testid="theme-toggle"]');
    await page.waitForTimeout(300); // Wait for transition
    await expect(page).toHaveScreenshot('app-dark.png');
  });
});
```

### 4. Performance Tests

#### Component Performance
```typescript
// performance.test.tsx
import { render } from '@testing-library/react';
import { measureRender } from '@/test/utils/performance';
import { Stack, Grid, Container } from '@/components/layout';

describe('Performance Benchmarks', () => {
  describe('Render Performance', () => {
    it('Stack renders within performance budget', () => {
      const result = measureRender(
        <Stack>
          {Array.from({ length: 100 }, (_, i) => (
            <div key={i}>Item {i}</div>
          ))}
        </Stack>
      );
      
      expect(result.renderTime).toBeLessThan(16); // 60fps
      expect(result.commitTime).toBeLessThan(10);
    });

    it('Grid handles large datasets efficiently', () => {
      const result = measureRender(
        <Grid cols={3}>
          {Array.from({ length: 1000 }, (_, i) => (
            <div key={i}>Cell {i}</div>
          ))}
        </Grid>
      );
      
      expect(result.renderTime).toBeLessThan(50);
      expect(result.updateTime).toBeLessThan(20);
    });
  });

  describe('Memory Usage', () => {
    it('does not leak memory on unmount', async () => {
      const initialMemory = performance.memory.usedJSHeapSize;
      
      // Mount and unmount component multiple times
      for (let i = 0; i < 100; i++) {
        const { unmount } = render(
          <Stack spacing="md">
            <div>Content</div>
          </Stack>
        );
        unmount();
      }
      
      // Force garbage collection (if available)
      if (global.gc) {
        global.gc();
      }
      
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const finalMemory = performance.memory.usedJSHeapSize;
      const memoryIncrease = finalMemory - initialMemory;
      
      // Allow for some memory increase but flag potential leaks
      expect(memoryIncrease).toBeLessThan(1000000); // 1MB threshold
    });
  });

  describe('Re-render Performance', () => {
    it('Stack avoids unnecessary re-renders', () => {
      let renderCount = 0;
      
      const TestComponent = () => {
        renderCount++;
        return <div>Child</div>;
      };
      
      const { rerender } = render(
        <Stack spacing="md" className="test">
          <TestComponent />
        </Stack>
      );
      
      expect(renderCount).toBe(1);
      
      // Re-render with same props
      rerender(
        <Stack spacing="md" className="test">
          <TestComponent />
        </Stack>
      );
      
      expect(renderCount).toBe(1); // Should not re-render child
      
      // Re-render with different props
      rerender(
        <Stack spacing="lg" className="test">
          <TestComponent />
        </Stack>
      );
      
      expect(renderCount).toBe(2); // Should re-render
    });
  });
});
```

#### Bundle Size Analysis
```typescript
// bundle-analysis.test.ts
import { analyzeBundleSize } from '@/test/utils/bundle';
import * as LayoutComponents from '@/components/layout';

describe('Bundle Size', () => {
  it('layout components stay within size budget', async () => {
    const analysis = await analyzeBundleSize(LayoutComponents);
    
    expect(analysis.Stack.gzipped).toBeLessThan(2000); // 2KB
    expect(analysis.Grid.gzipped).toBeLessThan(3000);  // 3KB
    expect(analysis.Container.gzipped).toBeLessThan(1500); // 1.5KB
    
    const totalSize = Object.values(analysis)
      .reduce((sum, component) => sum + component.gzipped, 0);
    
    expect(totalSize).toBeLessThan(20000); // 20KB total
  });

  it('tree shaking works correctly', async () => {
    // Import just Stack
    const { Stack } = await import('@/components/layout/Stack');
    const singleImport = await analyzeBundleSize({ Stack });
    
    // Import all components
    const allComponents = await import('@/components/layout');
    const allImport = await analyzeBundleSize(allComponents);
    
    // Verify tree shaking is working
    expect(singleImport.total).toBeLessThan(allImport.total * 0.3);
  });
});
```

### 5. Accessibility Tests

#### Component Accessibility
```typescript
// accessibility.test.tsx
import { render } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';
import userEvent from '@testing-library/user-event';

expect.extend(toHaveNoViolations);

describe('Accessibility', () => {
  describe('Layout Components', () => {
    it('Stack has no accessibility violations', async () => {
      const { container } = render(
        <Stack spacing="md">
          <button>Button 1</button>
          <button>Button 2</button>
          <button>Button 3</button>
        </Stack>
      );
      
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('Grid maintains semantic structure', async () => {
      const { container } = render(
        <Grid cols={2} role="list">
          <div role="listitem">Item 1</div>
          <div role="listitem">Item 2</div>
          <div role="listitem">Item 3</div>
          <div role="listitem">Item 4</div>
        </Grid>
      );
      
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });

  describe('Keyboard Navigation', () => {
    it('maintains focus order in Stack', async () => {
      const user = userEvent.setup();
      const focusOrder: string[] = [];
      
      render(
        <Stack>
          <button onFocus={() => focusOrder.push('1')}>Button 1</button>
          <button onFocus={() => focusOrder.push('2')}>Button 2</button>
          <button onFocus={() => focusOrder.push('3')}>Button 3</button>
        </Stack>
      );
      
      await user.tab();
      await user.tab();
      await user.tab();
      
      expect(focusOrder).toEqual(['1', '2', '3']);
    });

    it('supports arrow key navigation in Grid', async () => {
      const user = userEvent.setup();
      
      render(
        <Grid cols={3} role="grid">
          <button role="gridcell">1</button>
          <button role="gridcell">2</button>
          <button role="gridcell">3</button>
          <button role="gridcell">4</button>
          <button role="gridcell">5</button>
          <button role="gridcell">6</button>
        </Grid>
      );
      
      const firstButton = screen.getByText('1');
      await user.click(firstButton);
      
      // Right arrow
      await user.keyboard('{ArrowRight}');
      expect(document.activeElement).toHaveTextContent('2');
      
      // Down arrow
      await user.keyboard('{ArrowDown}');
      expect(document.activeElement).toHaveTextContent('5');
    });
  });

  describe('Screen Reader Support', () => {
    it('provides appropriate ARIA labels', () => {
      render(
        <Stack aria-label="Navigation menu">
          <a href="/home">Home</a>
          <a href="/about">About</a>
          <a href="/contact">Contact</a>
        </Stack>
      );
      
      expect(screen.getByLabelText('Navigation menu')).toBeInTheDocument();
    });

    it('announces dynamic content changes', async () => {
      const { rerender } = render(
        <div role="status" aria-live="polite">
          <Stack>
            <div>Initial content</div>
          </Stack>
        </div>
      );
      
      rerender(
        <div role="status" aria-live="polite">
          <Stack>
            <div>Updated content</div>
          </Stack>
        </div>
      );
      
      // Screen reader should announce the change
      expect(screen.getByText('Updated content')).toBeInTheDocument();
    });
  });
});
```

## Test Utilities

### Mock Utilities
```typescript
// test/utils/mocks.ts
export function createMatchMedia(width: number) {
  return (query: string) => ({
    matches: width >= parseInt(query.match(/\d+/)?.[0] || '0'),
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  });
}

export function mockIntersectionObserver() {
  const mockIntersectionObserver = jest.fn();
  mockIntersectionObserver.mockReturnValue({
    observe: () => null,
    unobserve: () => null,
    disconnect: () => null
  });
  window.IntersectionObserver = mockIntersectionObserver;
}

export function mockResizeObserver() {
  const mockResizeObserver = jest.fn();
  mockResizeObserver.mockReturnValue({
    observe: () => null,
    unobserve: () => null,
    disconnect: () => null
  });
  window.ResizeObserver = mockResizeObserver;
}
```

### Performance Utilities
```typescript
// test/utils/performance.ts
import { Profiler, ProfilerOnRenderCallback } from 'react';

interface RenderMetrics {
  renderTime: number;
  commitTime: number;
  updateTime?: number;
}

export function measureRender(component: React.ReactElement): RenderMetrics {
  let metrics: RenderMetrics = { renderTime: 0, commitTime: 0 };
  
  const onRender: ProfilerOnRenderCallback = (
    id,
    phase,
    actualDuration,
    baseDuration,
    startTime,
    commitTime
  ) => {
    metrics = {
      renderTime: actualDuration,
      commitTime: commitTime - startTime,
      updateTime: phase === 'update' ? actualDuration : undefined
    };
  };
  
  const { rerender } = render(
    <Profiler id="test" onRender={onRender}>
      {component}
    </Profiler>
  );
  
  // Force update to measure update performance
  rerender(
    <Profiler id="test" onRender={onRender}>
      {React.cloneElement(component, { key: 'updated' })}
    </Profiler>
  );
  
  return metrics;
}

export async function measureMemoryUsage(
  callback: () => void
): Promise<number> {
  if (!performance.memory) {
    throw new Error('Performance.memory API not available');
  }
  
  // Force garbage collection if available
  if (global.gc) {
    global.gc();
  }
  
  const before = performance.memory.usedJSHeapSize;
  callback();
  
  // Wait for async operations
  await new Promise(resolve => setTimeout(resolve, 100));
  
  const after = performance.memory.usedJSHeapSize;
  return after - before;
}
```

## CI/CD Integration

### GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run unit tests
        run: npm run test:unit -- --coverage
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/lcov.info

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Start test server
        run: npm run start:test &
        env:
          NODE_ENV: test
      
      - name: Wait for server
        run: npx wait-on http://localhost:3000
      
      - name: Run integration tests
        run: npm run test:integration

  visual-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Install Playwright
        run: npx playwright install --with-deps
      
      - name: Run visual tests
        run: npm run test:visual
      
      - name: Upload visual diff
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: visual-regression-diff
          path: tests/visual/__diff__/

  performance-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Build production bundle
        run: npm run build
      
      - name: Run performance tests
        run: npm run test:performance
      
      - name: Upload performance report
        uses: actions/upload-artifact@v3
        with:
          name: performance-report
          path: reports/performance/

  accessibility-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run accessibility tests
        run: npm run test:a11y
      
      - name: Run lighthouse audit
        run: npm run lighthouse
      
      - name: Upload lighthouse report
        uses: actions/upload-artifact@v3
        with:
          name: lighthouse-report
          path: reports/lighthouse/
```

## Test Scripts

### Package.json Scripts
```json
{
  "scripts": {
    "test": "jest",
    "test:unit": "jest --testPathPattern='.test.tsx?$'",
    "test:integration": "jest --testPathPattern='.integration.test.tsx?$'",
    "test:visual": "playwright test",
    "test:visual:update": "playwright test --update-snapshots",
    "test:performance": "jest --testPathPattern='.performance.test.tsx?$' --runInBand",
    "test:a11y": "jest --testPathPattern='.accessibility.test.tsx?$'",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage",
    "test:ci": "npm run test -- --ci --maxWorkers=2",
    "lighthouse": "lighthouse http://localhost:3000 --output=json --output-path=./reports/lighthouse/report.json"
  }
}
```

## Test Configuration

### Jest Configuration
```typescript
// jest.config.ts
export default {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/tests/setup.ts'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
  },
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  },
  coveragePathIgnorePatterns: [
    '/node_modules/',
    '/tests/',
    '.stories.tsx$'
  ],
  testMatch: [
    '**/__tests__/**/*.+(ts|tsx|js)',
    '**/?(*.)+(spec|test).+(ts|tsx|js)'
  ],
  transform: {
    '^.+\\.(ts|tsx)$': ['ts-jest', {
      tsconfig: {
        jsx: 'react'
      }
    }]
  }
};
```

### Test Setup
```typescript
// tests/setup.ts
import '@testing-library/jest-dom';
import { configure } from '@testing-library/react';
import { mockIntersectionObserver, mockResizeObserver } from './utils/mocks';

// Configure testing library
configure({ testIdAttribute: 'data-testid' });

// Mock browser APIs
mockIntersectionObserver();
mockResizeObserver();

// Mock next/image
jest.mock('next/image', () => ({
  __esModule: true,
  default: (props: any) => {
    return <img {...props} />;
  },
}));

// Suppress console errors in tests
const originalError = console.error;
beforeAll(() => {
  console.error = (...args: any[]) => {
    if (
      typeof args[0] === 'string' &&
      args[0].includes('Warning: ReactDOM.render')
    ) {
      return;
    }
    originalError.call(console, ...args);
  };
});

afterAll(() => {
  console.error = originalError;
});
```