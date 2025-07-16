# BWR Tools Layout System Architecture

## Overview

The BWR Tools layout system provides a set of composable primitives for building responsive, performant application interfaces. It emphasizes:

- **Performance**: Virtualized scrolling, optimized resize handling
- **Flexibility**: Composable primitives that work together
- **Developer Experience**: Type-safe, predictable APIs
- **Accessibility**: ARIA compliant, keyboard navigable

## Core Architecture

### 1. ViewportProvider (Context Layer)

Manages global viewport state and provides window dimension tracking.

```typescript
interface ViewportState {
  width: number;
  height: number;
  scrollY: number;
  scrollX: number;
  breakpoint: 'sm' | 'md' | 'lg' | 'xl' | '2xl';
  isDesktop: boolean;
  isMobile: boolean;
  isTablet: boolean;
}
```

### 2. AppShell (Main Container)

The root layout container that provides the basic structure.

```typescript
interface AppShellProps {
  header?: ReactNode;
  sidebar?: ReactNode;
  main: ReactNode;
  footer?: ReactNode;
  sidebarPosition?: 'left' | 'right';
  sidebarCollapsed?: boolean;
  headerHeight?: number;
  footerHeight?: number;
}
```

### 3. FlexLayout (Region Manager)

Flexible layout system for creating resizable regions.

```typescript
interface FlexLayoutProps {
  orientation: 'horizontal' | 'vertical';
  children: ReactElement<FlexItemProps>[];
  gap?: number;
  resizable?: boolean;
  onResize?: (sizes: number[]) => void;
}

interface FlexItemProps {
  size?: number | string;
  minSize?: number | string;
  maxSize?: number | string;
  resizable?: boolean;
  children: ReactNode;
}
```

### 4. ScrollArea (Virtualized Container)

Performance-optimized scrollable container with virtualization support.

```typescript
interface ScrollAreaProps {
  children: ReactNode;
  height?: number | string;
  width?: number | string;
  scrollbarWidth?: 'thin' | 'none' | 'auto';
  onScroll?: (event: ScrollEvent) => void;
  virtualizer?: VirtualizerOptions;
}
```

### 5. Panel (Content Region)

Base container for content with optional header and actions.

```typescript
interface PanelProps {
  title?: string;
  actions?: ReactNode;
  children: ReactNode;
  padding?: boolean | number;
  bordered?: boolean;
  elevated?: boolean;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
}
```

### 6. Grid (Responsive Grid)

CSS Grid-based layout system with responsive breakpoints.

```typescript
interface GridProps {
  columns?: number | ResponsiveValue<number>;
  rows?: number | ResponsiveValue<number>;
  gap?: number | ResponsiveValue<number>;
  areas?: string[] | ResponsiveValue<string[]>;
  children: ReactNode;
}

type ResponsiveValue<T> = T | {
  sm?: T;
  md?: T;
  lg?: T;
  xl?: T;
  '2xl'?: T;
}
```

## File Structure

```
src/components/layout/
├── README.md                    # This file
├── index.ts                     # Public exports
├── primitives/
│   ├── ViewportProvider.tsx     # Viewport context & provider
│   ├── AppShell.tsx            # Main application shell
│   ├── FlexLayout.tsx          # Flexible layout manager
│   ├── ScrollArea.tsx          # Virtualized scroll container
│   ├── Panel.tsx               # Base panel component
│   └── Grid.tsx                # Responsive grid system
├── hooks/
│   ├── useViewport.ts          # Viewport state hook
│   ├── useResizeObserver.ts    # Resize observer hook
│   ├── useScrollPosition.ts    # Scroll position tracking
│   └── useBreakpoint.ts        # Responsive breakpoint hook
├── utils/
│   ├── viewport.ts             # Viewport utilities
│   ├── responsive.ts           # Responsive helpers
│   └── performance.ts          # Performance utilities
├── styles/
│   ├── layout.module.css       # Layout-specific styles
│   └── variables.css           # CSS custom properties
└── types/
    └── index.ts                # TypeScript definitions
```

## CSS Architecture

### Decision: CSS Modules + CSS Custom Properties

We use CSS Modules for component isolation with CSS Custom Properties for theming:

1. **CSS Modules**: Component-scoped styles prevent conflicts
2. **CSS Custom Properties**: Dynamic theming without JavaScript
3. **Tailwind Utilities**: For rapid prototyping and common patterns

### Layout CSS Variables

```css
:root {
  /* Spacing Scale */
  --layout-spacing-xs: 0.25rem;
  --layout-spacing-sm: 0.5rem;
  --layout-spacing-md: 1rem;
  --layout-spacing-lg: 1.5rem;
  --layout-spacing-xl: 2rem;
  
  /* Container Sizes */
  --layout-container-sm: 640px;
  --layout-container-md: 768px;
  --layout-container-lg: 1024px;
  --layout-container-xl: 1280px;
  --layout-container-2xl: 1536px;
  
  /* Z-Index Scale */
  --layout-z-base: 0;
  --layout-z-dropdown: 1000;
  --layout-z-sticky: 1020;
  --layout-z-fixed: 1030;
  --layout-z-modal-backdrop: 1040;
  --layout-z-modal: 1050;
  --layout-z-popover: 1060;
  --layout-z-tooltip: 1070;
}
```

## Common Layout Patterns

### Three-Column Dashboard
```tsx
<AppShell>
  <FlexLayout orientation="horizontal">
    <FlexItem size="320px" minSize="280px" maxSize="400px">
      <Panel title="Data">
        <ScrollArea height="100%">
          {/* Data content */}
        </ScrollArea>
      </Panel>
    </FlexItem>
    
    <FlexItem size="1fr">
      <Panel title="Visualization">
        {/* Main content */}
      </Panel>
    </FlexItem>
    
    <FlexItem size="384px" minSize="320px" maxSize="480px">
      <Panel title="Configuration">
        <ScrollArea height="100%">
          {/* Config content */}
        </ScrollArea>
      </Panel>
    </FlexItem>
  </FlexLayout>
</AppShell>
```

### Single Column with Header
```tsx
<AppShell header={<Header />}>
  <Grid columns={1} gap={16}>
    <Panel title="Content">
      {/* Main content */}
    </Panel>
  </Grid>
</AppShell>
```

### Full-Screen Modal
```tsx
<AppShell>
  <div className="relative h-full">
    {/* Main content */}
    <Modal fullScreen>
      {/* Modal content */}
    </Modal>
  </div>
</AppShell>
```

### Sidebar Navigation
```tsx
<AppShell 
  sidebar={<NavigationSidebar />}
  sidebarPosition="left"
  sidebarCollapsed={isCollapsed}
>
  <ScrollArea height="100%">
    {/* Page content */}
  </ScrollArea>
</AppShell>
```

## Performance Optimizations

1. **ResizeObserver**: Efficient element size tracking
2. **RAF Throttling**: Smooth animations and updates
3. **Virtualization**: Large list rendering optimization
4. **Passive Listeners**: Non-blocking scroll events
5. **CSS Containment**: Layout performance hints

## Accessibility

- ARIA landmarks for all major regions
- Keyboard navigation support
- Focus management for modals and panels
- Screen reader announcements
- Reduced motion support