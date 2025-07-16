# Layout System Migration Strategy

## Overview

This document outlines the migration strategy from the current fixed layout system to a new flexible, composable layout architecture for the BWR Plots application.

## Current Architecture Analysis

### Current Component Hierarchy
```
App
├── (tools)/layout.tsx (PlatformHeader + children)
│   └── (tools)/plots/page.tsx
│       └── DashboardLayout
│           ├── DataPanel (w-80 fixed)
│           ├── PlotCanvas (flex-1)
│           └── ConfigurationPanel (w-96 fixed)
```

### Current Issues
1. **Fixed widths**: Panels use hardcoded widths (w-80, w-96)
2. **Rigid structure**: DashboardLayout enforces 3-column layout
3. **Limited responsiveness**: No mobile support
4. **Scroll handling**: Each panel manages its own scrolling
5. **No persistence**: Panel states/sizes not saved
6. **Component coupling**: Panels tightly coupled to dashboard

## New Layout System Architecture

### Core Components
```typescript
// Layout primitives
<LayoutContainer />      // Root container with resize observer
<FlexLayout />          // Flexible row/column layouts
<GridLayout />          // CSS Grid-based layouts
<SplitLayout />         // Resizable split panes
<StackLayout />         // Stacked/tabbed layouts

// Layout panels
<Panel />               // Base panel component
<ResizablePanel />      // Panel with resize handles
<CollapsiblePanel />    // Panel with collapse/expand
<DraggablePanel />      // Moveable panel (future)

// Layout utilities
<ScrollArea />          // Managed scrolling
<LayoutProvider />      // Context for layout state
<LayoutPersistence />   // Save/restore layouts
```

## Migration Phases

### Phase 1: Foundation (Coexistence)
**Goal**: Build new system alongside existing components

1. Create layout primitives in `src/components/layout/v2/`
2. Implement `LayoutProvider` for state management
3. Build adapter components for backward compatibility
4. Add feature flags for gradual rollout

### Phase 2: Component Migration
**Goal**: Migrate leaf components to new system

1. Migrate UI components (Button, Card, Input)
2. Update data components (FileUpload, DataPreview)
3. Convert plotting components
4. Refactor form components

### Phase 3: Panel Migration
**Goal**: Replace major panels with flexible versions

1. DataPanel → FlexibleDataPanel
2. PlotCanvas → FlexiblePlotCanvas
3. ConfigurationPanel → FlexibleConfigPanel
4. DashboardLayout → ComposableLayout

### Phase 4: Feature Enhancement
**Goal**: Add new capabilities

1. Implement drag-and-drop panel rearrangement
2. Add responsive breakpoints
3. Enable layout persistence
4. Support custom layouts

### Phase 5: Cleanup
**Goal**: Remove old system

1. Remove legacy components
2. Update documentation
3. Migrate tests
4. Performance optimization

## Detailed Migration Plan

### 1. Component Dependency Analysis

#### No Dependencies (Easy Migration)
- LoadingSpinner
- Button
- Input
- Select
- Card

#### Layout Dependencies (Medium Complexity)
- FileUpload (uses Card)
- DataPreview (uses Card)
- PlotDisplay (needs scroll management)
- FormField (uses Input)
- FormSection (uses Card)

#### Heavy Dependencies (Complex Migration)
- DataPanel (uses multiple components)
- PlotCanvas (scroll, zoom, fullscreen)
- ConfigurationPanel (complex state)
- DashboardLayout (orchestrates everything)

### 2. Adapter Components

```typescript
// Adapter to use new Panel in old DashboardLayout
export function LegacyPanelAdapter({ children, ...props }) {
  return (
    <Panel
      className="h-full"
      scrollable
      legacy
      {...props}
    >
      {children}
    </Panel>
  );
}

// Adapter to use old components in new layout
export function NewLayoutAdapter({ type, children }) {
  return (
    <FlexLayout>
      <ResizablePanel defaultSize={type === 'data' ? 320 : 384}>
        {children}
      </ResizablePanel>
    </FlexLayout>
  );
}
```

### 3. Migration Examples

#### Before: DataPanel
```tsx
export function DataPanel({ className = '' }: DataPanelProps) {
  return (
    <div className={`h-full overflow-y-auto p-4 space-y-4 ${className}`}>
      {/* Content */}
    </div>
  );
}
```

#### After: FlexibleDataPanel
```tsx
export function FlexibleDataPanel({ className = '' }: DataPanelProps) {
  return (
    <Panel
      id="data-panel"
      title="Data"
      icon="📁"
      resizable
      collapsible
      defaultWidth={320}
      minWidth={240}
      maxWidth={480}
      className={className}
    >
      <ScrollArea>
        <div className="p-4 space-y-4">
          {/* Same content */}
        </div>
      </ScrollArea>
    </Panel>
  );
}
```

#### Before: DashboardLayout
```tsx
export function DashboardLayout({ dataPanel, plotCanvas, configPanel }) {
  return (
    <div className="h-full w-full flex">
      <div className="w-80">{dataPanel}</div>
      <div className="flex-1">{plotCanvas}</div>
      <div className="w-96">{configPanel}</div>
    </div>
  );
}
```

#### After: ComposableLayout
```tsx
export function ComposableLayout({ panels }) {
  const { layout, updateLayout } = useLayout();
  
  return (
    <LayoutProvider value={{ layout, updateLayout }}>
      <SplitLayout
        orientation="horizontal"
        sizes={layout.sizes}
        onSizesChange={(sizes) => updateLayout({ sizes })}
      >
        {panels.map(panel => (
          <ResizablePanel key={panel.id} {...panel} />
        ))}
      </SplitLayout>
    </LayoutProvider>
  );
}
```

### 4. Testing Strategy

#### Unit Tests
```typescript
describe('Panel Migration', () => {
  it('should maintain scroll position during resize', () => {
    // Test scroll persistence
  });
  
  it('should restore layout from localStorage', () => {
    // Test layout persistence
  });
  
  it('should handle legacy props', () => {
    // Test backward compatibility
  });
});
```

#### Integration Tests
```typescript
describe('Layout System', () => {
  it('should allow panel resizing', () => {
    // Test resize functionality
  });
  
  it('should collapse and expand panels', () => {
    // Test collapse/expand
  });
  
  it('should adapt to viewport changes', () => {
    // Test responsiveness
  });
});
```

### 5. Migration Challenges & Solutions

#### Challenge: Scroll Position Preservation
**Solution**: Implement ScrollContext that tracks positions
```typescript
const ScrollContext = createContext<{
  positions: Map<string, number>;
  savePosition: (id: string, position: number) => void;
  restorePosition: (id: string) => number;
}>();
```

#### Challenge: State Management During Migration
**Solution**: Dual state management with sync
```typescript
// Bridge between old and new state
function useLayoutBridge() {
  const oldSession = useSession();
  const newLayout = useLayout();
  
  // Sync states
  useEffect(() => {
    newLayout.update(mapOldToNew(oldSession));
  }, [oldSession]);
}
```

#### Challenge: Maintaining Tools Layout
**Solution**: Progressive enhancement
```typescript
// Start with existing layout
<ToolsLayout>
  {useNewLayout ? (
    <ComposableLayout>{children}</ComposableLayout>
  ) : (
    <DashboardLayout>{children}</DashboardLayout>
  )}
</ToolsLayout>
```

### 6. Component Migration Checklist

#### Phase 1 Checklist
- [ ] Create `src/components/layout/v2/` directory
- [ ] Implement LayoutContainer
- [ ] Implement FlexLayout
- [ ] Implement Panel base component
- [ ] Implement ScrollArea
- [ ] Create LayoutProvider
- [ ] Add feature flags
- [ ] Create adapter components

#### Phase 2 Checklist
- [ ] Migrate LoadingSpinner
- [ ] Migrate Button
- [ ] Migrate Input
- [ ] Migrate Select
- [ ] Migrate Card
- [ ] Update FileUpload
- [ ] Update DataPreview
- [ ] Update PlotDisplay

#### Phase 3 Checklist
- [ ] Create FlexibleDataPanel
- [ ] Create FlexiblePlotCanvas
- [ ] Create FlexibleConfigPanel
- [ ] Implement ResizablePanel
- [ ] Implement CollapsiblePanel
- [ ] Add layout persistence
- [ ] Update page.tsx to use new layout

#### Phase 4 Checklist
- [ ] Add drag-and-drop support
- [ ] Implement responsive breakpoints
- [ ] Create layout presets
- [ ] Add layout import/export
- [ ] Implement panel docking

#### Phase 5 Checklist
- [ ] Remove old DashboardLayout
- [ ] Remove old panel components
- [ ] Update all imports
- [ ] Clean up unused CSS
- [ ] Update documentation
- [ ] Performance audit

### 7. Success Metrics

#### Functionality Metrics
- All existing features work without regression
- Panel resizing works smoothly
- Scroll positions maintained
- Layout persists across sessions
- Mobile responsive design works

#### Performance Metrics
- Initial render time ≤ current
- Resize performance > 30 FPS
- Memory usage stable
- No layout thrashing

#### Developer Experience
- Clear migration path
- Good documentation
- Helpful error messages
- Easy to extend

#### User Experience
- Intuitive resize handles
- Smooth animations
- Consistent behavior
- No surprises

## Implementation Timeline

### Week 1-2: Foundation
- Build core layout components
- Implement state management
- Create adapters

### Week 3-4: Component Migration
- Migrate UI components
- Update data components
- Test compatibility

### Week 5-6: Panel Migration
- Migrate major panels
- Implement resize/collapse
- Add persistence

### Week 7-8: Enhancement & Cleanup
- Add advanced features
- Remove old code
- Documentation
- Performance optimization

## Risk Mitigation

1. **Feature Flags**: Roll out gradually
2. **Adapter Pattern**: Maintain compatibility
3. **Comprehensive Testing**: Catch regressions early
4. **Incremental Migration**: Small, safe steps
5. **Rollback Plan**: Can revert at any phase

## Conclusion

This migration strategy provides a clear path from the current rigid layout system to a flexible, modern architecture. By following this plan, we can improve the application while maintaining stability and backward compatibility throughout the process.