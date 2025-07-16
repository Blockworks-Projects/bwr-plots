# BWR-Plots Scrolling Fix

## The Problem

The scrolling is broken because of `overflow-hidden` on the main element in `(tools)/layout.tsx`:

```tsx
// Line 22 in (tools)/layout.tsx
<main className="flex-1 overflow-hidden">  // ← THIS IS THE PROBLEM
  {children}
</main>
```

## Why It Breaks

1. **Height Constraint Chain**:
   - `h-screen` (100vh) on parent
   - `flex-1` on main (takes remaining space after header)
   - `overflow-hidden` STOPS any content from scrolling

2. **Child Panels Can't Scroll**:
   - Even though panels have `overflow-y-auto`
   - Parent's `overflow-hidden` clips all content
   - Scroll events have nowhere to bubble up

## The Solution

### Option 1: Remove overflow-hidden (Simplest)
```tsx
// (tools)/layout.tsx
<main className="flex-1">
  {children}
</main>
```

### Option 2: Add min-height constraint (Most Robust)
```tsx
// (tools)/layout.tsx
<main className="flex-1 min-h-0">
  {children}
</main>
```

### Option 3: Make main scrollable (Alternative)
```tsx
// (tools)/layout.tsx
<main className="flex-1 overflow-auto">
  {children}
</main>
```

## Visual Representation

### Current (Broken):
```
┌─────────────────────────────────┐
│         h-screen (100vh)        │
├─────────────────────────────────┤
│      PlatformHeader (56px)      │
├─────────────────────────────────┤
│                                 │
│   main (flex-1, overflow-hidden)│ ← Clips everything!
│   ┌───────────────────────────┐ │
│   │                           │ │
│   │   DashboardLayout         │ │
│   │   - Content below fold    │ │ ← Can't scroll to this
│   │   - Is completely hidden  │ │
│   │                           │ │
│   └───────────────────────────┘ │
└─────────────────────────────────┘
```

### Fixed:
```
┌─────────────────────────────────┐
│         h-screen (100vh)        │
├─────────────────────────────────┤
│      PlatformHeader (56px)      │
├─────────────────────────────────┤
│                                 │
│   main (flex-1, min-h-0)        │ ← Allows children to scroll
│   ┌───────────────────────────┐ │
│   │ ┌───┬─────────────┬─────┐ │ │
│   │ │ D │   Plot      │  C  │ │ │
│   │ │ a │   Canvas    │  o  │ │ │
│   │ │ t │             │  n  │ │ │
│   │ │ a │ ↕ scrolls   │  f  │ │ │
│   │ │ ↕ │             │  i  │ │ │
│   │ │   │             │  g  │ │ │
│   │ └───┴─────────────┴─────┘ │ │
│   └───────────────────────────┘ │
└─────────────────────────────────┘
```

## Implementation Steps

1. **Edit `(tools)/layout.tsx`**:
   ```tsx
   // Change line 22 from:
   <main className="flex-1 overflow-hidden">
   
   // To:
   <main className="flex-1 min-h-0">
   ```

2. **Verify each panel scrolls independently**:
   - Data Panel (left) - ✓ has `overflow-y-auto`
   - Plot Canvas (center) - ✓ has `overflow-y-auto`
   - Config Panel (right) - ✓ has `overflow-y-auto`

3. **Test with content that exceeds viewport height**

## Why min-h-0?

In flexbox, `min-height: auto` is the default, which can prevent flex items from shrinking smaller than their content. Setting `min-h-0` allows the flex item to shrink properly and enables child scrolling.

## Alternative Fixes (Not Recommended)

### Making DashboardLayout scroll instead:
```tsx
// Don't do this - it scrolls all panels together
<div className="h-full w-full flex overflow-auto">
```

### Adding fixed heights:
```tsx
// Don't do this - not responsive
<main className="flex-1" style={{ height: 'calc(100vh - 56px)' }}>
```

## Testing Checklist

- [ ] Each panel scrolls independently
- [ ] Scrollbars appear only when content overflows
- [ ] No double scrollbars
- [ ] Works on different viewport sizes
- [ ] No content is clipped or hidden
- [ ] Panels maintain their layout structure