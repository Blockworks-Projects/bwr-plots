# BWR-Plots Frontend Layout Analysis

## Component Hierarchy & Scrolling Chain

```
┌─────────────────────────────────────────────────────────────────────────┐
│ html/body                                                               │
│ └── RootLayout (layout.tsx)                                            │
│     └── (tools)/layout.tsx - h-screen flex flex-col                   │
│         ├── PlatformHeader - h-14 flex-shrink-0                       │
│         └── main - flex-1 overflow-hidden ⚠️                          │
│             └── plots/page.tsx                                         │
│                 └── DashboardLayout - h-full w-full flex              │
│                     ├── DataPanel (Left) - w-80 h-full               │
│                     │   ├── Header - px-4 py-3 border-b              │
│                     │   └── Content - flex-1 overflow-y-auto ✓       │
│                     │                                                  │
│                     ├── PlotCanvas (Middle) - flex-1 h-full flex     │
│                     │   ├── Header - px-4 py-3 border-b              │
│                     │   └── Content - flex-1 overflow-y-auto ✓       │
│                     │                                                  │
│                     └── ConfigPanel (Right) - w-96 h-full            │
│                         ├── Header - px-4 py-3 border-b              │
│                         └── Content - flex-1 overflow-y-auto ✓       │
└─────────────────────────────────────────────────────────────────────────┘
```

## Critical Issues Identified

### 1. **Broken Height Chain at `main` Element**
- Location: `(tools)/layout.tsx` line 22
- Issue: `overflow-hidden` on main prevents scrolling
- The main element has `flex-1` (which grows to fill available space) but `overflow-hidden` which clips content

### 2. **Missing Height Constraints**
- The `h-screen` on the tools layout creates a 100vh container
- PlatformHeader takes fixed `h-14` (3.5rem)
- Main should take remaining space but overflow-hidden prevents proper content flow

### 3. **Panel Scrolling Implementation**
- Each panel (Data, Plot, Config) correctly implements:
  - `h-full` on container
  - `flex flex-col` structure
  - Fixed header with `border-b`
  - `flex-1 overflow-y-auto` on content area
- But the parent container's overflow-hidden prevents this from working

## Height Flow Analysis

```
100vh (h-screen on tools layout)
├── 56px (h-14 PlatformHeader)
└── calc(100vh - 56px) (flex-1 main) ⚠️ overflow-hidden
    └── 100% (h-full DashboardLayout)
        └── 100% (h-full each panel)
            ├── ~44px (header)
            └── calc(100% - 44px) (flex-1 content) ✓ overflow-y-auto
```

## Problematic CSS Classes

1. **overflow-hidden Issues**
   - `(tools)/layout.tsx:22` - main element
   - Prevents any scrolling within the main content area

2. **Height Calculation Issues**
   - No explicit height on panels' scrollable areas
   - Relying on flexbox without proper constraints

3. **Nested Flex Containers**
   - Multiple levels of `flex flex-col` without explicit heights
   - Can cause height calculation issues in some browsers

## Visual Layout Breakdown

```
┌─────────────────────────────────────────────────────────────┐
│                    PlatformHeader (56px)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┬──────────────────────┬──────────────────┐    │
│  │  Data   │      PlotCanvas      │  Configuration  │    │ <- overflow-hidden
│  │  Panel  │                      │     Panel       │    │    clips all content
│  │         │                      │                  │    │
│  │ scroll  │      scroll          │     scroll      │    │
│  │ broken  │      broken          │     broken      │    │
│  │         │                      │                  │    │
│  └─────────┴──────────────────────┴──────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Component-Specific Issues

### PlatformHeader (h-14 flex-shrink-0)
- Correctly sized and positioned
- No issues with this component

### DashboardLayout
- Uses `h-full` which should work
- Three-column flex layout is correct
- Each column has proper structure

### Panel Components (Data, Plot, Config)
- All follow same pattern correctly
- Have scrollable content areas
- But parent's overflow-hidden breaks them

## Overflow Property Chain

```
body: default (visible)
  └── tools layout: h-screen
      └── main: overflow-hidden ⚠️ <- BREAKS HERE
          └── DashboardLayout: default
              └── Panel divs: overflow-y-auto ✓
```

## Recommended Fix

The primary issue is the `overflow-hidden` on the main element in `(tools)/layout.tsx`. This should be removed or changed to allow proper content flow. The panels already have their own overflow management.

```tsx
// Current (broken):
<main className="flex-1 overflow-hidden">

// Should be:
<main className="flex-1 min-h-0">
// or
<main className="flex-1">
```

The `min-h-0` might be needed to properly constrain flex children in some cases.