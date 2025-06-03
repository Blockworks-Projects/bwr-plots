# BWR Plots Frontend Refactor Checklist

## Overview
This checklist provides a step-by-step breakdown of the frontend refactor plan. Each task should be completed in order within its phase, with some tasks able to run in parallel where noted.

---

## 🏗️ Phase 1: Backend API Development (Weeks 1-3)

### Week 1: Project Setup & Core Infrastructure

#### 1.1 Backend Project Structure
- [x] Create `backend/` directory in project root
- [x] Initialize Python virtual environment
- [x] Create `requirements.txt` with FastAPI dependencies
- [x] Set up basic FastAPI application in `main.py`
- [x] Create directory structure:
  - [x] `api/` with `__init__.py`
  - [x] `api/routes/` with `__init__.py`
  - [x] `core/` with `__init__.py`
  - [x] `models/` with `__init__.py`
  - [x] `services/` with `__init__.py`
  - [x] `utils/` with `__init__.py`
  - [x] `storage/uploads/` directory
  - [x] `storage/sessions/` directory
  - [x] `tests/` with `__init__.py`

#### 1.2 Core Configuration
- [x] Create `core/config.py` with environment variables
- [x] Set up CORS middleware in `api/middleware.py`
- [x] Create `core/exceptions.py` with custom exception handlers
- [x] Add logging configuration
- [x] Create health check endpoint in `api/routes/health.py`

#### 1.3 Session Management Foundation
- [x] Create `services/session_manager.py`
- [x] Implement in-memory session storage with Redis fallback for production
- [x] Add session ID generation utility
- [x] Create session cleanup mechanism
- [x] **NEW**: Configure session storage for serverless environment

### Week 2: Data Management APIs ✅ COMPLETED

#### 2.1 File Upload Service
- [x] Extract file loading logic from `app.py` into `services/file_handler.py`
- [x] Create `load_data()` function with CSV/XLSX support
- [x] Add file validation (type, size, format)
- [x] **MODIFIED**: Implement temporary file storage using `/tmp` directory (Vercel serverless)
- [x] Add file cleanup mechanism with proper serverless handling
- [x] **NEW**: Configure file size limits for Vercel (10MB function payload limit)

#### 2.2 Data Processing Service
- [x] Extract data manipulation logic from `app.py` into `services/data_processor.py`
- [x] Implement column dropping functionality
- [x] Implement column renaming functionality
- [x] Implement data pivoting functionality
- [x] Add data preview generation
- [x] Create column analysis utilities

#### 2.3 Data Management Endpoints
- [x] Create `models/requests.py` with Pydantic models:
  - [x] `FileUploadRequest`
  - [x] `DataManipulationRequest`
  - [x] `DataOperation`
- [x] Create `models/responses.py` with response models:
  - [x] `DataPreviewResponse`
  - [x] `ColumnInfo`
- [x] Implement `POST /api/v1/data/upload` endpoint
- [x] Implement `GET /api/v1/data/preview/{session_id}` endpoint
- [x] Implement `POST /api/v1/data/manipulate` endpoint
- [x] Add comprehensive error handling for all endpoints
- [x] **NEW**: Optimize endpoints for Vercel's 10-second function timeout

#### 2.4 Backend Testing & Integration
- [x] Install all required dependencies
- [x] Fix import errors and exception handling
- [x] Test backend server startup
- [x] Verify API endpoints are accessible

### Week 3: Plot Generation APIs

#### 3.1 Plot Generation Service
- [ ] Extract plot building logic from `app.py` into `services/plot_generator.py`
- [ ] Create BWRPlots wrapper service
- [ ] Implement plot configuration handling
- [ ] Add data processing pipeline (filtering, resampling, smoothing)
- [ ] Create plot export functionality
- [ ] **NEW**: Optimize plot generation for serverless execution time limits

#### 3.2 Plot Generation Models
- [ ] Create plot-related Pydantic models:
  - [ ] `PlotGenerationRequest`
  - [ ] `PlotConfiguration`
  - [ ] `DataProcessingConfig`
  - [ ] `FilterConfig`
  - [ ] `ResamplingConfig`
  - [ ] `SmoothingConfig`
  - [ ] `WatermarkConfig`
  - [ ] `AxisConfig`
  - [ ] `StylingConfig`
- [ ] Create response models:
  - [ ] `PlotGenerationResponse`
  - [ ] `PlotTypeInfo`

#### 3.3 Plot Generation Endpoints
- [ ] Implement `GET /api/v1/plots/types` endpoint
- [ ] Implement `POST /api/v1/plots/generate` endpoint
- [ ] Implement `POST /api/v1/plots/export` endpoint
- [ ] Add plot type validation
- [ ] Add configuration validation

#### 3.4 Configuration Endpoints
- [ ] Implement `GET /api/v1/config/watermarks` endpoint
- [ ] Implement `GET /api/v1/config/plot-defaults/{plot_type}` endpoint
- [ ] Extract watermark configuration from BWR Plots
- [ ] Create plot type default configurations

#### 3.5 Testing & Documentation
- [ ] Write unit tests for all services
- [ ] Write integration tests for all endpoints
- [ ] Test with sample data files
- [ ] Generate API documentation with FastAPI
- [ ] Create Postman collection for API testing

---

## 🎨 Phase 2: Frontend Development (Weeks 4-7)

### Week 4: Next.js Setup & Core Components

#### 4.1 Frontend Project Setup
- [x] Verify Next.js project structure in `frontend/`
- [ ] Install additional dependencies:
  - [ ] `@tanstack/react-query`
  - [ ] `plotly.js-dist-min`
  - [ ] `react-plotly.js`
  - [ ] `react-dropzone`
  - [ ] `zod` for validation
  - [ ] `clsx` for conditional classes
- [ ] Configure TypeScript strict mode
- [x] Set up Tailwind CSS configuration
- [x] Create global styles in `app/globals.css`
- [x] **NEW**: Configure Next.js for Vercel deployment in `next.config.js`

#### 4.2 Type Definitions
- [ ] Create `types/api.ts` with API response/request types
- [ ] Create `types/data.ts` with data-related types
- [ ] Create `types/plots.ts` with plot configuration types
- [ ] Create `types/ui.ts` with UI component types
- [ ] Ensure type safety across all interfaces

#### 4.3 API Client Setup
- [ ] Create `lib/api.ts` with API client configuration
- [ ] Set up React Query client in `lib/queryClient.ts`
- [ ] Create API endpoint functions
- [ ] Add error handling and retry logic
- [ ] **MODIFIED**: Configure API client for Vercel API routes instead of external FastAPI
- [ ] **NEW**: Set up API routes in `app/api/` directory for Vercel

#### 4.4 Basic UI Components
- [ ] Create `components/ui/Button.tsx`
- [ ] Create `components/ui/Input.tsx`
- [ ] Create `components/ui/Select.tsx`
- [ ] Create `components/ui/Card.tsx`
- [ ] Create `components/ui/LoadingSpinner.tsx`
- [ ] Create `components/ui/Tabs.tsx`
- [ ] Style components with Tailwind CSS

### Week 5: Data Management Components

#### 5.1 File Upload Component
- [ ] Create `components/data/FileUpload.tsx`
- [ ] Implement drag-and-drop functionality
- [ ] Add file validation (type, size)
- [ ] Create upload progress indicator
- [ ] Add error handling and user feedback
- [ ] Style with modern UI design
- [ ] **NEW**: Configure for Vercel's 4.5MB request limit

#### 5.2 Data Preview Component
- [ ] Create `components/data/DataPreview.tsx`
- [ ] Display data table with pagination
- [ ] Show column information and data types
- [ ] Add row count and basic statistics
- [ ] Implement responsive design

#### 5.3 Data Manipulation Components
- [ ] Create `components/data/DataManipulation.tsx`
- [ ] Implement column dropping interface
- [ ] Create column renaming form
- [ ] Build pivot configuration interface
- [ ] Add real-time preview updates
- [ ] Create `components/data/ColumnSelector.tsx`

#### 5.4 Custom Hooks for Data Management
- [ ] Create `hooks/useDataUpload.ts`
- [ ] Create `hooks/useDataProcessing.ts`
- [ ] Create `hooks/useSession.ts`
- [ ] Add proper error handling and loading states
- [ ] Implement optimistic updates where appropriate

### Week 6: Plot Configuration & Display

#### 6.1 Plot Type Selection
- [ ] Create `components/plotting/PlotTypeSelector.tsx`
- [ ] Fetch available plot types from API
- [ ] Display plot type cards with descriptions
- [ ] Add plot type validation

#### 6.2 Plot Configuration Component
- [ ] Create `components/plotting/PlotConfiguration.tsx`
- [ ] Build dynamic form based on plot type
- [ ] Implement conditional field rendering
- [ ] Add form validation with Zod
- [ ] Create collapsible advanced settings
- [ ] Add real-time configuration preview

#### 6.3 Plot Display Component
- [ ] Create `components/plotting/PlotDisplay.tsx`
- [ ] Integrate Plotly.js for chart rendering
- [ ] Implement responsive chart sizing
- [ ] Add loading states and error handling
- [ ] Create export controls

#### 6.4 Plot Generation Hook
- [ ] Create `hooks/usePlotGeneration.ts`
- [ ] Handle plot generation API calls
- [ ] Manage plot state and caching
- [ ] Add error handling and retry logic

### Week 7: Layout & Integration

#### 7.1 Layout Components
- [ ] Create `components/layout/Header.tsx`
- [ ] Create `components/layout/Sidebar.tsx`
- [ ] Create `components/layout/Footer.tsx`
- [ ] Implement responsive navigation
- [ ] Add branding and styling

#### 7.2 Form Components
- [ ] Create `components/forms/FormField.tsx`
- [ ] Create `components/forms/FormSection.tsx`
- [ ] Create `components/forms/FormValidation.tsx`
- [ ] Implement consistent form styling
- [ ] Add accessibility features

#### 7.3 Main Application Page
- [ ] Update `app/page.tsx` with main interface
- [ ] Implement step-by-step workflow
- [ ] Add state management between components
- [ ] Create responsive layout
- [ ] Add keyboard navigation support

#### 7.4 Additional Utilities
- [ ] Create `lib/utils.ts` with helper functions
- [ ] Create `lib/constants.ts` with application constants
- [ ] Create `lib/validators.ts` with form validation schemas
- [ ] Add `hooks/useLocalStorage.ts` for state persistence

#### 7.5 Vercel API Routes Setup
- [ ] **NEW**: Create `app/api/data/upload/route.ts`
- [ ] **NEW**: Create `app/api/data/preview/[sessionId]/route.ts`
- [ ] **NEW**: Create `app/api/data/manipulate/route.ts`
- [ ] **NEW**: Create `app/api/plots/generate/route.ts`
- [ ] **NEW**: Create `app/api/plots/export/route.ts`
- [ ] **NEW**: Create `app/api/config/watermarks/route.ts`
- [ ] **NEW**: Create `app/api/health/route.ts`

---

## 🔗 Phase 3: Integration & Testing (Weeks 8-9)

### Week 8: API Integration & Error Handling

#### 8.1 Frontend-Backend Integration
- [ ] Connect file upload component to Vercel API routes
- [ ] Connect data manipulation to API endpoints
- [ ] Connect plot generation to API routes
- [ ] Test all API integrations thoroughly
- [ ] Verify data flow between frontend and API routes
- [ ] **NEW**: Test serverless function cold starts and performance

#### 8.2 Error Handling & User Feedback
- [ ] Implement comprehensive error handling
- [ ] Add user-friendly error messages
- [ ] Create error boundary components
- [ ] Add loading states for all async operations
- [ ] Implement toast notifications for feedback
- [ ] **NEW**: Handle Vercel-specific errors (timeouts, payload limits)

#### 8.3 State Management Optimization
- [ ] Optimize React Query cache configuration
- [ ] Implement proper data invalidation
- [ ] Add optimistic updates where beneficial
- [ ] Test state persistence across page refreshes

#### 8.4 Performance Optimization
- [ ] Implement code splitting for large components
- [ ] Optimize bundle size
- [ ] Add lazy loading for heavy components
- [ ] Optimize API call patterns
- [ ] **NEW**: Optimize for Vercel Edge Runtime where applicable

### Week 9: Feature Parity & Testing

#### 9.1 Feature Parity Verification
- [ ] Test all 7 plot types from original Streamlit app
- [ ] Verify data manipulation features work correctly
- [ ] Test filtering, resampling, and smoothing
- [ ] Verify watermark and styling options
- [ ] Test export functionality

#### 9.2 Data Processing Testing
- [ ] Test with various CSV file formats
- [ ] Test with XLSX files
- [ ] **MODIFIED**: Test with datasets up to 4.5MB (Vercel limit)
- [ ] Test edge cases (empty files, malformed data)
- [ ] Verify date parsing and handling

#### 9.3 User Experience Testing
- [ ] Test responsive design on mobile devices
- [ ] Verify accessibility compliance (WCAG 2.1)
- [ ] Test keyboard navigation
- [ ] Test with screen readers
- [ ] Cross-browser compatibility testing

#### 9.4 Performance Testing
- [ ] **MODIFIED**: Test serverless function performance
- [ ] API response time measurement
- [ ] Frontend rendering performance
- [ ] Memory usage monitoring
- [ ] File upload performance testing
- [ ] **NEW**: Test function cold start times

---

## 🚀 Phase 4: Deployment & Production (Weeks 10-11)

### Week 10: Vercel Deployment Setup

#### 10.1 Vercel Configuration
- [ ] **NEW**: Create `vercel.json` configuration file
- [ ] **NEW**: Configure build settings for Next.js
- [ ] **NEW**: Set up environment variables in Vercel dashboard
- [ ] **NEW**: Configure function regions and runtime settings
- [ ] **NEW**: Set up custom domains if needed

#### 10.2 Environment Configuration
- [ ] Set up environment variables for API routes
- [ ] Configure frontend environment variables
- [ ] Create `.env.example` files
- [ ] Set up different configs for dev/staging/prod
- [ ] Implement configuration validation
- [ ] **NEW**: Configure Vercel-specific environment variables

#### 10.3 Production Optimizations
- [ ] Configure production build optimizations
- [ ] **MODIFIED**: Optimize for Vercel Edge Network
- [ ] **NEW**: Configure Vercel Analytics and Speed Insights
- [ ] Implement security headers via `next.config.js`
- [ ] **NEW**: Set up Vercel Web Analytics

#### 10.4 Monitoring & Logging
- [ ] Set up application logging for serverless functions
- [ ] **NEW**: Configure Vercel Function Logs
- [ ] **NEW**: Set up Vercel Analytics
- [ ] Implement health check endpoints
- [ ] **NEW**: Set up Vercel monitoring dashboards

### Week 11: Final Testing & Go-Live

#### 11.1 Production Testing
- [ ] **MODIFIED**: Deploy to Vercel preview environment
- [ ] Run full end-to-end tests
- [ ] Performance testing in Vercel environment
- [ ] Security testing and vulnerability scanning
- [ ] **NEW**: Test serverless function scaling

#### 11.2 Documentation & Training
- [ ] Create user documentation
- [ ] **MODIFIED**: Write Vercel deployment documentation
- [ ] Create troubleshooting guide
- [ ] Document API endpoints
- [ ] Prepare user training materials

#### 11.3 Migration Preparation
- [ ] Create migration checklist
- [ ] Set up rollback procedures using Vercel deployments
- [ ] Prepare user communication
- [ ] **MODIFIED**: Use Vercel's instant rollback feature
- [ ] Create backup procedures

#### 11.4 Go-Live Activities
- [ ] Deploy to production on Vercel
- [ ] Monitor application performance
- [ ] Verify all features work correctly
- [ ] Monitor error rates and user feedback
- [ ] Execute rollback plan if needed using Vercel

---

## 📋 Post-Launch Activities

### Immediate (Week 12)
- [ ] Monitor application stability via Vercel dashboard
- [ ] Collect user feedback
- [ ] Fix any critical issues
- [ ] Performance optimization based on real usage
- [ ] Documentation updates

### Short-term (Weeks 13-16)
- [ ] Implement user-requested features
- [ ] Performance optimizations
- [ ] UI/UX improvements based on feedback
- [ ] Additional testing and bug fixes
- [ ] Security updates and patches

---

## 🎯 Success Criteria Checklist

### Technical Metrics
- [ ] API response times < 5 seconds (adjusted for serverless)
- [ ] Frontend load times < 3 seconds
- [ ] 99.9% uptime achieved via Vercel
- [ ] Zero data loss incidents
- [ ] All existing features functional

### User Experience Metrics
- [ ] User satisfaction score > 8/10
- [ ] Task completion rate > 95%
- [ ] Error rate < 1%
- [ ] Mobile usability score > 80%

### Performance Metrics
- [ ] **MODIFIED**: Support for concurrent serverless executions
- [ ] Handle files up to 4.5MB (Vercel limit)
- [ ] Generate plots in < 10 seconds (serverless timeout)
- [ ] **NEW**: Function cold start times < 2 seconds

---

## 📝 Notes & Tips

### Vercel-Specific Considerations
- **File Size Limits**: Vercel has a 4.5MB request body limit and 10MB function payload limit
- **Function Timeouts**: 10 seconds for Hobby plan, 60 seconds for Pro plan
- **Cold Starts**: Plan for serverless function cold start delays
- **Edge Runtime**: Consider using Edge Runtime for faster cold starts where applicable
- **Environment Variables**: Use Vercel dashboard for secure environment variable management

### Parallel Work Streams
- Frontend development and API route development can run in parallel
- Testing can begin as soon as individual components are complete
- Documentation should be written alongside development

### Risk Mitigation
- Keep the original Streamlit app running until full migration is complete
- Test each feature thoroughly before moving to the next
- Use Vercel's preview deployments for testing
- Regular stakeholder check-ins to ensure requirements are met

### Quality Assurance
- Code reviews for all major changes
- Automated testing where possible
- Manual testing for user experience
- Performance monitoring throughout development

This checklist should be updated as work progresses and new requirements emerge. 