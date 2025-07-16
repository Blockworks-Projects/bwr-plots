# Backend Integration Tests

This directory contains comprehensive integration tests to verify that **Phase 1 (Backend API Development)** of the BWR Plots frontend refactor is complete and working correctly before proceeding to **Phase 2 (Frontend Development)**.

## Purpose

These tests ensure that:
- ✅ All backend API endpoints are functional
- ✅ File upload and data processing works correctly
- ✅ Session management is working
- ✅ Plot generation endpoints are operational
- ✅ Error handling is properly implemented
- ✅ Basic performance requirements are met

## Test Files

### `test_backend_integration.py`
Comprehensive integration test suite that covers:
- Health check endpoints
- File upload (CSV/Excel) functionality
- Data preview and manipulation
- Plot generation and configuration
- Session management
- Error handling
- Basic performance testing

### `run_backend_tests.py`
Simple test runner script that:
- Checks if backend server is running
- Installs required dependencies
- Runs the integration tests
- Provides clear pass/fail results

## How to Run the Tests

### Prerequisites

1. **Start the Backend Server**
   ```bash
   cd backend
   python main.py
   ```
   The server should be running on `http://localhost:8005`

2. **Install Required Packages** (if not already installed)
   ```bash
   pip install requests pandas numpy pytest
   ```

### Running the Tests

#### Option 1: Use the Test Runner (Recommended)
```bash
python tests/run_backend_tests.py
```

This will:
- Check if the backend is running
- Install any missing dependencies
- Run all integration tests
- Provide a clear summary of results

#### Option 2: Run Tests Directly
```bash
# From the project root
python tests/test_backend_integration.py
```

#### Option 3: Use pytest
```bash
# From the project root
pytest tests/test_backend_integration.py -v
```

## Test Coverage

The integration tests cover all the key functionality from Phase 1:

### ✅ Core Infrastructure (Week 1)
- [x] FastAPI server startup
- [x] Health check endpoints
- [x] CORS middleware
- [x] Session management foundation

### ✅ Data Management APIs (Week 2)
- [x] File upload service (CSV/Excel)
- [x] Data validation and processing
- [x] Data preview generation
- [x] Data manipulation operations
- [x] Session storage and retrieval

### ✅ Plot Generation APIs (Week 3)
- [x] Plot types configuration
- [x] Plot generation endpoints
- [x] Plot validation
- [x] Export functionality
- [x] BWR Plots integration

## Expected Test Results

When all tests pass, you should see:

```
============================================================
🎉 ALL BACKEND TESTS PASSED!
✅ Phase 1 (Backend API Development) is COMPLETE
🚀 Ready to proceed to Phase 2 (Frontend Development)
============================================================
```

## Troubleshooting

### Backend Server Not Running
```
❌ Backend server is not accessible
```
**Solution**: Start the backend server:
```bash
cd backend
python main.py
```

### Missing Dependencies
```
❌ Failed to install requirements
```
**Solution**: Install dependencies manually:
```bash
pip install requests pandas numpy pytest
```

### Test Failures
If specific tests fail, check:
1. Backend server logs for errors
2. File permissions for upload directory
3. Python environment and package versions
4. Network connectivity to localhost:8005

## Integration with Refactor Plan

These tests verify completion of:

- **Phase 1: Backend API Development (Weeks 1-3)** ✅
  - Week 1: Project Setup & Core Infrastructure
  - Week 2: Data Management APIs  
  - Week 3: Plot Generation APIs

Once these tests pass, you can proceed to:

- **Phase 2: Frontend Development (Weeks 4-7)**
  - Week 4: Next.js Setup & Core Components
  - Week 5: Data Management Components
  - Week 6: Plot Configuration & Display
  - Week 7: Layout & Integration

## Updating the Checklist

After successful test completion, update the refactor checklist:

```bash
# Mark Phase 1 as complete in .project/REFACTOR_CHECKLIST.md
# Begin Phase 2 tasks
```

## Next Steps

1. ✅ Run these backend tests
2. ✅ Verify all tests pass
3. ✅ Update refactor checklist
4. 🚀 Begin Phase 2: Frontend Development

The backend foundation is now solid and ready for frontend integration! 