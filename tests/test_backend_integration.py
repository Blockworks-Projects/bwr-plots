"""
Backend Integration Tests for BWR Plots Frontend Refactor

This test suite verifies that Phase 1 (Backend API Development) is working correctly
before proceeding to Phase 2 (Frontend Development). It tests all the core backend
functionality that will be consumed by the Next.js frontend.

Test Coverage:
- FastAPI server startup and health checks
- File upload and data processing endpoints
- Session management functionality
- Plot generation endpoints
- Data manipulation operations
- Error handling and validation

This ensures we have a solid backend foundation before building the frontend.
"""

import pytest
import requests
import pandas as pd
import numpy as np
import tempfile
import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add backend to path for imports
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Test configuration
BASE_URL = "http://localhost:8005"
API_BASE = f"{BASE_URL}/api/v1"

class TestBackendIntegration:
    """Integration tests for the BWR Plots backend API"""
    
    @classmethod
    def setup_class(cls):
        """Set up test class - ensure backend is running"""
        cls.session_ids = []  # Track session IDs for cleanup
        
        # Check if backend is running
        try:
            response = requests.get(f"{API_BASE}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Backend server is not running. Start with: cd backend && python main.py")
        except requests.exceptions.RequestException:
            pytest.skip("Backend server is not running. Start with: cd backend && python main.py")
    
    @classmethod
    def teardown_class(cls):
        """Clean up test sessions"""
        for session_id in cls.session_ids:
            try:
                requests.delete(f"{API_BASE}/data/{session_id}")
            except:
                pass  # Ignore cleanup errors
    
    def create_test_csv_file(self, filename="test_data.csv", num_rows=100):
        """Create a test CSV file with sample data"""
        # Generate test data similar to what BWR Plots would handle
        dates = pd.date_range(start='2023-01-01', periods=num_rows, freq='D')
        
        data = {
            'Date': dates,
            'Ethereum': np.random.uniform(1000, 2000, num_rows),
            'Solana': np.random.uniform(50, 150, num_rows),
            'Avalanche': np.random.uniform(10, 50, num_rows),
            'Polygon': np.random.uniform(0.5, 2.0, num_rows),
            'Volume': np.random.uniform(1000000, 10000000, num_rows),
            'Category': np.random.choice(['DeFi', 'NFT', 'Gaming', 'Infrastructure'], num_rows)
        }
        
        df = pd.DataFrame(data)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return temp_file.name, df
    
    def test_01_health_check(self):
        """Test basic health check endpoint"""
        response = requests.get(f"{API_BASE}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["service"] == "BWR Plots API"
        print("✅ Health check endpoint working")
    
    def test_02_detailed_health_check(self):
        """Test detailed health check with system information"""
        response = requests.get(f"{API_BASE}/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "python_version" in data
        assert "system_info" in data
        assert "storage" in data
        print("✅ Detailed health check endpoint working")
    
    def test_03_file_upload_csv(self):
        """Test CSV file upload and data processing"""
        # Create test CSV file
        csv_file_path, expected_df = self.create_test_csv_file()
        
        try:
            # Upload file
            with open(csv_file_path, 'rb') as f:
                files = {'file': ('test_data.csv', f, 'text/csv')}
                response = requests.post(f"{API_BASE}/data/upload", files=files)
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "session_id" in data
            assert data["filename"] == "test_data.csv"
            assert data["total_rows"] == len(expected_df)
            assert len(data["columns"]) == len(expected_df.columns)
            assert data["file_size_bytes"] > 0
            
            # Store session ID for cleanup and further tests
            session_id = data["session_id"]
            self.session_ids.append(session_id)
            
            # Verify column information
            column_names = [col["name"] for col in data["columns"]]
            assert set(column_names) == set(expected_df.columns)
            
            print(f"✅ CSV file upload successful, session: {session_id}")
            return session_id
            
        finally:
            # Clean up temporary file
            os.unlink(csv_file_path)
    
    def test_04_data_preview(self):
        """Test data preview endpoint"""
        # First upload a file to get a session
        session_id = self.test_03_file_upload_csv()
        
        # Get data preview
        response = requests.get(f"{API_BASE}/data/preview/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify preview structure
        assert data["session_id"] == session_id
        assert "data" in data
        assert "columns" in data
        assert data["total_rows"] > 0
        assert data["preview_rows"] <= 100  # Default max_rows
        assert len(data["data"]) <= 100
        
        # Verify we have the expected columns
        column_names = [col["name"] for col in data["columns"]]
        expected_columns = ['Date', 'Ethereum', 'Solana', 'Avalanche', 'Polygon', 'Volume', 'Category']
        assert set(column_names) == set(expected_columns)
        
        print(f"✅ Data preview working for session: {session_id}")
    
    def test_05_data_manipulation(self):
        """Test data manipulation operations"""
        # First upload a file to get a session
        session_id = self.test_03_file_upload_csv()
        
        # Test column dropping operation
        manipulation_request = {
            "session_id": session_id,
            "operations": [
                {
                    "type": "drop_columns",
                    "params": {
                        "columns": ["Category"]
                    }
                }
            ]
        }
        
        response = requests.post(
            f"{API_BASE}/data/manipulate",
            json=manipulation_request,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify manipulation result
        assert data["session_id"] == session_id
        assert data["success"] is True
        assert "drop_columns" in data["operations_applied"]
        assert data["new_shape"][1] == data["original_shape"][1] - 1  # One less column
        assert "Category" not in data["columns"]
        
        print(f"✅ Data manipulation working for session: {session_id}")
    
    def test_06_plot_types_endpoint(self):
        """Test plot types configuration endpoint"""
        response = requests.get(f"{API_BASE}/plots/types")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify we have plot types
        assert "plot_types" in data
        plot_types = data["plot_types"]
        
        # Check for expected BWR plot types
        expected_types = ["line", "bar", "stacked_bar", "area", "scatter"]
        available_types = [pt["type"] for pt in plot_types]
        
        # At least some of the expected types should be available
        assert len(set(expected_types) & set(available_types)) > 0
        
        # Each plot type should have required fields
        for plot_type in plot_types:
            assert "type" in plot_type
            assert "name" in plot_type
            assert "description" in plot_type
            assert "required_columns" in plot_type
        
        print(f"✅ Plot types endpoint working, found {len(plot_types)} plot types")
    
    def test_07_plot_generation(self):
        """Test plot generation endpoint"""
        # First upload a file to get a session
        session_id = self.test_03_file_upload_csv()
        
        # Generate a simple line plot
        plot_request = {
            "session_id": session_id,
            "plot_type": "line",
            "configuration": {
                "x_column": "Date",
                "y_columns": ["Ethereum", "Solana"],
                "title": "Test Crypto Prices",
                "subtitle": "Integration Test Plot",
                "source": "Test Data",
                "date": "2024"
            },
            "data_processing": {
                "date_column": "Date",
                "filter_config": {},
                "resample_config": {},
                "smooth_config": {}
            }
        }
        
        response = requests.post(
            f"{API_BASE}/plots/generate",
            json=plot_request,
            headers={"Content-Type": "application/json"},
            timeout=30  # Plot generation might take time
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify plot generation result
        assert data["success"] is True
        assert "plot_json" in data
        assert "plot_html" in data
        assert data["session_id"] == session_id
        
        # Verify plot JSON structure (basic Plotly structure)
        plot_json = data["plot_json"]
        assert "data" in plot_json
        assert "layout" in plot_json
        
        # Verify HTML contains plot
        plot_html = data["plot_html"]
        assert "plotly" in plot_html.lower()
        assert "html" in plot_html.lower()
        
        print(f"✅ Plot generation working for session: {session_id}")
    
    def test_08_session_management(self):
        """Test session management functionality"""
        # Upload a file to create a session
        session_id = self.test_03_file_upload_csv()
        
        # Test getting column options
        response = requests.get(f"{API_BASE}/data/columns/{session_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert "columns" in data
        assert len(data["columns"]) > 0
        
        # Test data summary
        response = requests.get(f"{API_BASE}/data/summary/{session_id}")
        assert response.status_code == 200
        
        summary = response.json()
        assert "session_id" in summary
        assert "data_shape" in summary
        assert "column_info" in summary
        
        print(f"✅ Session management working for session: {session_id}")
    
    def test_09_error_handling(self):
        """Test error handling for invalid requests"""
        # Test invalid session ID
        response = requests.get(f"{API_BASE}/data/preview/invalid_session_id")
        assert response.status_code == 404
        
        # Test invalid file upload
        files = {'file': ('test.txt', b'invalid content', 'text/plain')}
        response = requests.post(f"{API_BASE}/data/upload", files=files)
        assert response.status_code == 400
        
        # Test invalid plot request
        invalid_plot_request = {
            "session_id": "invalid_session",
            "plot_type": "invalid_type",
            "configuration": {}
        }
        response = requests.post(
            f"{API_BASE}/plots/generate",
            json=invalid_plot_request,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 404]
        
        print("✅ Error handling working correctly")
    
    def test_10_performance_basic(self):
        """Test basic performance requirements"""
        # Test health check response time
        start_time = time.time()
        response = requests.get(f"{API_BASE}/health")
        health_time = time.time() - start_time
        
        assert response.status_code == 200
        assert health_time < 1.0  # Should respond within 1 second
        
        # Test file upload performance with small file
        csv_file_path, _ = self.create_test_csv_file(num_rows=50)
        
        try:
            start_time = time.time()
            with open(csv_file_path, 'rb') as f:
                files = {'file': ('small_test.csv', f, 'text/csv')}
                response = requests.post(f"{API_BASE}/data/upload", files=files)
            upload_time = time.time() - start_time
            
            assert response.status_code == 200
            assert upload_time < 10.0  # Should upload within 10 seconds
            
            # Clean up session
            if response.status_code == 200:
                session_id = response.json()["session_id"]
                self.session_ids.append(session_id)
            
        finally:
            os.unlink(csv_file_path)
        
        print(f"✅ Basic performance tests passed (health: {health_time:.2f}s, upload: {upload_time:.2f}s)")


def test_backend_readiness():
    """
    Main test function to verify backend readiness for frontend development.
    
    This test ensures that Phase 1 (Backend API Development) is complete and
    we can proceed to Phase 2 (Frontend Development).
    """
    print("\n" + "="*60)
    print("BWR PLOTS BACKEND INTEGRATION TEST")
    print("Testing Phase 1 completion before Phase 2 development")
    print("="*60)
    
    # Run the test suite
    test_instance = TestBackendIntegration()
    test_instance.setup_class()
    
    try:
        # Core functionality tests
        test_instance.test_01_health_check()
        test_instance.test_02_detailed_health_check()
        test_instance.test_03_file_upload_csv()
        test_instance.test_04_data_preview()
        test_instance.test_05_data_manipulation()
        test_instance.test_06_plot_types_endpoint()
        test_instance.test_07_plot_generation()
        test_instance.test_08_session_management()
        test_instance.test_09_error_handling()
        test_instance.test_10_performance_basic()
        
        print("\n" + "="*60)
        print("🎉 ALL BACKEND TESTS PASSED!")
        print("✅ Phase 1 (Backend API Development) is COMPLETE")
        print("🚀 Ready to proceed to Phase 2 (Frontend Development)")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Backend test failed: {str(e)}")
        print("🔧 Please fix backend issues before proceeding to frontend development")
        return False
        
    finally:
        test_instance.teardown_class()


if __name__ == "__main__":
    """
    Run this test to verify backend readiness.
    
    Usage:
    1. Start the backend server: cd backend && python main.py
    2. Run this test: python tests/test_backend_integration.py
    """
    success = test_backend_readiness()
    exit(0 if success else 1) 