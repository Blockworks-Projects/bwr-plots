#!/usr/bin/env python3
"""
Backend Test Runner for BWR Plots Frontend Refactor

This script helps run the backend integration tests to verify that Phase 1
(Backend API Development) is complete before proceeding to Phase 2 (Frontend Development).

Usage:
    python tests/run_backend_tests.py

Prerequisites:
    1. Backend server must be running: cd backend && python main.py
    2. Required Python packages: requests, pandas, numpy, pytest
"""

import sys
import subprocess
import time
import requests
from pathlib import Path

def check_backend_server():
    """Check if the backend server is running"""
    try:
        response = requests.get("http://localhost:8005/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend server is running")
            return True
        else:
            print(f"❌ Backend server responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend server is not accessible: {e}")
        return False

def install_requirements():
    """Install required packages if not available"""
    required_packages = ["requests", "pandas", "numpy", "pytest"]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def run_tests():
    """Run the backend integration tests"""
    print("\n" + "="*60)
    print("BWR PLOTS BACKEND TEST RUNNER")
    print("="*60)
    
    # Check if backend is running
    if not check_backend_server():
        print("\n🔧 Please start the backend server first:")
        print("   cd backend")
        print("   python main.py")
        print("\nThen run this test again.")
        return False
    
    # Install requirements
    print("\n📦 Checking required packages...")
    try:
        install_requirements()
        print("✅ All required packages are available")
    except Exception as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    
    # Run the integration test
    print("\n🧪 Running backend integration tests...")
    try:
        # Import and run the test
        from test_backend_integration import test_backend_readiness
        success = test_backend_readiness()
        return success
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    success = run_tests()
    
    if success:
        print("\n🎉 BACKEND TESTS COMPLETED SUCCESSFULLY!")
        print("✅ Phase 1 (Backend API Development) is verified and complete")
        print("🚀 You can now proceed to Phase 2 (Frontend Development)")
        print("\nNext steps:")
        print("1. Update the refactor checklist to mark Phase 1 as complete")
        print("2. Begin Phase 2: Next.js setup and frontend development")
    else:
        print("\n❌ BACKEND TESTS FAILED!")
        print("🔧 Please fix the issues before proceeding to frontend development")
        print("\nTroubleshooting:")
        print("1. Ensure backend server is running: cd backend && python main.py")
        print("2. Check backend logs for errors")
        print("3. Verify all backend dependencies are installed")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 