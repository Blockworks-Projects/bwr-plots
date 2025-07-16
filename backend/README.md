# BWR Plots Backend API

FastAPI backend for the BWR Plots data visualization application.

## Features

- RESTful API for data processing and plot generation
- Session-based data management
- File upload support (CSV, XLSX)
- Health check endpoints
- CORS support for frontend integration
- Comprehensive error handling

## Quick Start

### Prerequisites

- Python 3.10+
- Poetry

### Installation

1. Install dependencies from the project root:
```bash
cd /path/to/bwr-plots
poetry install
```

2. Run the development server:
```bash
poetry run python backend/main.py
```

Or from the backend directory:
```bash
cd backend
poetry run python main.py
```

The API will be available at `http://localhost:8005`

### Docker

Build and run with Docker:

```bash
docker build -t bwr-plots-backend .
docker run -p 8005:8005 bwr-plots-backend
```

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8005/docs`
- ReDoc: `http://localhost:8005/redoc`

## Project Structure

```
backend/
├── main.py                 # FastAPI application entry point
├── Dockerfile             # Container configuration
├── api/
│   ├── __init__.py
│   └── routes/
│       ├── __init__.py
│       └── health.py      # Health check endpoints
├── core/
│   ├── __init__.py
│   ├── config.py          # Application configuration
│   └── exceptions.py      # Custom exception handlers
├── models/
│   └── __init__.py
├── services/
│   ├── __init__.py
│   └── session_manager.py # Session management
├── utils/
│   └── __init__.py
├── storage/
│   ├── uploads/           # Temporary file storage
│   └── sessions/          # Session data storage
└── tests/
    └── __init__.py
```

## Configuration

The application uses environment variables for configuration. See the settings in `core/config.py` for available options.

Key settings:
- `CORS_ORIGINS`: Allowed frontend origins
- `MAX_FILE_SIZE`: Maximum upload file size (default: 100MB)
- `SESSION_TIMEOUT`: Session expiration time (default: 1 hour)

## Development

### Running Tests

From the project root:
```bash
poetry run pytest backend/tests/
```

### Code Style

The project follows PEP 8 style guidelines.

## Next Steps

This is the initial backend structure. The following endpoints will be added:

- `/api/v1/data/upload` - File upload
- `/api/v1/data/preview` - Data preview
- `/api/v1/data/manipulate` - Data manipulation
- `/api/v1/plots/generate` - Plot generation
- `/api/v1/plots/export` - Plot export

## License

This project is part of the BWR Plots application. 