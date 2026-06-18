"""
Legacy API - use api/main.py instead (Docker CMD points to api.main:app).
This file kept for backward compatibility with root Dockerfile.
"""

from api.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
