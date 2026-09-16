"""LLM by Hand: one local server for the source curriculum and its execution tools."""
from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from qk.course_api import create_course_router

ROOT = Path(__file__).resolve().parent
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")


def create_app(db_path: Path | None = None, *, lesson_db_path: Path | None = None) -> FastAPI:
    database = Path(lesson_db_path or db_path or os.environ.get("QK_DB", ROOT / "data" / "progress.sqlite3"))
    application = FastAPI(title="LLM by Hand", version="1.0.0", docs_url=None, redoc_url=None)
    application.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"]
    )
    application.include_router(create_course_router(database))

    @application.middleware("http")
    async def local_browser_boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
            parsed = urlsplit(origin)
            if parsed.netloc != request.headers.get("host") or parsed.scheme not in {"http", "https"}:
                return JSONResponse({"detail": "Cross-origin writes are not allowed."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @application.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "web" / "course" / "index.html", headers={"Cache-Control": "no-cache"})

    @application.get("/api/health")
    def health():
        return {"ok": True, "application": "LLM by Hand", "local_only": True}

    application.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
    return application


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("QK_PORT", "8017"))
    print(f"LLM by Hand: http://127.0.0.1:{port}/#/001/follow")
    print("Trusted local Python only. This execution environment is not a security sandbox.")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
