# app/routes/home.py

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from app.config import settings
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory=settings.BASE_DIR + "/templates")

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})