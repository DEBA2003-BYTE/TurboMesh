from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse

from database import create_tables, get_connection

app = FastAPI()

templates = Jinja2Templates(directory="templates")

create_tables()


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request}
    )

@app.get("/register")
def register_page(request: Request):
    return templates.TemplateResponse(
        "register.html",
        {"request": request}
    )


@app.post("/register")
def register_user(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...)
):
    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO users (name, email, password, role)
            VALUES (?, ?, ?, ?)
            """,
            (name, email, password, role)
        )

        connection.commit()

    except Exception:
        connection.close()

        return {
            "error": "An account with this email already exists."
        }

    connection.close()

    return RedirectResponse(
        url="/",
        status_code=303
    )