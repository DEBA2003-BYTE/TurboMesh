from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from database import create_tables, get_connection


app = FastAPI()
app.add_middleware(
    SessionMiddleware,
    secret_key="turbomesh-development-secret"
)

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
@app.post("/login")
def login_user(
    request: Request,
    email: str = Form(...),
    password: str = Form(...)
):
    connection = get_connection()

    user = connection.execute(
        "SELECT * FROM users WHERE email = ? AND password = ?",
        (email, password)
    ).fetchone()

    connection.close()

    if user is None:
        return {
            "error": "Invalid email or password."
        }

    request.session["user_id"] = user["id"]
    request.session["role"] = user["role"]

    if user["role"] == "HOST":
        return RedirectResponse(
            url="/host-dashboard",
            status_code=303
        )

    return RedirectResponse(
        url="/user-dashboard",
        status_code=303
    )
@app.get("/host-dashboard")
def host_dashboard(request: Request):
    return templates.TemplateResponse(
        "host_dashboard.html",
        {"request": request}
    )

@app.get("/user-dashboard")
def user_dashboard(request: Request):
    return templates.TemplateResponse(
        "user_dashboard.html",
        {"request": request}
    )

@app.get("/enable-gpu")
def enable_gpu(request: Request):

    user_id = request.session.get("user_id")
    role = request.session.get("role")

    if user_id is None:
        return RedirectResponse(
            url="/login",
            status_code=303
        )

    if role != "HOST":
        return {
            "error": "Only GPU hosts can enable GPU sharing."
        }

    connection = get_connection()

    existing_host = connection.execute(
        "SELECT * FROM gpu_hosts WHERE user_id = ?",
        (user_id,)
    ).fetchone()

    if existing_host is None:

        connection.execute(
            """
            INSERT INTO gpu_hosts (user_id, status)
            VALUES (?, ?)
            """,
            (user_id, "OFFLINE")
        )

        connection.commit()

    connection.close()

    return {
        "message": "GPU host registered successfully."
    }

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