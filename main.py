from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
import os

from database import create_tables, get_connection


class GPUInfo(BaseModel):
    user_id: int
    gpu_name: str
    vram: str
    backend: str


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

    user_id = request.session.get("user_id")
    role = request.session.get("role")

    if user_id is None:
        return RedirectResponse(
            url="/login",
            status_code=303
        )

    if role != "HOST":
        return {
            "error": "Only GPU hosts can access this dashboard."
        }

    connection = get_connection()

    host = connection.execute(
        """
        SELECT *
        FROM gpu_hosts
        WHERE user_id = ?
        """,
        (user_id,)
    ).fetchone()

    requests = []

    if host is not None:
        requests = connection.execute(
            """
            SELECT
                access_requests.id,
                users.name,
                users.email,
                access_requests.status
            FROM access_requests
            JOIN users
            ON access_requests.user_id = users.id
            WHERE access_requests.host_id = ?
            AND access_requests.status = 'PENDING'
            """,
            (host["id"],)
        ).fetchall()

    connection.close()

    return templates.TemplateResponse(
        "host_dashboard.html",
        {
            "request": request,
            "access_requests": requests
        }
    )

@app.post("/approve-request")
def approve_request(
    request: Request,
    request_id: int = Form(...)
):

    user_id = request.session.get("user_id")
    role = request.session.get("role")

    if user_id is None:
        return RedirectResponse(
            url="/login",
            status_code=303
        )

    if role != "HOST":
        return {
            "error": "Only GPU hosts can approve requests."
        }

    connection = get_connection()

    host = connection.execute(
        """
        SELECT id
        FROM gpu_hosts
        WHERE user_id = ?
        """,
        (user_id,)
    ).fetchone()

    if host is None:
        connection.close()

        return {
            "error": "GPU host not found."
        }

    connection.execute(
        """
        UPDATE access_requests
        SET status = 'APPROVED'
        WHERE id = ?
        AND host_id = ?
        """,
        (request_id, host["id"])
    )

    connection.commit()
    connection.close()

    return RedirectResponse(
        url="/host-dashboard",
        status_code=303
    )

@app.post("/reject-request")
def reject_request(
    request: Request,
    request_id: int = Form(...)
):

    user_id = request.session.get("user_id")
    role = request.session.get("role")

    if user_id is None:
        return RedirectResponse(
            url="/login",
            status_code=303
        )

    if role != "HOST":
        return {
            "error": "Only GPU hosts can reject requests."
        }

    connection = get_connection()

    host = connection.execute(
        """
        SELECT id
        FROM gpu_hosts
        WHERE user_id = ?
        """,
        (user_id,)
    ).fetchone()

    if host is None:
        connection.close()

        return {
            "error": "GPU host not found."
        }

    connection.execute(
        """
        UPDATE access_requests
        SET status = 'REJECTED'
        WHERE id = ?
        AND host_id = ?
        """,
        (request_id, host["id"])
    )

    connection.commit()
    connection.close()

    return RedirectResponse(
        url="/host-dashboard",
        status_code=303
    )

@app.get("/user-dashboard")
def user_dashboard(request: Request):

    user_id = request.session.get("user_id")
    role = request.session.get("role")

    if user_id is None:
        return RedirectResponse(
            url="/login",
            status_code=303
        )

    if role != "USER":
        return {
            "error": "Only GPU users can access this dashboard."
        }

    connection = get_connection()

    hosts = connection.execute(
        """
        SELECT
            gpu_hosts.id,
            users.name,
            gpu_hosts.gpu_name,
            gpu_hosts.vram,
            gpu_hosts.status
        FROM gpu_hosts
        JOIN users
        ON gpu_hosts.user_id = users.id
        WHERE gpu_hosts.status = 'ONLINE'
        """
    ).fetchall()

    access_requests = connection.execute(
        """
        SELECT
            access_requests.id,
            users.name AS host_name,
            gpu_hosts.gpu_name,
            access_requests.status
        FROM access_requests
        JOIN gpu_hosts
        ON access_requests.host_id = gpu_hosts.id
        JOIN users
        ON gpu_hosts.user_id = users.id
        WHERE access_requests.user_id = ?
        ORDER BY access_requests.id DESC
        """,
        (user_id,)
    ).fetchall()

    jobs = connection.execute(
        """
        SELECT
            jobs.id,
            jobs.job_type,
            jobs.status,
            jobs.result,
            users.name AS host_name,
            gpu_hosts.gpu_name
        FROM jobs
        JOIN gpu_hosts
        ON jobs.host_id = gpu_hosts.id
        JOIN users
        ON gpu_hosts.user_id = users.id
        WHERE jobs.user_id = ?
        ORDER BY jobs.id DESC
        """,
        (user_id,)
    ).fetchall()

    connection.close()

    return templates.TemplateResponse(
        "user_dashboard.html",
        {
            "request": request,
            "hosts": hosts,
            "access_requests": access_requests,
            "jobs": jobs
        }
    )

@app.post("/request-access")
def request_access(
    request: Request,
    host_id: int = Form(...)
):

    user_id = request.session.get("user_id")
    role = request.session.get("role")

    if user_id is None:
        return RedirectResponse(
            url="/login",
            status_code=303
        )

    if role != "USER":
        return {
            "error": "Only GPU users can request access."
        }

    connection = get_connection()

    host = connection.execute(
        "SELECT * FROM gpu_hosts WHERE id = ?",
        (host_id,)
    ).fetchone()

    if host is None:
        connection.close()

        return {
            "error": "GPU host not found."
        }

    existing_request = connection.execute(
        """
        SELECT * FROM access_requests
        WHERE user_id = ?
        AND host_id = ?
        AND status = 'PENDING'
        """,
        (user_id, host_id)
    ).fetchone()

    if existing_request is not None:
        connection.close()

        return {
            "message": "Access request already pending."
        }

    connection.execute(
        """
        INSERT INTO access_requests (user_id, host_id, status)
        VALUES (?, ?, ?)
        """,
        (user_id, host_id, "PENDING")
    )

    connection.commit()
    connection.close()

    return {
        "message": "GPU access request sent successfully."
    }


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

@app.get("/create-job")
def create_job_page(request: Request):

    user_id = request.session.get("user_id")
    role = request.session.get("role")

    if user_id is None:
        return RedirectResponse(
            url="/login",
            status_code=303
        )

    if role != "USER":
        return {
            "error": "Only GPU users can create jobs."
        }

    connection = get_connection()

    hosts = connection.execute(
        """
        SELECT
            gpu_hosts.id,
            users.name,
            gpu_hosts.gpu_name
        FROM gpu_hosts
        JOIN users
        ON gpu_hosts.user_id = users.id
        JOIN access_requests
        ON access_requests.host_id = gpu_hosts.id
        WHERE access_requests.user_id = ?
        AND access_requests.status = 'APPROVED'
        AND gpu_hosts.status = 'ONLINE'
        """,
        (user_id,)
    ).fetchall()

    connection.close()

    return templates.TemplateResponse(
        "create_job.html",
        {
            "request": request,
            "hosts": hosts
        }
    )


@app.post("/host/register")
def register_host_gpu(gpu: GPUInfo):

    connection = get_connection()

    host = connection.execute(
        "SELECT * FROM gpu_hosts WHERE user_id = ?",
        (gpu.user_id,)
    ).fetchone()

    if host is None:
        connection.close()

        return {
            "error": "GPU host is not registered."
        }

    connection.execute(
        """
        UPDATE gpu_hosts
        SET gpu_name = ?, vram = ?, status = ?
        WHERE user_id = ?
        """,
        (
            gpu.gpu_name,
            gpu.vram,
            "ONLINE",
            gpu.user_id
        )
    )

    connection.commit()
    connection.close()

    return {
        "message": "GPU registered successfully."
    }



@app.post("/create-job")
def create_job(
    request: Request,
    host_id: int = Form(...),
    job_type: str = Form(...),
    image: UploadFile = File(None)
):

    user_id = request.session.get("user_id")
    role = request.session.get("role")

    if user_id is None:
        return RedirectResponse(
            url="/login",
            status_code=303
        )

    if role != "USER":
        return {
            "error": "Only GPU users can create jobs."
        }

    allowed_types = {
        "IMAGE_PROCESSING",
        "MODEL_TRAINING",
        "MODEL_INFERENCE",
        "VIDEO_PROCESSING"
    }

    if job_type not in allowed_types:
        return {
            "error": "Invalid job type."
        }
    image_path = None

    if job_type == "IMAGE_PROCESSING":

        if image is None:
            return {
                "error": "Please upload an image."
            }

        image_path = f"uploads/{image.filename}"

        contents = image.file.read()

        with open(image_path, "wb") as file:
            file.write(contents)

    connection = get_connection()

    approved_access = connection.execute(
        """
        SELECT *
        FROM access_requests
        WHERE user_id = ?
        AND host_id = ?
        AND status = 'APPROVED'
        """,
        (user_id, host_id)
    ).fetchone()

    if approved_access is None:
        connection.close()

        return {
            "error": "You do not have approved access to this GPU host."
        }

    connection.execute(
        """
        INSERT INTO jobs (
            user_id,
            host_id,
            job_type,
            status,
            image_path
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            user_id,
            host_id,
            job_type,
            "QUEUED",
            image_path
        )
    )

    connection.commit()
    connection.close()

    return {
        "message": "GPU job submitted successfully."
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

@app.post("/host/jobs/next")
def get_next_job(user_id: int):

    connection = get_connection()

    host = connection.execute(
        """
        SELECT *
        FROM gpu_hosts
        WHERE user_id = ?
        AND status = 'ONLINE'
        """,
        (user_id,)
    ).fetchone()

    if host is None:
        connection.close()

        return {
            "job": None
        }

    job = connection.execute(
        """
        SELECT *
        FROM jobs
        WHERE host_id = ?
        AND status = 'QUEUED'
        ORDER BY id ASC
        LIMIT 1
        """,
        (host["id"],)
    ).fetchone()

    if job is None:
        connection.close()

        return {
            "job": None
        }

    connection.execute(
        """
        UPDATE jobs
        SET status = 'RUNNING'
        WHERE id = ?
        """,
        (job["id"],)
    )

    connection.commit()
    connection.close()

    return {
    "job": {
        "id": job["id"],
        "user_id": job["user_id"],
        "host_id": job["host_id"],
        "job_type": job["job_type"],
        "status": "RUNNING",
        "image_path": job["image_path"]
    }
}
@app.get("/host/jobs/{job_id}/image")
def download_job_image(job_id: int):

    connection = get_connection()

    job = connection.execute(
        """
        SELECT image_path
        FROM jobs
        WHERE id = ?
        """,
        (job_id,)
    ).fetchone()

    connection.close()

    if job is None:
        return {"error": "Job not found."}

    if job["image_path"] is None:
        return {"error": "This job has no image."}

    if not os.path.exists(job["image_path"]):
        return {"error": "Image file not found."}

    return FileResponse(job["image_path"])



class JobResult(BaseModel):
    status: str
    result: dict


@app.post("/host/jobs/{job_id}/complete")
def complete_job(job_id: int, job_result: JobResult):

    connection = get_connection()

    job = connection.execute(
        """
        SELECT *
        FROM jobs
        WHERE id = ?
        """,
        (job_id,)
    ).fetchone()

    if job is None:
        connection.close()

        return {
            "error": "Job not found."
        }

    connection.execute(
        """
        UPDATE jobs
        SET status = ?, result = ?
        WHERE id = ?
        """,
        (
            job_result.status,
            str(job_result.result),
            job_id
        )
    )

    connection.commit()
    connection.close()

    return {
        "message": "Job completed successfully."
    }