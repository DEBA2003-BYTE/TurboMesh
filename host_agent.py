import torch
import requests
import time

from PIL import Image
from io import BytesIO


# --------------------------------------------------
# TURBOMESH SERVER
# --------------------------------------------------

SERVER_URL = "http://127.0.0.1:8000"

# Temporary local testing user ID
USER_ID = 2


# --------------------------------------------------
# GPU DETECTION
# --------------------------------------------------

def detect_gpu():

    # NVIDIA GPU
    if torch.cuda.is_available():

        gpu_name = torch.cuda.get_device_name(0)

        total_memory = torch.cuda.get_device_properties(
            0
        ).total_memory

        vram = round(
            total_memory / (1024 ** 3),
            2
        )

        backend = "CUDA"

        return gpu_name, vram, backend


    # Apple Silicon GPU
    if torch.backends.mps.is_available():

        gpu_name = "Apple Silicon GPU"

        vram = "Shared Memory"

        backend = "MPS"

        return gpu_name, vram, backend


    # No supported GPU
    return None, None, None


# --------------------------------------------------
# GET NEXT JOB
# --------------------------------------------------

def get_next_job():

    response = requests.post(
        f"{SERVER_URL}/host/jobs/next",
        params={
            "user_id": USER_ID
        }
    )

    return response.json()


# --------------------------------------------------
# DOWNLOAD JOB IMAGE
# --------------------------------------------------

def download_job_image(job_id):

    response = requests.get(
        f"{SERVER_URL}/host/jobs/{job_id}/image"
    )

    if response.status_code != 200:

        raise Exception(
            "Failed to download image"
        )

    image = Image.open(
        BytesIO(response.content)
    ).convert("RGB")

    return image


# --------------------------------------------------
# OBJECT DETECTION
# TEMPORARY TEST VERSION
# --------------------------------------------------

def execute_object_detection(job_id):

    print(
        "Downloading image for object detection..."
    )

    # Download image from TurboMesh server
    pil_image = download_job_image(job_id)

    print(
        "Image downloaded successfully!"
    )

    print(
        "Image size:",
        pil_image.size
    )

    return {
        "status": "IMAGE_DOWNLOADED",
        "resolution": (
            f"{pil_image.width}x"
            f"{pil_image.height}"
        )
    }


# --------------------------------------------------
# GPU DETECTION
# --------------------------------------------------

gpu_name, vram, backend = detect_gpu()


if gpu_name is None:

    print(
        "No supported GPU detected."
    )


else:

    print("GPU detected!")

    print(
        "GPU:",
        gpu_name
    )

    print(
        "VRAM:",
        vram,
        "GB"
    )

    print(
        "Backend:",
        backend
    )


    # --------------------------------------------------
    # REGISTER GPU WITH TURBOMESH SERVER
    # --------------------------------------------------

    data = {

        "user_id": USER_ID,

        "gpu_name": gpu_name,

        "vram": str(vram),

        "backend": backend
    }


    response = requests.post(
        f"{SERVER_URL}/host/register",
        json=data
    )


    print()
    print("Server response:")
    print(response.json())


    # --------------------------------------------------
    # WAIT FOR GPU JOBS
    # --------------------------------------------------

    print()
    print(
        "Host Agent is now waiting for jobs..."
    )


    while True:

        try:

            # Get next queued job
            data = get_next_job()

            job = data.get("job")


            # --------------------------------------------------
            # NO JOB
            # --------------------------------------------------

            if job is None:

                print(
                    "No jobs available."
                )


            # --------------------------------------------------
            # JOB RECEIVED
            # --------------------------------------------------

            else:

                print()

                print(
                    "================================"
                )

                print(
                    "New GPU job received!"
                )

                print(
                    "Job ID:",
                    job["id"]
                )

                print(
                    "Job Type:",
                    job["job_type"]
                )

                print(
                    "Status:",
                    job["status"]
                )

                print(
                    "================================"
                )


                # --------------------------------------------------
                # OBJECT DETECTION JOB
                # --------------------------------------------------

                if job["job_type"] == "OBJECT_DETECTION":

                    try:

                        result = execute_object_detection(
                            job["id"]
                        )


                        print()

                        print(
                            "Job Result:"
                        )

                        print(
                            result
                        )

                        print()


                    except Exception as error:

                        print()

                        print(
                            "GPU job failed!"
                        )

                        print(
                            "Error:",
                            error
                        )

                        print()


                # --------------------------------------------------
                # OTHER JOB TYPES
                # --------------------------------------------------

                else:

                    print(
                        "Job type not implemented yet:",
                        job["job_type"]
                    )


        except Exception as error:

            print(
                "Host Agent error:",
                error
            )


        # Wait before asking the server again
        time.sleep(2)