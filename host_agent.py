import torch
import requests
import time
from PIL import Image
from io import BytesIO
from PIL import Image


SERVER_URL = "http://127.0.0.1:8000"
USER_ID = 2


def detect_gpu():

    if torch.cuda.is_available():

        gpu_name = torch.cuda.get_device_name(0)

        total_memory = torch.cuda.get_device_properties(0).total_memory

        vram = round(total_memory / (1024 ** 3), 2)

        backend = "CUDA"

        return gpu_name, vram, backend

    if torch.backends.mps.is_available():

        gpu_name = "Apple Silicon GPU"

        vram = "Shared Memory"

        backend = "MPS"

        return gpu_name, vram, backend

    return None, None, None


def get_next_job():

    response = requests.post(
        f"{SERVER_URL}/host/jobs/next",
        params={
            "user_id": USER_ID
        }
    )

    return response.json()

def download_job_image(job_id):

    response = requests.get(
        f"{SERVER_URL}/host/jobs/{job_id}/image"
    )

    if response.status_code != 200:
        raise Exception("Failed to download image")

    image = Image.open(
        BytesIO(response.content)
    ).convert("RGB")

    return image


def execute_image_processing(job_id):

    print("Downloading image...")

    pil_image = download_job_image(job_id)

    print("Image downloaded:", pil_image.size)

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    print("Using device:", device)

    import numpy as np

    image = torch.from_numpy(
        np.array(pil_image)
    ).float() / 255.0

    image = image.permute(
        2, 0, 1
    ).unsqueeze(0).to(device)

    print("Image tensor moved to:", device)

    processed = torch.nn.functional.avg_pool2d(
        image,
        kernel_size=5,
        stride=1,
        padding=2
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        torch.mps.synchronize()

    print("GPU processing finished!")

    # Move processed tensor back to CPU
    processed = processed.squeeze(0)
    processed = processed.permute(1, 2, 0)
    processed = processed.cpu()

    # Convert values back to image format
    processed = (processed * 255).clamp(0, 255)
    processed = processed.byte().numpy()

    output_image = Image.fromarray(processed)

    output_path = f"outputs/job_{job_id}_processed.jpg"

    output_image.save(output_path)

    print("Processed image saved:", output_path)

    return {
        "device": str(device),
        "resolution": f"{pil_image.width}x{pil_image.height}",
        "operation": "Average Blur",
        "output_path": output_path,
        "status": "SUCCESS"
    }


# --------------------------------------------------
# GPU DETECTION
# --------------------------------------------------

gpu_name, vram, backend = detect_gpu()


if gpu_name is None:

    print("No supported GPU detected.")


else:

    print("GPU detected!")

    print("GPU:", gpu_name)

    print("VRAM:", vram, "GB")

    print("Backend:", backend)


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


    print("Server response:")

    print(response.json())


    # --------------------------------------------------
    # WAIT FOR GPU JOBS
    # --------------------------------------------------

    print("Host Agent is now waiting for jobs...")


    while True:

        try:

            data = get_next_job()

            job = data.get("job")


            if job is None:

                print("No jobs available.")


            else:

                print()
                print("================================")
                print("New GPU job received!")
                print("Job ID:", job["id"])
                print("Job Type:", job["job_type"])
                print("Status:", job["status"])
                print("================================")


                # --------------------------------------------------
                # EXECUTE IMAGE PROCESSING JOB
                # --------------------------------------------------

            if job["job_type"] == "IMAGE_PROCESSING":

                try:

                    result = execute_image_processing(
                        job["id"]
                    )

                    print()
                    print("Job Result:")
                    print(result)
                    print()

                    output_path = result["output_path"]

                    print("Uploading processed image...")

                    with open(output_path, "rb") as file:

                        upload_response = requests.post(
                            f"{SERVER_URL}/host/jobs/{job['id']}/upload-result",
                            files={
                                "result_file": (
                                    "processed.jpg",
                                    file,
                                    "image/jpeg"
                                )
                            }
                        )

                    print("Upload response:")
                    print(upload_response.json())

                    completion_data = {
                        "status": "COMPLETED",
                        "result": result
                    }

                    completion_response = requests.post(
                        f"{SERVER_URL}/host/jobs/{job['id']}/complete",
                        json=completion_data
                    )

                    print("Server completion response:")
                    print(completion_response.json())

                except Exception as error:

                    print()
                    print("GPU job failed!")
                    print("Error:", error)
                    print()

                    failure_data = {
                        "status": "FAILED",
                        "result": {
                            "error": str(error)
                        }
                    }

                    failure_response = requests.post(
                        f"{SERVER_URL}/host/jobs/{job['id']}/complete",
                        json=failure_data
                    )

                    print("Failure response:")
                    print(failure_response.json())

            else:

                print(
                        "Job type not implemented yet:",
                        job["job_type"]
                )


        except Exception as error:

            print("Host Agent error:", error)


        time.sleep(2)