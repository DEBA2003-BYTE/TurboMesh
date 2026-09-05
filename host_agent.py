import torch
import requests
import time


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


def execute_image_processing():

    print("Starting image processing...")

    # Select the best available device
    if torch.cuda.is_available():

        device = torch.device("cuda")

    elif torch.backends.mps.is_available():

        device = torch.device("mps")

    else:

        device = torch.device("cpu")

    print("Using device:", device)

    # Create an image-like tensor directly on the selected device
    image = torch.rand(
        (1, 3, 2048, 2048),
        device=device
    )

    print("Image tensor created on:", image.device)

    # Perform actual image processing
    processed_image = torch.nn.functional.avg_pool2d(
        image,
        kernel_size=3,
        stride=1,
        padding=1
    )

    # Wait for GPU operations to finish
    if device.type == "mps":

        torch.mps.synchronize()

    elif device.type == "cuda":

        torch.cuda.synchronize()

    print("Image processing completed.")

    print(
        "Processed tensor shape:",
        processed_image.shape
    )

    return {
        "device": str(device),
        "input_size": str(tuple(image.shape)),
        "output_size": str(tuple(processed_image.shape)),
        "operation": "Average Pooling",
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
                    result = execute_image_processing()

                    print()
                    print("Job Result:")
                    print(result)
                    print()

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


                else:

                    print(
                        "Job type not implemented yet:",
                        job["job_type"]
                    )


        except Exception as error:

            print("Host Agent error:", error)


        time.sleep(2)