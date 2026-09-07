import torch
import requests
import time
import os

from ultralytics import YOLO
from PIL import Image
from io import BytesIO


# ==================================================
# TURBOMESH SERVER
# ==================================================

SERVER_URL = "http://127.0.0.1:8000"

# Temporary local testing user ID
USER_ID = 2


# ==================================================
# LOAD YOLO MODEL
# ==================================================

print("Loading YOLO model...")

yolo_model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully!")


# ==================================================
# CREATE OUTPUT DIRECTORY
# ==================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


# ==================================================
# GPU DETECTION
# ==================================================

def detect_gpu():

    # ----------------------------------------------
    # NVIDIA GPU
    # ----------------------------------------------

    if torch.cuda.is_available():

        gpu_name = torch.cuda.get_device_name(0)

        total_memory = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
        )

        vram = round(
            total_memory / (1024 ** 3),
            2
        )

        backend = "CUDA"

        return gpu_name, vram, backend


    # ----------------------------------------------
    # Apple Silicon GPU
    # ----------------------------------------------

    if torch.backends.mps.is_available():

        gpu_name = "Apple Silicon GPU"

        vram = "Shared Memory"

        backend = "MPS"

        return gpu_name, vram, backend


    # ----------------------------------------------
    # No supported GPU
    # ----------------------------------------------

    return None, None, None


# ==================================================
# GET NEXT JOB
# ==================================================

def get_next_job():

    response = requests.post(
        f"{SERVER_URL}/host/jobs/next",
        params={
            "user_id": USER_ID
        }
    )

    response.raise_for_status()

    return response.json()


# ==================================================
# DOWNLOAD JOB IMAGE
# ==================================================

def download_job_image(job_id):

    response = requests.get(
        f"{SERVER_URL}/host/jobs/{job_id}/image"
    )

    response.raise_for_status()

    image = Image.open(
        BytesIO(response.content)
    ).convert("RGB")

    return image


# ==================================================
# OBJECT DETECTION
# ==================================================

def execute_object_detection(job_id):

    print()
    print("Downloading image for object detection...")

    # ----------------------------------------------
    # Download image
    # ----------------------------------------------

    pil_image = download_job_image(job_id)

    print(
        "Image downloaded successfully!"
    )

    print(
        "Image size:",
        pil_image.size
    )


    # ----------------------------------------------
    # Select computation device
    # ----------------------------------------------

    if torch.cuda.is_available():

        device = "cuda"

    elif torch.backends.mps.is_available():

        device = "mps"

    else:

        device = "cpu"


    print(
        "Running YOLO on:",
        device
    )


    # ----------------------------------------------
    # Run YOLO
    # ----------------------------------------------

    results = yolo_model.predict(
        source=pil_image,
        device=device,
        verbose=False
    )


    # ----------------------------------------------
    # Collect detections
    # ----------------------------------------------

    detections = []

    object_counts = {}


    for result in results:

        for box in result.boxes:

            # Class ID
            class_id = int(
                box.cls[0]
            )

            # Confidence
            confidence = float(
                box.conf[0]
            )

            # Object name
            class_name = result.names[
                class_id
            ]


            # --------------------------------------
            # Store detection
            # --------------------------------------

            detections.append({

                "object": class_name,

                "confidence": round(
                    confidence,
                    3
                )

            })


            # --------------------------------------
            # Count objects
            # --------------------------------------

            if class_name not in object_counts:

                object_counts[class_name] = 0

            object_counts[class_name] += 1


    # ==================================================
    # GENERATE ANNOTATED IMAGE
    # ==================================================

    annotated_image = results[0].plot()


    # ==================================================
    # SAVE ANNOTATED IMAGE
    # ==================================================

    output_path = (
        f"outputs/"
        f"job_{job_id}_detected.jpg"
    )


    Image.fromarray(
        annotated_image
    ).save(
        output_path
    )


    print()
    print(
        "Annotated image saved:"
    )

    print(
        output_path
    )


    # ==================================================
    # PRINT DETECTION ANALYTICS
    # ==================================================

    print()
    print(
        "Detected objects:"
    )

    for object_name, count in object_counts.items():

        print(
            f"{object_name}: {count}"
        )


    # ==================================================
    # RETURN RESULT
    # ==================================================

    return {

        "status":
            "YOLO_DETECTION_COMPLETED",

        "device":
            device,

        "resolution":
            (
                f"{pil_image.width}x"
                f"{pil_image.height}"
            ),

        "detections":
            detections,

        "object_counts":
            object_counts,

        "output_path":
            output_path
    }


# ==================================================
# DETECT GPU
# ==================================================

gpu_name, vram, backend = detect_gpu()


# ==================================================
# START HOST AGENT
# ==================================================

if gpu_name is None:

    print()
    print(
        "No supported GPU detected."
    )

else:

    # ----------------------------------------------
    # GPU INFORMATION
    # ----------------------------------------------

    print()
    print(
        "================================"
    )

    print(
        "GPU detected!"
    )

    print(
        "GPU:",
        gpu_name
    )

    print(
        "VRAM:",
        vram
    )

    print(
        "Backend:",
        backend
    )

    print(
        "================================"
    )


    # ==================================================
    # REGISTER GPU WITH TURBOMESH SERVER
    # ==================================================

    data = {

        "user_id":
            USER_ID,

        "gpu_name":
            gpu_name,

        "vram":
            str(vram),

        "backend":
            backend
    }


    response = requests.post(
        f"{SERVER_URL}/host/register",
        json=data
    )

    response.raise_for_status()


    print()
    print(
        "Server response:"
    )

    print(
        response.json()
    )


    # ==================================================
    # WAIT FOR JOBS
    # ==================================================

    print()
    print(
        "Host Agent is now waiting for jobs..."
    )


    while True:

        try:

            # ------------------------------------------
            # Get next job
            # ------------------------------------------

            data = get_next_job()

            job = data.get(
                "job"
            )


            # ==================================================
            # NO JOB
            # ==================================================

            if job is None:

                print(
                    "No jobs available."
                )


            # ==================================================
            # JOB RECEIVED
            # ==================================================

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


                # ==================================================
                # OBJECT DETECTION
                # ==================================================

                if job["job_type"] == "OBJECT_DETECTION":

                    try:

                        # ------------------------------------------
                        # Execute YOLO
                        # ------------------------------------------

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


                        # ==================================================
                        # UPLOAD RESULT IMAGE
                        # ==================================================

                        output_path = result[
                            "output_path"
                        ]


                        print(
                            "Uploading YOLO result..."
                        )


                        with open(
                            output_path,
                            "rb"
                        ) as file:

                            upload_response = requests.post(

                                f"{SERVER_URL}"
                                f"/host/jobs/"
                                f"{job['id']}"
                                f"/upload-result",

                                files={

                                    "result_file": (

                                        "detected.jpg",

                                        file,

                                        "image/jpeg"
                                    )
                                }
                            )


                        upload_response.raise_for_status()


                        print(
                            "Upload response:"
                        )

                        print(
                            upload_response.json()
                        )


                        # ==================================================
                        # MARK JOB COMPLETED
                        # ==================================================

                        completion_data = {

                            "status":
                                "COMPLETED",

                            "result":
                                result
                        }


                        completion_response = requests.post(

                            f"{SERVER_URL}"
                            f"/host/jobs/"
                            f"{job['id']}"
                            f"/complete",

                            json=completion_data
                        )


                        completion_response.raise_for_status()


                        print(
                            "Server completion response:"
                        )

                        print(
                            completion_response.json()
                        )


                    # ==================================================
                    # JOB FAILED
                    # ==================================================

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


                        failure_data = {

                            "status":
                                "FAILED",

                            "result": {

                                "error":
                                    str(error)
                            }
                        }


                        try:

                            failure_response = requests.post(

                                f"{SERVER_URL}"
                                f"/host/jobs/"
                                f"{job['id']}"
                                f"/complete",

                                json=failure_data
                            )


                            print(
                                "Failure response:"
                            )

                            print(
                                failure_response.json()
                            )


                        except Exception as failure_error:

                            print(
                                "Could not update job status:",
                                failure_error
                            )


                # ==================================================
                # OTHER JOB TYPES
                # ==================================================

                else:

                    print(
                        "Job type not implemented yet:",
                        job["job_type"]
                    )


        # ==================================================
        # HOST AGENT ERROR
        # ==================================================

        except Exception as error:

            print()
            print(
                "Host Agent error:",
                error
            )

            print()


        # ==================================================
        # WAIT BEFORE POLLING AGAIN
        # ==================================================

        time.sleep(2)