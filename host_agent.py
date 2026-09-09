import torch
import requests
import time
import os
import cv2
import numpy as np
import imagehash
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from skimage.metrics import structural_similarity as ssim
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
# VEHICLE CLASSES
# ==================================================

VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "bus",
    "truck"
}


# ==================================================
# LOAD YOLO MODEL
# ==================================================

print("Loading YOLO model...")

yolo_model = YOLO("yolo11n.pt")
license_plate_model = YOLO("models/license_plate.pt")


print("YOLO model loaded successfully!")

# ==================================================
# LOAD IMAGE FEATURE MODEL
# ==================================================

print("Loading ResNet18 feature model...")

feature_weights = ResNet18_Weights.DEFAULT

feature_model = resnet18(
    weights=feature_weights
)

# Remove the final classification layer
feature_model.fc = torch.nn.Identity()

feature_model.eval()

print("ResNet18 feature model loaded successfully!")
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
# DOWNLOAD SECOND JOB IMAGE
# ==================================================

def download_job_image_2(job_id):

    print("Downloading second image...")

    response = requests.get(
        f"{SERVER_URL}/host/jobs/{job_id}/image/2"
    )

    response.raise_for_status()

    image = Image.open(
        BytesIO(response.content)
    ).convert("RGB")

    print("Second image downloaded successfully!")
    print("Second image size:", image.size)

    return image


# ==================================================
# TEST IMAGE SIMILARITY DOWNLOAD
# ==================================================

def test_image_similarity_download(job_id):

    print()
    print("================================")
    print("TESTING IMAGE SIMILARITY")
    print("================================")

    image1 = download_job_image(job_id)

    image2 = download_job_image_2(job_id)

    print()
    print("Image 1 size:", image1.size)
    print("Image 2 size:", image2.size)

    print("Both images downloaded successfully!")


# ==================================================
# IMAGE SIMILARITY - SSIM
# ==================================================

def calculate_ssim(image1, image2):

    # Convert both images to grayscale

    image1 = image1.convert("L")

    image2 = image2.convert("L")


    # Resize image2 to image1's size

    image2 = image2.resize(
        image1.size
    )


    # Convert PIL images to NumPy arrays

    image1_array = np.array(image1)

    image2_array = np.array(image2)


    # Calculate SSIM

    score = ssim(
        image1_array,
        image2_array,
        data_range=255
    )

    return score


# ==================================================
# IMAGE SIMILARITY - PERCEPTUAL HASH
# ==================================================

def calculate_phash(image1, image2):

    hash1 = imagehash.phash(image1)

    hash2 = imagehash.phash(image2)


    # Number of different bits

    difference = hash1 - hash2


    # pHash uses 64 bits

    similarity = 1 - (
        difference / 64
    )

    return similarity


# ==================================================
# EXECUTE IMAGE SIMILARITY
# ==================================================
def calculate_deep_similarity(image1, image2):

    # ----------------------------------------------
    # Detect GPU
    # ----------------------------------------------

    gpu_name, vram, backend = detect_gpu()

    if backend is None:

        device = "cpu"

    else:

        device = backend.lower()


    print(
        "Running deep feature extraction on:",
        device
    )


    # ----------------------------------------------
    # Preprocessing
    # ----------------------------------------------

    weights = ResNet18_Weights.DEFAULT

    transform = weights.transforms()


    input1 = transform(
        image1
    ).unsqueeze(0)

    input2 = transform(
        image2
    ).unsqueeze(0)


    # ----------------------------------------------
    # Move images to GPU
    # ----------------------------------------------

    input1 = input1.to(device)

    input2 = input2.to(device)


    # ----------------------------------------------
    # Move model to GPU
    # ----------------------------------------------

    feature_model.to(device)


    # ----------------------------------------------
    # Extract feature vectors
    # ----------------------------------------------

    with torch.no_grad():

        features1 = feature_model(
            input1
        )

        features2 = feature_model(
            input2
        )


    # ----------------------------------------------
    # Cosine similarity
    # ----------------------------------------------

    similarity = F.cosine_similarity(
        features1,
        features2
    )


    score = float(
        similarity.item()
    )


    return score

def execute_image_similarity(job_id):

    print()
    print("================================")
    print("IMAGE SIMILARITY JOB")
    print("================================")


    # ----------------------------------------------
    # Download first image
    # ----------------------------------------------

    print("Downloading first image...")

    image1 = download_job_image(
        job_id
    )

    print(
        "First image downloaded successfully!"
    )

    print(
        "First image size:",
        image1.size
    )


    # ----------------------------------------------
    # Download second image
    # ----------------------------------------------

    image2 = download_job_image_2(
        job_id
    )


    # ----------------------------------------------
    # Calculate SSIM
    # ----------------------------------------------

    print()
    print("Calculating SSIM...")

    ssim_score = calculate_ssim(
        image1,
        image2
    )

    print(
        "SSIM:",
        round(ssim_score * 100, 2),
        "%"
    )


    # ----------------------------------------------
    # Calculate perceptual hash
    # ----------------------------------------------

    print()

    print(
        "Calculating perceptual hash..."
    )

    phash_score = calculate_phash(
        image1,
        image2
    )

    print(
        "Perceptual Hash:",
        round(phash_score * 100, 2),
        "%"
    )
    print()

    print(
        "Calculating deep feature similarity..."
    )

    deep_score = calculate_deep_similarity(
        image1,
        image2
    )

    print(
        "Deep Feature Similarity:",
        round(deep_score * 100, 2),
        "%"
    )


    # ----------------------------------------------
    # Combined similarity
    # ----------------------------------------------

    overall_score = (
    ssim_score * 0.30
    +
    phash_score * 0.20
    +
    deep_score * 0.50
    )


    # ----------------------------------------------
    # Determine verdict
    # ----------------------------------------------

    if overall_score >= 0.85:

        verdict = "HIGHLY SIMILAR"

    elif overall_score >= 0.65:

        verdict = "SIMILAR"

    else:

        verdict = "DIFFERENT"


    # ----------------------------------------------
    # Print result
    # ----------------------------------------------

    print()

    print(
        "================================"
    )

    print(
        "IMAGE SIMILARITY COMPLETED"
    )

    print(
        "================================"
    )

    print(
        "SSIM:",
        round(ssim_score * 100, 2),
        "%"
    )

    print(
        "Perceptual Hash:",
        round(phash_score * 100, 2),
        "%"
    )

    print(
        "Overall Similarity:",
        round(overall_score * 100, 2),
        "%"
    )

    print(
        "Verdict:",
        verdict
    )

    print(
        "================================"
    )


    # ----------------------------------------------
    # Return structured result
    # ----------------------------------------------

    return {
        "status":
            "IMAGE_SIMILARITY_COMPLETED",
        "ssim":
            round(ssim_score, 4),
        "phash":
            round(phash_score, 4),
        "deep_feature_similarity": 
            round(deep_score,4),
        "overall_similarity":
            round(overall_score, 4),
        "similarity_percentage":
            round(
                overall_score * 100,
                2
            ),
        "verdict":
            verdict
    }


# ==================================================
# VEHICLE COLOR ESTIMATION
# ==================================================

def estimate_vehicle_color(cropped_image):

    """
    Estimate the dominant color of a vehicle crop.
    """

    # ----------------------------------------------
    # Convert PIL RGB image to OpenCV BGR
    # ----------------------------------------------

    image = cv2.cvtColor(
        np.array(cropped_image),
        cv2.COLOR_RGB2BGR
    )


    # ----------------------------------------------
    # Calculate average RGB/BGR values
    # ----------------------------------------------

    average_color = (
        image
        .mean(axis=0)
        .mean(axis=0)
    )


    blue = average_color[0]
    green = average_color[1]
    red = average_color[2]


    # ----------------------------------------------
    # Calculate brightness
    # ----------------------------------------------

    brightness = (
        red +
        green +
        blue
    ) / 3


    if brightness < 60:

        return "Black"


    if (
        brightness > 190
        and
        max(red, green, blue)
        -
        min(red, green, blue)
        < 40
    ):

        return "White"


    if (
        max(red, green, blue)
        -
        min(red, green, blue)
        < 35
    ):

        return "Gray/Silver"


    if (
        red > green * 1.3
        and
        red > blue * 1.3
    ):

        return "Red"


    if (
        blue > red * 1.2
        and
        blue > green * 1.1
    ):

        return "Blue"


    if (
        green > red * 1.2
        and
        green > blue * 1.1
    ):

        return "Green"


    return "Other"


# ==================================================
# OBJECT DETECTION + VEHICLE ANALYTICS
# ==================================================

def execute_object_detection(job_id):

    print(
        "Downloading image for object detection..."
    )

    pil_image = download_job_image(
        job_id
    )

    print(
        "Image downloaded successfully!"
    )

    print(
        "Image size:",
        pil_image.size
    )


    # -----------------------------------------
    # Select GPU
    # -----------------------------------------

    gpu_name, vram, backend = detect_gpu()

    device = backend.lower()

    print(
        "Running YOLO on:",
        device
    )


    # -----------------------------------------
    # Object Detection
    # -----------------------------------------

    results = yolo_model.predict(
        source=pil_image,
        device=device,
        verbose=False
    )

    result = results[0]


    # -----------------------------------------
    # Convert image to OpenCV format
    # -----------------------------------------

    image = np.array(
        pil_image
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )


    # -----------------------------------------
    # Detection statistics
    # -----------------------------------------

    detections = []

    object_counts = {}

    vehicle_counts = {}

    vehicle_confidences = []

    vehicle_colors = []

    total_vehicles = 0


    # -----------------------------------------
    # Process YOLO detections
    # -----------------------------------------

    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )

        class_name = result.names[
            class_id
        ]

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )


        detections.append({

            "object":
                class_name,

            "confidence":
                round(
                    confidence,
                    3
                )
        })


        # Count all detected objects

        object_counts[class_name] = (
            object_counts.get(
                class_name,
                0
            ) + 1
        )


        # -------------------------------------
        # Vehicle analytics
        # -------------------------------------

        if class_name in VEHICLE_CLASSES:

            total_vehicles += 1

            vehicle_counts[class_name] = (
                vehicle_counts.get(
                    class_name,
                    0
                ) + 1
            )

            vehicle_confidences.append(
                confidence
            )


            # Make sure coordinates stay inside image

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                image.shape[1],
                x2
            )

            y2 = min(
                image.shape[0],
                y2
            )


            if x2 > x1 and y2 > y1:

                vehicle_crop = image[
                    y1:y2,
                    x1:x2
                ]


                # Convert OpenCV crop back to PIL

                vehicle_crop_rgb = cv2.cvtColor(
                    vehicle_crop,
                    cv2.COLOR_BGR2RGB
                )

                vehicle_crop_pil = Image.fromarray(
                    vehicle_crop_rgb
                )


                color = estimate_vehicle_color(
                    vehicle_crop_pil
                )


                vehicle_colors.append({

                    "vehicle":
                        class_name,

                    "color":
                        color,

                    "confidence":
                        round(
                            confidence,
                            3
                        )
                })


    # -----------------------------------------
    # License Plate Detection
    # -----------------------------------------

    print(
        "Detecting license plates..."
    )

    plate_results = (
        license_plate_model.predict(
            source=pil_image,
            device=device,
            conf=0.25,
            verbose=False
        )
    )

    plate_result = plate_results[0]

    license_plates = []


    # -----------------------------------------
    # Blur detected license plates
    # -----------------------------------------

    if plate_result.boxes is not None:

        for box in plate_result.boxes:

            confidence = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )


            # Keep coordinates inside image

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                image.shape[1],
                x2
            )

            y2 = min(
                image.shape[0],
                y2
            )


            if x2 <= x1 or y2 <= y1:

                continue


            # ---------------------------------
            # Extract plate
            # ---------------------------------

            plate = image[
                y1:y2,
                x1:x2
            ]


            if plate.size == 0:

                continue


            # ---------------------------------
            # Apply Gaussian blur
            # ---------------------------------

            width = x2 - x1

            height = y2 - y1


            # Kernel must be odd

            kernel_width = max(
                15,
                (width // 2) * 2 + 1
            )

            kernel_height = max(
                15,
                (height // 2) * 2 + 1
            )


            blurred_plate = cv2.GaussianBlur(
                plate,
                (
                    kernel_width,
                    kernel_height
                ),
                0
            )


            image[
                y1:y2,
                x1:x2
            ] = blurred_plate


            # ---------------------------------
            # Draw license plate rectangle
            # ---------------------------------

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )


            cv2.putText(
                image,
                f"License Plate {confidence:.2f}",
                (
                    x1,
                    max(
                        20,
                        y1 - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )


            license_plates.append({

                "confidence":
                    round(
                        confidence,
                        3
                    ),

                "box": [
                    x1,
                    y1,
                    x2,
                    y2
                ],

                "blurred":
                    True
            })


    # -----------------------------------------
    # Draw object detections
    # -----------------------------------------

    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )

        class_name = result.names[
            class_id
        ]

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )


        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )


        cv2.putText(
            image,
            f"{class_name} {confidence:.2f}",
            (
                x1,
                max(
                    20,
                    y1 - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )


    # -----------------------------------------
    # Calculate average confidence
    # -----------------------------------------

    if vehicle_confidences:

        average_vehicle_confidence = (
            sum(vehicle_confidences)
            /
            len(vehicle_confidences)
        )

    else:

        average_vehicle_confidence = 0


    # -----------------------------------------
    # Save processed image
    # -----------------------------------------

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    output_path = (
        f"outputs/job_{job_id}_detected.jpg"
    )


    cv2.imwrite(
        output_path,
        image
    )


    print()

    print(
        "================================"
    )

    print(
        "OBJECT DETECTION COMPLETED"
    )

    print(
        "================================"
    )

    print(
        "Device:",
        device
    )

    print(
        "Total vehicles:",
        total_vehicles
    )

    print(
        "Vehicle counts:",
        vehicle_counts
    )

    print(
        "License plates:",
        len(license_plates)
    )

    print(
        "Output:",
        output_path
    )

    print(
        "================================"
    )


    # -----------------------------------------
    # Return structured result
    # -----------------------------------------

    return {

        "status":
            "YOLO_DETECTION_COMPLETED",

        "device":
            device,

        "gpu_name":
            gpu_name,

        "vram":
            vram,

        "backend":
            backend,

        "resolution":
            f"{pil_image.width}x{pil_image.height}",

        "total_objects":
            len(detections),

        "object_counts":
            object_counts,

        "total_vehicles":
            total_vehicles,

        "vehicle_counts":
            vehicle_counts,

        "vehicle_colors":
            vehicle_colors,

        "average_vehicle_confidence":
            round(
                average_vehicle_confidence,
                3
            ),

        "license_plate_count":
            len(license_plates),

        "license_plates":
            license_plates,

        "detections":
            detections,

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

    # ==================================================
    # GPU INFORMATION
    # ==================================================

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
    # REGISTER GPU
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

        f"{SERVER_URL}"
        f"/host/register",

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

                if (
                    job["job_type"]
                    ==
                    "OBJECT_DETECTION"
                ):

                    try:

                        # ------------------------------------------
                        # Run object detection
                        # ------------------------------------------

                        result = (
                            execute_object_detection(
                                job["id"]
                            )
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
                        # UPLOAD RESULT
                        # ==================================================

                        output_path = (
                            result[
                                "output_path"
                            ]
                        )


                        print(
                            "Uploading YOLO result..."
                        )


                        with open(
                            output_path,
                            "rb"
                        ) as file:

                            upload_response = requests.post(

                                f"{SERVER_URL}/host/jobs/"
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

                            f"{SERVER_URL}/host/jobs/"
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

                                f"{SERVER_URL}/host/jobs/"
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
                # IMAGE SIMILARITY
                # ==================================================

                elif (
                    job["job_type"]
                    ==
                    "IMAGE_SIMILARITY"
                ):

                    try:

                        # ------------------------------------------
                        # Run image similarity
                        # ------------------------------------------

                        result = (
                            execute_image_similarity(
                                job["id"]
                            )
                        )


                        print()

                        print(
                            "Job Result:"
                        )

                        print(
                            result
                        )


                        # ------------------------------------------
                        # Mark job completed
                        # ------------------------------------------

                        completion_data = {

                            "status":
                                "COMPLETED",

                            "result":
                                result
                        }


                        completion_response = requests.post(

                            f"{SERVER_URL}/host/jobs/"
                            f"{job['id']}"
                            f"/complete",

                            json=completion_data
                        )


                        completion_response.raise_for_status()


                        print()

                        print(
                            "Server completion response:"
                        )

                        print(
                            completion_response.json()
                        )


                    except Exception as error:

                        print()

                        print(
                            "Image similarity job failed!"
                        )

                        print(
                            "Error:",
                            error
                        )


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

                                f"{SERVER_URL}/host/jobs/"
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
        # WAIT
        # ==================================================

        time.sleep(2)