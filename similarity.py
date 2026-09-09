from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
import imagehash

def calculate_ssim(image1, image2):

    # Convert both images to grayscale
    image1 = image1.convert("L")
    image2 = image2.convert("L")

    # Resize image2 to image1's size
    image2 = image2.resize(image1.size)

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

def calculate_phash(image1, image2):

    hash1 = imagehash.phash(image1)
    hash2 = imagehash.phash(image2)

    difference = hash1 - hash2
    similarity = 1 - (difference / 64)

    return similarity


if __name__ == "__main__":

    image1 = Image.open("image1.jpg").convert("RGB")
    image2 = Image.open("image2.jpg").convert("RGB")

    # SSIM
    ssim_score = calculate_ssim(image1, image2)

    # Perceptual Hash
    phash_score = calculate_phash(image1, image2)

    # Combined score
    overall_score = (
        ssim_score * 0.5
        +
        phash_score * 0.5
    )

    print()
    print("================================")
    print("IMAGE SIMILARITY TEST")
    print("================================")

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

    if overall_score >= 0.85:
        verdict = "HIGHLY SIMILAR"
    elif overall_score >= 0.65:
        verdict = "SIMILAR"
    else:
        verdict = "DIFFERENT"

    print("Verdict:", verdict)

    print("================================")