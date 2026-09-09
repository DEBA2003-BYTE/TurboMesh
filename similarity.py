from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np


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


if __name__ == "__main__":

    image1 = Image.open("image1.jpg").convert("RGB")
    image2 = Image.open("image2.jpg").convert("RGB")

    score = calculate_ssim(image1, image2)

    print()
    print("================================")
    print("IMAGE SIMILARITY TEST")
    print("================================")
    print("SSIM Score:", round(score, 4))
    print("Similarity:", round(score * 100, 2), "%")
    print("================================")