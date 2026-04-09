# modules/ela.py

import os
from PIL import Image, ImageChops, ImageEnhance
import numpy as np

def perform_ela(image_path, quality=90):
    """
    Performs Error Level Analysis on an image.
    
    How it works:
    1. Open original image
    2. Re-save it at lower quality (this re-compresses it)
    3. Find pixel-by-pixel difference between original and re-saved
    4. Amplify that difference so we can see it
    5. Bright areas in result = possibly tampered
    """
    
    # Step 1: Open the original image and convert to RGB
    # (some images are RGBA with transparency - we remove that)
    original = Image.open(image_path).convert('RGB')
    
    # Step 2: Save a temporary compressed version
    temp_path = 'uploads/temp_ela.jpg'
    original.save(temp_path, 'JPEG', quality=quality)
    
    # Step 3: Open the compressed version
    compressed = Image.open(temp_path)
    
    # Step 4: Find the difference between original and compressed
    # ImageChops.difference() subtracts pixel values
    # If no editing happened → small difference
    # If editing happened → large difference in that region
    ela_image = ImageChops.difference(original, compressed)
    
    # Step 5: Amplify the differences so they're visible
    # Find the maximum difference value in the image
    extrema = ela_image.getextrema()  # returns (min, max) for each channel
    max_diff = max([channel[1] for channel in extrema])
    
    # Avoid division by zero
    if max_diff == 0:
        max_diff = 1
    
    # Scale factor: stretch the differences to full 0-255 range
    scale = 255.0 / max_diff
    
    # Apply brightness enhancement using scale
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    # Step 6: Calculate a "suspicion score" from ELA
    # Convert to numpy array for math operations
    ela_array = np.array(ela_image)
    
    # Mean brightness of ELA image = how much tampering
    # Higher mean = more suspicious
    ela_mean = np.mean(ela_array)
    
    # Normalize to 0-100 range for display
    # Typical untampered images have ela_mean < 20
    # Tampered images have ela_mean > 40
    suspicion_score = min(100, (ela_mean / 128) * 100)
    
    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return ela_image, round(suspicion_score, 2)


def get_ela_regions(ela_image):
    """
    Finds the most suspicious (brightest) regions in ELA image.
    Returns bounding boxes of suspicious areas.
    """
    import cv2
    
    # Convert PIL image to OpenCV format
    ela_array = np.array(ela_image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY)
    
    # Threshold: keep only very bright pixels (suspicious areas)
    # 200 = threshold value, 255 = replace with white
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours (outlines) of suspicious regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes of suspicious regions
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Ignore tiny noise regions
            x, y, w, h = cv2.boundingRect(contour)
            regions.append((x, y, w, h))
    
    return regions