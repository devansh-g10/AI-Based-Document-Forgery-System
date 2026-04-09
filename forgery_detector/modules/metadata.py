# modules/metadata.py

import exifread
import os
from datetime import datetime

# List of software that indicates editing/manipulation
EDITING_SOFTWARE = [
    'photoshop', 'gimp', 'paint', 'paintshop',
    'illustrator', 'inkscape', 'pixelmator',
    'affinity', 'canva', 'picsart', 'snapseed'
]

def extract_metadata(image_path):
    """
    Extracts EXIF metadata from image.
    Returns a dictionary of all metadata found.
    """
    metadata = {}
    
    try:
        with open(image_path, 'rb') as f:
            # exifread reads binary EXIF data from image files
            tags = exifread.process_file(f, details=False)
        
        # Convert tags to readable dictionary
        for tag, value in tags.items():
            # Skip thumbnail data (too long)
            if 'thumbnail' not in tag.lower():
                metadata[tag] = str(value)
                
    except Exception as e:
        metadata['error'] = str(e)
    
    return metadata


def analyze_metadata(image_path):
    """
    Analyzes metadata for signs of forgery.
    Returns list of suspicious findings and a risk level.
    """
    metadata = extract_metadata(image_path)
    suspicious_findings = []
    risk_score = 0  # 0 = safe, 100 = very suspicious
    
    # ---- Check 1: Was editing software used? ----
    software_tag = metadata.get('Image Software', '').lower()
    
    if software_tag:
        for editor in EDITING_SOFTWARE:
            if editor in software_tag:
                suspicious_findings.append(
                    f"⚠️ Edited with: {software_tag.title()}"
                )
                risk_score += 40  # Major red flag
                break
    
    # ---- Check 2: Date/Time inconsistencies ----
    # Original date should match modification date
    original_date = metadata.get('EXIF DateTimeOriginal', '')
    modified_date = metadata.get('Image DateTime', '')
    
    if original_date and modified_date:
        if original_date != modified_date:
            suspicious_findings.append(
                f"⚠️ Date mismatch: Created {original_date} | Modified {modified_date}"
            )
            risk_score += 25
    
    # ---- Check 3: Missing metadata (suspicious for scanned docs) ----
    important_fields = ['Image Make', 'Image Model', 'EXIF DateTimeOriginal']
    missing_count = sum(1 for field in important_fields if field not in metadata)
    
    if missing_count == len(important_fields):
        suspicious_findings.append(
            "⚠️ No camera/device metadata found — could be digitally created"
        )
        risk_score += 15
    
    # ---- Check 4: GPS data present? ----
    # Documents shouldn't have GPS data — suspicious if present
    if 'GPS GPSLatitude' in metadata:
        suspicious_findings.append(
            "⚠️ GPS location data embedded — unusual for documents"
        )
        risk_score += 10
    
    # Cap risk score at 100
    risk_score = min(100, risk_score)
    
    # Determine risk level
    if risk_score == 0:
        risk_level = "LOW"
    elif risk_score < 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"
    
    return {
        'findings': suspicious_findings,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'raw_metadata': metadata
    }