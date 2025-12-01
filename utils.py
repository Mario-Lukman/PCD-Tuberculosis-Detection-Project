"""
Utility functions for image preprocessing, lung segmentation, and feature extraction.
"""
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from config import IMAGE_SIZE, GLCM_DISTANCES, GLCM_ANGLES, LBP_P, LBP_R


def preprocess_image(image_path):
    """Load image, resize, convert to grayscale and apply CLAHE for contrast."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.resize(img, IMAGE_SIZE)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img).astype(np.uint8)


def segment_lung(preprocessed_image):
    """Segment lung region using Otsu thresholding, morphological ops and contour filtering."""
    img = preprocessed_image.copy()
    h, w = img.shape
    
    # Smooth
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Otsu threshold & Invert
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    
    # Remove border
    border = int(min(h, w) * 0.08)
    thresh[:border, :] = 0
    thresh[-border:, :] = 0
    thresh[:, :border] = 0
    thresh[:, -border:] = 0
    
    # Morphological operations
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_med = np.ones((5, 5), np.uint8)
    
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small, iterations=3)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_med, iterations=3)
    eroded = cv2.erode(closed, kernel_small, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(eroded)
    if contours:
        img_area = h * w
        valid = []
        
        for c in contours:
            area = cv2.contourArea(c)
            # Filter by area (2-35% of image)
            if 0.02 * img_area < area < 0.35 * img_area:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Filter by position (central 70%)
                    if 0.15 * w < cx < 0.85 * w and 0.15 * h < cy < 0.85 * h:
                        x, y, w_box, h_box = cv2.boundingRect(c)
                        # Filter by aspect ratio
                        if max(w_box, h_box) / (min(w_box, h_box) + 1e-6) < 3:
                            valid.append((area, c))
        
        # Take top 2 largest valid contours
        valid.sort(reverse=True, key=lambda x: x[0])
        for _, c in valid[:2]:
            cv2.drawContours(mask, [c], -1, 255, -1)
            
    lung_mask = (mask > 0).astype(np.uint8)
    
    segmented = img.copy()
    segmented[lung_mask == 0] = 0
    
    return segmented, lung_mask


def get_glcm_features(segmented_lung, lung_mask):
    """Compute GLCM texture features (contrast, correlation, energy, homogeneity)."""
    masked = segmented_lung.copy()
    masked[lung_mask == 0] = 0
    
    glcm = graycomatrix(
        masked.astype(np.uint8), 
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES, 
        levels=256, 
        symmetric=True, 
        normed=True
    )
    
    feats = []
    for prop in ["contrast", "correlation", "energy", "homogeneity"]:
        feats.extend(graycoprops(glcm, prop)[0])
        
    return np.array(feats, dtype=np.float32)


def compute_lbp(image, P=LBP_P, R=LBP_R):
    """Compute Local Binary Pattern for a grayscale image."""
    h, w = image.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    
    offsets = [
        (0, R), (-R, R), (-R, 0), (-R, -R),
        (0, -R), (R, -R), (R, 0), (R, R)
    ]
    
    for y in range(R, h - R):
        for x in range(R, w - R):
            center = image[y, x]
            code = 0
            for i, (dy, dx) in enumerate(offsets):
                neighbor = image[y + dy, x + dx]
                if neighbor >= center:
                    code |= (1 << i)
            lbp[y, x] = code
            
    return lbp


def compute_lbp_histogram(lbp_image, mask=None):
    """Return normalized histogram (256 bins) of LBP values."""
    if mask is not None:
        values = lbp_image[mask == 1]
    else:
        values = lbp_image.flatten()
        
    hist, _ = np.histogram(values, bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    
    return hist


def get_lbp_features(segmented_lung, lung_mask):
    """Compute LBP image and its histogram for the segmented lung region."""
    lbp_img = compute_lbp(segmented_lung)
    lbp_hist = compute_lbp_histogram(lbp_img, lung_mask)
    return lbp_img, lbp_hist


def extract_features(image_path):
    """Full pipeline: preprocess, segment, extract LBP and GLCM features."""
    img = preprocess_image(image_path)
    seg, mask = segment_lung(img)
    _, lbp_hist = get_lbp_features(seg, mask)
    glcm_vec = get_glcm_features(seg, mask)
    
    return np.concatenate([lbp_hist, glcm_vec])
