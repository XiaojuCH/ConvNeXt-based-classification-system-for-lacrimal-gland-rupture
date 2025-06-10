# estimate_ring_params.py
import cv2
import numpy as np
import glob

def estimate_ring_params(img_paths,
                         rad_min=0.30,    # 调低到 0.30
                         rad_max=0.90,    # 调高到 0.90
                         nbins=100):
    all_rad = []
    for p in img_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        H, W = img.shape
        cx, cy = W/2, H/2

        # Canny 边缘
        edges = cv2.Canny(img, 50, 150)

        ys, xs = np.where(edges>0)
        dists = np.sqrt((xs-cx)**2 + (ys-cy)**2) / (min(H,W)/2)

        mask = (dists > rad_min) & (dists < rad_max)
        all_rad.append(dists[mask])

    all_rad = np.concatenate(all_rad)
    hist, bins = np.histogram(all_rad,
                              bins=nbins,
                              range=(rad_min, rad_max),
                              density=True)
    centers = (bins[:-1] + bins[1:]) / 2

    # 峰值
    peak = np.argmax(hist)
    ring_center = centers[peak]

    # 半峰宽度
    half_h = hist[peak] / 2
    left  = np.where(hist[:peak] < half_h)[0]
    right = np.where(hist[peak:] < half_h)[0]
    if left.size and right.size:
        l = left[-1]
        r = peak + right[0]
        ring_width = centers[r] - centers[l]
    else:
        ring_width = (rad_max - rad_min) * 0.3
    return ring_center, ring_width

if __name__ == "__main__":
    imgs = glob.glob("image/images/**/*.png", recursive=True)
    c, w = estimate_ring_params(imgs)
    print(f"Estimated RING_CENTER={c:.3f}, RING_WIDTH≈{w:.3f}")
