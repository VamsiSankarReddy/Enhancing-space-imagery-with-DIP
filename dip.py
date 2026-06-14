import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ---------- Manual DIP Function Implementations ----------

def log_transform(img):
    img_f = img.astype(np.float64)
    c = 255.0 / np.log(1.0 + img_f.max())
    out = c * np.log(1.0 + img_f)
    return np.clip(out, 0, 255).astype(np.uint8)


def gamma_transform(img, gamma=2.0):
    img_f = img.astype(np.float64)
    c = 255.0 / (img_f.max() ** gamma)
    out = c * np.power(img_f, gamma)
    return np.clip(out, 0, 255).astype(np.uint8)


def histogram_equalization(img):
    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()

    # Avoid divide-by-zero
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype(np.uint8)

    out_gray = cdf_final[gray]

    return np.stack([out_gray] * 3, axis=-1).astype(np.uint8)


def contrast_stretching(img, in_low=0, in_high=255,
                        out_low=0, out_high=255):

    img_f = img.astype(np.float64)

    out = (img_f - in_low) * (
        (out_high - out_low) / (in_high - in_low)
    ) + out_low

    return np.clip(out, 0, 255).astype(np.uint8)


def dft_transform(img):
    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = np.log(1 + np.abs(fshift))

    mag_norm = 255 * (magnitude / magnitude.max())

    return np.stack([mag_norm] * 3, axis=-1).astype(np.uint8)


def geometric_transform(img, tx=0, ty=0, angle=0, scale=1.0):

    rows, cols = img.shape[:2]

    # Translation matrix
    M_t = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float64)

    # Rotation + scaling matrix
    theta = np.deg2rad(angle)

    M_r = np.array([
        [scale * np.cos(theta), -scale * np.sin(theta), 0],
        [scale * np.sin(theta),  scale * np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=np.float64)

    M = M_r @ M_t

    # Coordinate mapping
    y_coords, x_coords = np.indices((rows, cols))

    coords = np.vstack([
        x_coords.ravel(),
        y_coords.ravel(),
        np.ones(rows * cols)
    ])

    mapped = M @ coords

    x = np.round(mapped[0]).astype(int)
    y = np.round(mapped[1]).astype(int)

    out = np.zeros_like(img)

    valid = (
        (x >= 0) & (x < cols) &
        (y >= 0) & (y < rows)
    )

    out[y[valid], x[valid]] = img[
        y_coords.ravel()[valid],
        x_coords.ravel()[valid]
    ]

    return out


# ---------- Additional Transformations ----------

def negative_transform(img):
    return 255 - img


def thresholding(img, thresh=128):

    gray = np.dot(img[..., :3],
                  [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    out = np.where(gray >= thresh, 255, 0).astype(np.uint8)

    return np.stack([out] * 3, axis=-1)


def laplacian_edge_detection(img):

    gray = np.dot(img[..., :3],
                  [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    # Optional blur for better edge detection
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    gray = gray.astype(np.int32)

    K = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ])

    rows, cols = gray.shape

    out = np.zeros_like(gray)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):

            region = gray[i - 1:i + 2, j - 1:j + 2]

            out[i, j] = np.sum(region * K)

    out = np.clip(np.abs(out), 0, 255).astype(np.uint8)

    return np.stack([out] * 3, axis=-1)


# ---------- Main Program ----------

def main():

    root = tk.Tk()
    root.withdraw()

    path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[
            ("Image Files", "*.jpg;*.jpeg;*.png;*.tif;*.bmp")
        ]
    )

    if not path:
        print("No file selected. Exiting.")
        return

    bgr = cv2.imread(path)

    if bgr is None:
        print("Failed to load image. Exiting.")
        return

    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Resize large images for easier display
    max_size = 800

    h, w = img.shape[:2]

    if max(h, w) > max_size:
        scale = max_size / max(h, w)

        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale))
        )

    print("Loaded image:", img.shape, img.dtype)

    print("\nChoose the transform:")

    options = [
        "1) Log Transform",
        "2) Gamma Transform",
        "3) Histogram Equalization",
        "4) Contrast Stretching",
        "5) DFT Spectrum",
        "6) Geometric Transform",
        "7) Negative Transform",
        "8) Thresholding",
        "9) Laplacian Edge Detection"
    ]

    for opt in options:
        print(opt)

    try:
        ch = int(input("\nEnter choice [1-9]: "))

    except ValueError:
        print("Invalid input.")
        return

    # ---------- Processing ----------

    if ch == 1:

        result = log_transform(img)
        title = "Log Transform"

    elif ch == 2:

        gamma = float(input("Enter gamma value: "))

        result = gamma_transform(img, gamma)
        title = f"Gamma Transform (γ={gamma})"

    elif ch == 3:

        result = histogram_equalization(img)
        title = "Histogram Equalization"

    elif ch == 4:

        result = contrast_stretching(img)
        title = "Contrast Stretching"

    elif ch == 5:

        result = dft_transform(img)
        title = "DFT Spectrum"

    elif ch == 6:

        tx = int(input("Translation X (tx): "))
        ty = int(input("Translation Y (ty): "))

        angle = float(input("Rotation angle (degrees): "))

        scale = float(input("Scaling factor: "))

        result = geometric_transform(
            img,
            tx,
            ty,
            angle,
            scale
        )

        title = "Geometric Transform"

    elif ch == 7:

        result = negative_transform(img)
        title = "Negative Transform"

    elif ch == 8:

        thresh = int(input("Threshold value [0-255]: "))

        result = thresholding(img, thresh)

        title = f"Thresholding ({thresh})"

    elif ch == 9:

        result = laplacian_edge_detection(img)
        title = "Laplacian Edge Detection"

    else:
        print("Invalid choice.")
        return

    # ---------- Display ----------

    print(
        "\nResult stats:",
        "dtype =", result.dtype,
        "min =", result.min(),
        "max =", result.max()
    )

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.axis("off")
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.title(title)
    plt.axis("off")
    plt.imshow(result)

    plt.tight_layout()
    plt.show()

    # ---------- Save Result ----------

    save = input("\nSave result image? (y/n): ")

    if save.lower() == 'y':

        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg")
            ]
        )

        if save_path:

            cv2.imwrite(
                save_path,
                cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            )

            print("Image saved successfully!")


# ---------- Run Program ----------

if __name__ == "__main__":
    main()