import numpy as np
import imageio.v3 as iio

def prepare(image_path, size_out=512, bg_value=255):
    
    """
    Load an image, convert to greyscale, circular-crop, and save a size_out x size_out
    greyscale image. Outside the circle is set to bg_value (0 to 255).
    """
    
    # Read
    img = iio.imread(image_path)

    # To grayscale (luminance)
    if img.ndim == 3:          # RGB or RGBA
        rgb = img[..., :3].astype(np.float32)
        gray = 0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]
    else:                      # already grayscale
        gray = img.astype(np.float32)

    # Normalize to 0..255 if needed
    if gray.max() <= 1.0:
        gray = gray * 255.0
    gray = np.clip(gray, 0, 255)

    # Center-crop to square
    h, w = gray.shape[:2]
    s = min(h, w)
    top  = (h - s) // 2
    left = (w - s) // 2
    square = gray[top:top+s, left:left+s]

    # Nearest-neighbor resize to size_out x size_out 
    # Map output pixel centers to source indices
    y_out, x_out = np.ogrid[:size_out, :size_out]
    y_src = (y_out * s / size_out).astype(int)
    x_src = (x_out * s / size_out).astype(int)
    y_src[y_src >= s] = s - 1
    x_src[x_src >= s] = s - 1
    img_resized = square[y_src, x_src].astype(np.uint8)

    # Circular mask in output grid
    yy, xx = np.ogrid[:size_out, :size_out]
    c = (size_out - 1) / 2.0
    r = size_out / 2.0
    mask = (xx - c)**2 + (yy - c)**2 <= r**2

    # Apply mask: outside circle -> bg_value
    out = np.full((size_out, size_out), int(bg_value), dtype=np.uint8)
    out[mask] = img_resized[mask]

    iio.imwrite("image_start.png", out)


    return out
