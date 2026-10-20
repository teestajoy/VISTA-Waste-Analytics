import math

from PIL import Image

from vista.detection.object_detector import Detection


def crop_detection(image: Image.Image, detection: Detection) -> Image.Image:
    """Return the part of the original image covered by a detection box."""
    box = detection.box
    coordinates = (box.x_min, box.y_min, box.x_max, box.y_max)

    try:
        x_min, y_min, x_max, y_max = (float(value) for value in coordinates)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Bounding box coordinates must be finite numbers.") from error

    if not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max)):
        raise ValueError("Bounding box coordinates must be finite numbers.")
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Bounding box must have positive width and height.")

    left = max(0, min(image.width, math.floor(x_min)))
    top = max(0, min(image.height, math.floor(y_min)))
    right = max(0, min(image.width, math.ceil(x_max)))
    bottom = max(0, min(image.height, math.ceil(y_max)))

    if right <= left or bottom <= top:
        raise ValueError("Bounding box has no visible area inside the image.")

    crop = image.crop((left, top, right, bottom))
    if crop.width == 0 or crop.height == 0:
        raise ValueError("Bounding box produced an empty crop.")
    return crop
