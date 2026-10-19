from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


class ImageInputError(ValueError):
    """Raised when a file exists but is not a readable image."""


@dataclass(frozen=True)
class ImageMetadata:
    width: int
    height: int
    format: str


def validate_image(path: str | Path) -> ImageMetadata:
    """Open and fully load an image, returning its basic metadata."""
    image_path = Path(path)

    try:
        with Image.open(image_path) as image:
            image.load()
            return ImageMetadata(
                width=image.width,
                height=image.height,
                format=image.format or "",
            )
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Image file not found: {image_path}") from error
    except (UnidentifiedImageError, OSError) as error:
        raise ImageInputError(f"File is not a supported, readable image: {image_path}") from error
