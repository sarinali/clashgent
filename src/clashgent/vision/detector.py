"""Object detection utilities and interfaces."""

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import torch


@dataclass
class BoundingBox:
    """Axis-aligned bounding box.

    Attributes:
        x1: Left edge (pixels or normalized)
        y1: Top edge (pixels or normalized)
        x2: Right edge (pixels or normalized)
        y2: Bottom edge (pixels or normalized)
        confidence: Detection confidence (0.0 to 1.0)
        class_id: Integer class identifier
        class_name: Optional string class name
    """
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    class_id: int = 0
    class_name: Optional[str] = None

    @property
    def width(self) -> float:
        """Box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Box area."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Box center point."""
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def iou(self, other: "BoundingBox") -> float:
        """Compute Intersection over Union with another box.

        Args:
            other: Another BoundingBox

        Returns:
            IoU value between 0.0 and 1.0
        """
        # Intersection
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0

    def to_normalized(self, img_width: int, img_height: int) -> "BoundingBox":
        """Convert pixel coordinates to normalized (0-1).

        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            New BoundingBox with normalized coordinates
        """
        return BoundingBox(
            x1=self.x1 / img_width,
            y1=self.y1 / img_height,
            x2=self.x2 / img_width,
            y2=self.y2 / img_height,
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
        )

    def to_pixels(self, img_width: int, img_height: int) -> "BoundingBox":
        """Convert normalized coordinates to pixels.

        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            New BoundingBox with pixel coordinates
        """
        return BoundingBox(
            x1=self.x1 * img_width,
            y1=self.y1 * img_height,
            x2=self.x2 * img_width,
            y2=self.y2 * img_height,
            confidence=self.confidence,
            class_id=self.class_id,
            class_name=self.class_name,
        )


class Detector(Protocol):
    """Protocol for object detectors.

    Any detector implementation should follow this interface.
    """

    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        """Detect objects in an image.

        Args:
            image: RGB image array (H, W, 3)

        Returns:
            List of detected bounding boxes
        """
        ...

    def detect_batch(self, images: np.ndarray) -> list[list[BoundingBox]]:
        """Detect objects in a batch of images.

        Args:
            images: Batch of RGB images (B, H, W, 3)

        Returns:
            List of detection lists, one per image
        """
        ...


def non_max_suppression(
    boxes: list[BoundingBox],
    iou_threshold: float = 0.5,
) -> list[BoundingBox]:
    """Apply non-maximum suppression to filter overlapping detections.

    Keeps the highest confidence box when boxes overlap significantly.

    Args:
        boxes: List of candidate bounding boxes
        iou_threshold: IoU threshold above which to suppress

    Returns:
        Filtered list of bounding boxes
    """
    if not boxes:
        return []

    # Sort by confidence descending
    sorted_boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)

    keep = []
    while sorted_boxes:
        # Keep the highest confidence box
        best = sorted_boxes.pop(0)
        keep.append(best)

        # Remove boxes with high IoU with the kept box
        sorted_boxes = [
            box for box in sorted_boxes
            if best.iou(box) < iou_threshold
        ]

    return keep


def non_max_suppression_class_aware(
    boxes: list[BoundingBox],
    iou_threshold: float = 0.5,
) -> list[BoundingBox]:
    """Apply NMS separately for each class.

    Args:
        boxes: List of candidate bounding boxes
        iou_threshold: IoU threshold above which to suppress

    Returns:
        Filtered list of bounding boxes
    """
    # Group by class
    class_boxes: dict[int, list[BoundingBox]] = {}
    for box in boxes:
        if box.class_id not in class_boxes:
            class_boxes[box.class_id] = []
        class_boxes[box.class_id].append(box)

    # Apply NMS per class
    keep = []
    for class_id, class_box_list in class_boxes.items():
        keep.extend(non_max_suppression(class_box_list, iou_threshold))

    return keep


def crop_region(
    image: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
    normalized: bool = True,
) -> np.ndarray:
    """Crop a region from an image.

    Args:
        image: Input image (H, W, 3) or (H, W)
        x: Left edge of crop region
        y: Top edge of crop region
        width: Width of crop region
        height: Height of crop region
        normalized: If True, coordinates are normalized (0-1)

    Returns:
        Cropped image region
    """
    img_h, img_w = image.shape[:2]

    if normalized:
        x1 = int(x * img_w)
        y1 = int(y * img_h)
        x2 = int((x + width) * img_w)
        y2 = int((y + height) * img_h)
    else:
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + width), int(y + height)

    # Clamp to image bounds
    x1 = max(0, min(x1, img_w))
    x2 = max(0, min(x2, img_w))
    y1 = max(0, min(y1, img_h))
    y2 = max(0, min(y2, img_h))

    return image[y1:y2, x1:x2]


def preprocess_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (416, 416),
    normalize: bool = True,
) -> torch.Tensor:
    """Preprocess image for neural network input.

    Args:
        image: RGB image (H, W, 3) with uint8 values
        target_size: Output size (width, height)
        normalize: Whether to normalize to [0, 1]

    Returns:
        Preprocessed tensor (1, 3, H, W)
    """
    import cv2

    # Resize
    resized = cv2.resize(image, target_size)

    # Convert to float and normalize
    if normalize:
        tensor = resized.astype(np.float32) / 255.0
    else:
        tensor = resized.astype(np.float32)

    # HWC to CHW
    tensor = np.transpose(tensor, (2, 0, 1))

    # Add batch dimension
    tensor = np.expand_dims(tensor, 0)

    return torch.from_numpy(tensor)
