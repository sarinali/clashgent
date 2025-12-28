"""YOLO-style neural network models for game object detection."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Detection:
    """Single object detection result.

    Attributes:
        class_id: Integer class index
        confidence: Detection confidence (0.0 to 1.0)
        x: Center x coordinate (normalized 0.0 to 1.0)
        y: Center y coordinate (normalized 0.0 to 1.0)
        w: Width (normalized 0.0 to 1.0)
        h: Height (normalized 0.0 to 1.0)
    """
    class_id: int
    confidence: float
    x: float  # center x (0-1)
    y: float  # center y (0-1)
    w: float  # width (0-1)
    h: float  # height (0-1)

    def to_xyxy(self, img_width: int, img_height: int) -> tuple[int, int, int, int]:
        """Convert to pixel coordinates (x1, y1, x2, y2).

        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            Tuple of (x1, y1, x2, y2) pixel coordinates
        """
        cx, cy = self.x * img_width, self.y * img_height
        w, h = self.w * img_width, self.h * img_height
        return (
            int(cx - w / 2),
            int(cy - h / 2),
            int(cx + w / 2),
            int(cy + h / 2),
        )


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + LeakyReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class YOLOBackbone(nn.Module):
    """Darknet-style backbone for feature extraction.

    A simplified backbone inspired by YOLO architectures.
    Can be replaced with pretrained backbones (ResNet, CSPNet, etc.)

    Output: Multi-scale feature maps for detection.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # Downsampling stages
        self.stage1 = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64, stride=2),  # /2
        )

        self.stage2 = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 128, stride=2),  # /4
        )

        self.stage3 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 256, stride=2),  # /8
        )

        self.stage4 = nn.Sequential(
            ConvBlock(256, 256),
            ConvBlock(256, 512, stride=2),  # /16
        )

        self.stage5 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 1024, stride=2),  # /32
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            List of feature maps at different scales
        """
        x1 = self.stage1(x)    # /2
        x2 = self.stage2(x1)   # /4
        x3 = self.stage3(x2)   # /8
        x4 = self.stage4(x3)   # /16
        x5 = self.stage5(x4)   # /32

        return [x3, x4, x5]  # Return last 3 scales for detection


class ClashDetector(nn.Module):
    """YOLO-style single-shot detector for Clash Royale objects.

    Detects:
    - Troops (knights, archers, giants, etc.)
    - Spells (fireball, arrows, etc.)
    - Buildings (tesla, cannon, etc.)

    Uses grid-based prediction where each cell predicts:
    - B bounding boxes with (x, y, w, h, confidence)
    - Class probabilities for each box

    Attributes:
        num_classes: Number of object classes to detect
        num_boxes: Number of boxes to predict per grid cell
        backbone: Feature extraction network
    """

    def __init__(
        self,
        num_classes: int = 100,
        num_boxes: int = 3,
    ):
        """Initialize detector.

        Args:
            num_classes: Number of troop/spell/building types
            num_boxes: Anchor boxes per grid cell
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes

        self.backbone = YOLOBackbone()

        # Detection head
        # Output channels per cell: B * (5 + num_classes)
        # 5 = x_offset, y_offset, w, h, objectness
        out_channels = num_boxes * (5 + num_classes)

        self.detect_head = nn.Sequential(
            ConvBlock(1024, 512),
            ConvBlock(512, 512),
            nn.Conv2d(512, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run detection forward pass.

        Args:
            x: Input image tensor (B, 3, H, W)

        Returns:
            Raw predictions tensor (B, num_boxes*(5+C), grid_h, grid_w)
        """
        features = self.backbone(x)
        predictions = self.detect_head(features[-1])  # Use deepest features
        return predictions

    def decode_predictions(
        self,
        predictions: torch.Tensor,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> list[list[Detection]]:
        """Decode raw predictions to Detection objects.

        Args:
            predictions: Raw model output (B, C, H, W)
            conf_threshold: Minimum confidence to keep
            nms_threshold: IoU threshold for NMS

        Returns:
            List of Detection lists, one per batch item
        """
        # TODO: Implement prediction decoding
        # 1. Reshape predictions to (B, H, W, num_boxes, 5+num_classes)
        # 2. Apply sigmoid to objectness and class scores
        # 3. Convert offsets to absolute coordinates
        # 4. Filter by confidence threshold
        # 5. Apply non-maximum suppression

        raise NotImplementedError(
            "Detection decoding not yet implemented. "
            "Implement sigmoid activations, coordinate conversion, and NMS."
        )


class CardClassifier(nn.Module):
    """Classifier for cards in the player's hand.

    Cards appear at fixed positions in the UI, so we can crop
    those regions and classify each independently.

    Attributes:
        num_card_types: Number of different cards to classify
        card_positions: Normalized positions of card slots
    """

    # Normalized card slot positions (x, y, width, height)
    # These should be calibrated to your specific screen layout
    CARD_REGIONS = [
        (0.11, 0.83, 0.12, 0.14),  # Card 0 (leftmost)
        (0.31, 0.83, 0.12, 0.14),  # Card 1
        (0.51, 0.83, 0.12, 0.14),  # Card 2
        (0.71, 0.83, 0.12, 0.14),  # Card 3 (rightmost)
        (0.88, 0.75, 0.08, 0.10),  # Next card (smaller, top right)
    ]

    def __init__(self, num_card_types: int = 100):
        """Initialize card classifier.

        Args:
            num_card_types: Number of card types to classify
        """
        super().__init__()
        self.num_card_types = num_card_types

        # Simple CNN for card classification
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_card_types),
        )

    def forward(self, card_crops: torch.Tensor) -> torch.Tensor:
        """Classify card images.

        Args:
            card_crops: Batch of card images (B, 3, H, W)

        Returns:
            Class logits (B, num_card_types)
        """
        features = self.features(card_crops)
        logits = self.classifier(features)
        return logits

    def classify_hand(self, screenshot: torch.Tensor) -> torch.Tensor:
        """Classify all cards in hand from full screenshot.

        Args:
            screenshot: Full game screenshot (B, 3, H, W)

        Returns:
            Logits for all 5 card slots (B, 5, num_card_types)
        """
        # TODO: Implement card region cropping and classification
        # 1. Extract card regions using CARD_REGIONS
        # 2. Resize crops to fixed size
        # 3. Run classifier on each crop
        # 4. Stack results

        raise NotImplementedError(
            "Card hand classification not yet implemented. "
            "Implement region cropping and batch classification."
        )


# make this simpler using OCR
class ElixirDetector(nn.Module):
    """Detector for elixir bar value.

    The elixir bar is at a fixed position and can be detected by:
    - Color segmentation (purple bar)
    - Regression from cropped region
    - OCR on the number display

    This implementation uses regression on the bar region.
    """

    # Elixir bar region (normalized coordinates)
    BAR_REGION = (0.08, 0.80, 0.58, 0.025)  # x, y, width, height

    def __init__(self):
        super().__init__()

        # Simple CNN for elixir regression
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 10)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 10, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output 0-1, multiply by 10
        )

    def forward(self, elixir_crop: torch.Tensor) -> torch.Tensor:
        """Predict elixir value from bar crop.

        Args:
            elixir_crop: Cropped elixir bar region (B, 3, H, W)

        Returns:
            Elixir values (B, 1) in range [0, 10]
        """
        features = self.features(elixir_crop)
        elixir = self.regressor(features) * 10.0
        return elixir

    def detect_from_screenshot(self, screenshot: torch.Tensor) -> torch.Tensor:
        """Detect elixir from full screenshot.

        Args:
            screenshot: Full game screenshot (B, 3, H, W)

        Returns:
            Elixir values (B, 1)
        """
        # TODO: Implement bar region cropping and detection
        raise NotImplementedError(
            "Elixir detection from screenshot not yet implemented. "
            "Implement region cropping using BAR_REGION."
        )


class ClashVisionModel(nn.Module):
    """Combined vision model for complete game state extraction.

    Integrates all detection components:
    - Troop detection (YOLO-style)
    - Card classification
    - Elixir detection

    Attributes:
        troop_detector: ClashDetector for troop detection
        card_classifier: CardClassifier for hand cards
        elixir_detector: ElixirDetector for elixir bar
    """

    def __init__(self, num_classes: int = 100):
        """Initialize combined vision model.

        Args:
            num_classes: Number of troop/card types
        """
        super().__init__()
        self.troop_detector = ClashDetector(num_classes=num_classes)
        self.card_classifier = CardClassifier(num_card_types=num_classes)
        self.elixir_detector = ElixirDetector()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run all detection models.

        Args:
            x: Input screenshot (B, 3, H, W)

        Returns:
            Dictionary with keys:
            - troop_detections: Raw troop detection tensor
            - card_logits: Card classification logits
            - elixir: Elixir value predictions
        """
        return {
            "troop_detections": self.troop_detector(x),
            # Card and elixir require cropping first
            # "card_logits": self.card_classifier.classify_hand(x),
            # "elixir": self.elixir_detector.detect_from_screenshot(x),
        }

    def load_pretrained(self, path: str) -> None:
        """Load pretrained weights.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"])

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            "model_state_dict": self.state_dict(),
        }, path)
