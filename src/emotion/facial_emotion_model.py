from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from PIL import Image


class FacialEmotionCNN(nn.Module):
    """Compact seven-class CNN baseline for Chuwi facial expression estimation."""

    CLASSES = (
        "ANGER",
        "DISGUST",
        "FEAR",
        "HAPPY",
        "CALM",
        "SAD",
        "SURPRISE",
    )

    def __init__(self, num_classes=None):
        super().__init__()
        num_classes = num_classes or len(self.CLASSES)
        if num_classes != len(self.CLASSES):
            raise ValueError(
                f"Chuwi facial model expects {len(self.CLASSES)} classes, got {num_classes}"
            )
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class FacialEmotionModel:
    """Inference wrapper for Chuwi's seven canonical facial emotion states."""

    def __init__(self, checkpoint=None, device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = FacialEmotionCNN().to(self.device)

        if checkpoint:
            checkpoint_path = Path(checkpoint)
            state = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self.model.load_state_dict(state)

        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

    @torch.inference_mode()
    def predict(self, image):
        """Return canonical emotion, confidence and class probabilities."""
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        index = int(torch.argmax(probabilities))

        scores = {
            label: float(probabilities[i])
            for i, label in enumerate(FacialEmotionCNN.CLASSES)
        }
        return {
            "emotion": FacialEmotionCNN.CLASSES[index],
            "confidence": float(probabilities[index]),
            "scores": scores,
            "source": "face",
        }
