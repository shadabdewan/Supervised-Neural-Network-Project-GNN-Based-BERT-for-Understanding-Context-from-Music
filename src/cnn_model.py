import torch
import torch.nn as nn


class MelSpecCNN(nn.Module):

  def __init__(self, output_dim: int = 8, dropout: float = 0.3):
    """Convolutional Neural Network baseline for Log-Mel Spectrogram genre classification.

    Input shape expected: (batch_size, 1, 128, 1300)
    """
    super(MelSpecCNN, self).__init__()

    # 4 Conv blocks with channel progression: 1 -> 16 -> 32 -> 64 -> 128
    self.conv1 = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )

    self.conv3 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )

    self.conv4 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )

    # Adaptive Pooling to fix spatial output dimensions regardless of minor input shifts
    self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

    # Dense classification head
    # Flattened feature dimension: 128 channels * 4 * 4 spatial shape = 2048
    self.classifier = nn.Sequential(
        nn.Linear(128 * 4 * 4, 128),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(128, output_dim),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass returning raw logits (batch_size, 8)."""
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)

    x = self.adaptive_pool(x)
    x = torch.flatten(x, start_dim=1)

    logits = self.classifier(x)
    return logits