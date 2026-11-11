# Your model classes (A3, A4, A5) and helpers that later notebooks need (e.g. A2 feature functions), and one
# preprocessing function per model (normalization constants included), so P1_V, P1_B and P1_665 prepare inputs
# exactly as P1_A did. Save the training statistics they need (e.g. channel means and stds) to CKPT_DIR.
# This cell writes the file; the next cell imports it. Re-run both after every change.
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import cv2
import timm


#A1
class A1Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3 * 32 * 32, 19)

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x.flatten(1))


def resize_a1(images):
    return np.stack([cv2.resize(i, (32, 32), interpolation=cv2.INTER_AREA) for i in images])


def preprocess_a1(images, mean, std):
    images = np.asarray(images)
    if images.shape[1:3] != (32, 32):
        images = resize_a1(images)

    x = images.astype(np.float32)
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 1, 3)
    x = (x - mean) / std

    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


#A2
class A2Linear(nn.Module):
  def __init__(self, feature_dim):
      super().__init__()
      self.linear = nn.Linear(feature_dim, 19)

      nn.init.zeros_(self.linear.weight)
      nn.init.zeros_(self.linear.bias)

  def forward(self, x):
      return self.linear(x)


def extract_a2_features(images):
    images = np.asarray(images)

    hog = cv2.HOGDescriptor(
        (64, 64), (16, 16), (8, 8), (8, 8), 9
    )

    kernels = []

    for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
        for wavelength, size in ((4.0, 13), (8.0, 27)):
            kernel = cv2.getGaborKernel(
                ksize=(size, size),
                sigma=0.56 * wavelength,
                theta=theta,
                lambd=wavelength,
                gamma=0.5,
                psi=0.0,
                ktype=cv2.CV_32F,
            )

            kernel = kernel - kernel.mean()
            kernels.append(kernel)

    hsv_rows = []
    hog_rows = []
    gabor_rows = []

    for im in images:
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        histograms = []

        for ch, upper in ((0, 180), (1, 256), (2, 256)):
            hist = cv2.calcHist(
                [hsv], [ch], None, [16], [0, upper]
            ).ravel()

            histograms.append(hist / hist.sum())

        hsv_rows.append(np.concatenate(histograms))
        hog_rows.append(hog.compute(gray).ravel())

        gray_float = gray.astype(np.float32) / 255.0
        gabor_features = []

        for kernel in kernels:
            response = cv2.filter2D(
                gray_float,
                cv2.CV_32F,
                kernel,
                borderType=cv2.BORDER_REFLECT_101,
            )

            absolute_response = np.abs(response)

            gabor_features.extend([
                absolute_response.mean(),
                absolute_response.var(),
            ])

        gabor_rows.append(gabor_features)

    return {
        "hsv": np.asarray(hsv_rows, dtype=np.float32),
        "hog": np.asarray(hog_rows, dtype=np.float32),
        "gabor": np.asarray(gabor_rows, dtype=np.float32),
    }


def standardize_a2(features, stats, feature_set):
    names = (
        ("hsv", "hog", "gabor")
        if feature_set == "concat"
        else (feature_set,)
    )

    blocks = [
        (features[name] - stats[f"{name}_mean"])
        / stats[f"{name}_std"]
        for name in names
    ]

    x = np.concatenate(blocks, axis=1).astype(np.float32)
    return torch.from_numpy(x)


def preprocess_a2(images, stats, feature_set):
    features = extract_a2_features(images)
    return standardize_a2(features, stats, feature_set)


# A3
class A3MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3 * 32 * 32, 256)
        self.relu = nn.ReLU()
        self.output = nn.Linear(256, 19)

    def forward(self, x):
        x = x.flatten(1)
        x = self.hidden(x)
        x = self.relu(x)
        return self.output(x)


def resize_a3(images):
    return resize_a1(images)


def preprocess_a3(images, mean, std):
    return preprocess_a1(images, mean, std)



# A4
class A4BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class A4ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem_conv = nn.Conv2d(
            3, 32, kernel_size=7, stride=1, padding=3, bias=False,
        )
        self.stem_bn = nn.BatchNorm2d(32)

        self.stage1 = nn.Sequential(
            A4BasicBlock(32, 32),
            A4BasicBlock(32, 32),
        )
        self.stage2 = nn.Sequential(
            A4BasicBlock(32, 64, stride=2),
            A4BasicBlock(64, 64),
        )
        self.stage3 = nn.Sequential(
            A4BasicBlock(64, 128, stride=2),
            A4BasicBlock(128, 128),
        )
        self.fc = nn.Linear(128, 19)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.mean(dim=(2, 3))
        return self.fc(x)


def preprocess_a4(images, mean, std):
    x = np.asarray(images, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 1, 3)
    x = (x - mean) / std
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


# A5
class A5ViT(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            3, 128, kernel_size=8, stride=8
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 128))
        self.pos_embed = nn.Parameter(torch.zeros(1, 65, 128))

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=512,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(4)
        ])

        self.norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, 19)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        cls = self.norm(x[:, 0])
        return self.head(cls)


def preprocess_a5(images, mean, std):
    return preprocess_a4(images, mean, std)


#A6
def A6ViT(pretrained=True):
    model = timm.create_model(
        "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        pretrained=pretrained,
    )

    torch.manual_seed(1)
    model.head = nn.Linear(192, 19)
    return model


def preprocess_a6(images, mean, std, device):
    x = torch.as_tensor(images, device=device)
    x = x.permute(0, 3, 1, 2).to(torch.float32).div_(255.0)

    if x.shape[-2:] != (224, 224):
        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

    mean = torch.as_tensor(mean, dtype=x.dtype, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(std, dtype=x.dtype, device=device).view(1, 3, 1, 1)
    return (x - mean) / std
