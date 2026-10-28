"""Person masks: load one you made yourself, or (optionally) detect people automatically."""
import numpy as np
import cv2


def load_mask(path, width, height):
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        # Use alpha if present (a cut-out PNG), otherwise luminance.
        m = m[..., 3] if m.shape[2] == 4 else cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
    return m > 127


def load_photo(path, width, height):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def auto_person_mask(photo_rgb, score=0.6):
    """Union of 'person' instances from torchvision's Mask R-CNN (COCO).

    Optional: needs torchvision, and downloads its pretrained weights on first use.
    """
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).eval()
    x = torch.from_numpy(photo_rgb).permute(2, 0, 1).float()
    with torch.no_grad():
        out = model([x])[0]
    keep = (out["labels"] == 1) & (out["scores"] >= score)
    if keep.sum() == 0:
        return np.zeros(photo_rgb.shape[:2], bool)
    return (out["masks"][keep, 0] > 0.5).any(0).numpy()
