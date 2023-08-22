import sys
sys.path.append("..")

import torch
import hydra
from omegaconf import DictConfig
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from lib.data.leafs import LeafDiseaseDataset
# import segmentation_models_pytorch as smp


def fetch_image(image_path, reduction_rate=32, device="cuda"):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape
    h_div8 = h // reduction_rate * reduction_rate
    w_div8 = w // reduction_rate * reduction_rate

    img = cv2.resize(img, (w_div8, h_div8))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    return img_tensor, (w, h)


def img_from_pred(pred):
    pred = pred.squeeze(0).permute(1, 2, 0)
    pred = pred.cpu().numpy()[..., 1]
    pred = (pred * 255).astype("uint8")
    return pred


# TODO: 4. Use predict() instead of forward()


@hydra.main(version_base=None, config_path="configs", config_name="eval_base")
def main(cfg: DictConfig) -> None:
    model = hydra.utils.instantiate(cfg.model)
    model.eval()

    dataset = LeafDiseaseDataset(root="../notebooks/data/leaf_disease", mode="val")

    infer_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    reduction_rate = model.model.encoder.feature_info.reduction()[-1]

    sample = dataset[0]
    img = sample["image"]
    mask = sample["mask"]

    h, w, _ = img.shape
    h_div8 = h // reduction_rate * reduction_rate
    w_div8 = w // reduction_rate * reduction_rate
    img = cv2.resize(img, (w_div8, h_div8))
    mask = cv2.resize(mask, (w_div8, h_div8), interpolation=cv2.INTER_NEAREST)
    transformed = infer_transform(image=img, mask=mask)
    img = transformed["image"].to("cuda")
    mask = transformed["mask"].to("cuda")

    with torch.no_grad():
        logits_mask = model(img.unsqueeze(0))["out"]
        prob_mask = logits_mask.sigmoid()  # convert mask values to probabilities
        pred_mask = (prob_mask > 0.5).float()  # apply thresholding

        tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(pred_mask, dim=1).squeeze().long(),
                                               mask.long(),
                                               mode="binary")

        iou = smp.metrics.iou_score(tp, fp, fn, tn)
        print(f"IOU: {iou}")


    mask = img_from_pred(pred_mask)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("mask.png", mask)


if __name__ == "__main__":
    main()

