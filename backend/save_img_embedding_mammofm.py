"""
MammoFM copy of LLaVA save_img_embedding.py with a fix for CPU / no-CUDA visibility.

Upstream always used torch.load(..., map_location="cuda") while main() could pick device
"cpu" when torch.cuda.is_available() is False, causing:
RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llava.model.multimodal_encoder.efficientnet import EfficientNet


def seed_all(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


class MammoVisionTower(torch.nn.Module):
    def __init__(self, args, map_location=None):
        super().__init__()
        if map_location is None:
            map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(map_location, str):
            map_location = torch.device(map_location)
        ckpt = torch.load(args.mammo_clip_chkpt, map_location=map_location)
        print("Image encoder configuration:\n", ckpt["config"]["model"]["image_encoder"])
        self.config = ckpt["config"]["model"]["image_encoder"]

        self.image_encoder = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)
        self.image_encoder.out_dim = 2048

        image_encoder_weights = {}
        for k in ckpt["model"].keys():
            if k.startswith("image_encoder."):
                new_key = ".".join(k.split(".")[1:])
                image_encoder_weights[new_key] = ckpt["model"][k]
        _ = self.image_encoder.load_state_dict(image_encoder_weights, strict=True)

        self.image_encoder_type = self.config["model_type"]
        self.arch = args.arch.lower()

        print("Freezing image encoder parameters.")
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, images):
        images = images.permute(0, 3, 1, 2)
        input_dict = {"image": images, "breast_clip_train_mode": True}
        image_features, raw_features = self.image_encoder.to(images.device)(input_dict)
        return raw_features


class MammoDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.df = pd.read_csv(self.args.data_csv)
        self.transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        dataset = data["dataset"]
        image_path = data["file_path"]
        if dataset == "BU":
            exam_id = Path(image_path).parent.name
            image_name = Path(image_path).name
            if "controls" in image_path:
                img_path = Path(self.args.bu_path) / "controls" / "test_images_png" / exam_id / image_name
            elif "cases" in image_path:
                img_path = Path(self.args.bu_path) / "cases" / "test_images_png" / exam_id / image_name
        elif dataset == "UPMC":
            img_path = str(image_path).replace(
                "/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset/External/UPMC/DICOM/images_png_CC_MLO",
                self.args.upmc_path,
            )
            img_path = Path(img_path)

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = np.array(img)
            augmented = self.transform(image=img)
            img = augmented["image"]

            img = img.astype("float32")
            img -= img.min()
            img /= img.max()
            img = torch.tensor((img - self.args.mean) / self.args.std, dtype=torch.float32)
        else:
            img = np.array(img)
            img = img.astype("float32")
            img -= img.min()
            img /= img.max()
            img = torch.tensor((img - self.args.mean) / self.args.std, dtype=torch.float32)

        img_path = str(img_path)
        return {
            "img_path": img_path,
            "img": img,
        }


def do_experiments(args, device):
    map_location = torch.device(device)
    model = MammoVisionTower(args, map_location=map_location).to(device)
    model.eval()
    dataset = MammoDataset(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting embeddings"):
        img = batch["img"].to(device)
        img_path = batch["img_path"][0]
        img_path = Path(img_path)
        raw_features = model(img)
        embed_dir = img_path.parent
        embed_filename = img_path.stem
        save_file = embed_dir / f"{embed_filename}.pt"
        torch.save(raw_features.cpu(), save_file)


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed configuration file")
    parser.add_argument(
        "--mammo-clip-chkpt",
        metavar="DIR",
        default="/work/nvme/beaq/shg121/LLaVa-Breast/src/Mammo-clip-chk_pt/model-best.tar",
        help="path to checkpoints",
    )
    parser.add_argument(
        "--data-csv",
        default="/work/nvme/beaq/shg121/LLaVa-Breast/dataset_json/train_LLaVA_Breast_stage1_single_image.csv",
        type=str,
        help="Path to data file",
    )
    parser.add_argument(
        "--bu_path",
        default="/work/nvme/beaq/shared/Data/RSNA_Breast_Imaging/Dataset/External/BU_Mammo/mammoclip",
        type=str,
        help="Path to image file",
    )
    parser.add_argument(
        "--upmc_path",
        default="/work/nvme/beaq/shared/Data/RSNA_Breast_Imaging/Dataset/External/UPMC/DICOM/images_png_CC_MLO",
        type=str,
        help="Path to image file",
    )
    parser.add_argument(
        "--csv-file",
        default="RSNA_Cancer_Detection/train_folds_w_concepts.csv",
        type=str,
        help="data csv file",
    )
    parser.add_argument("--arch", default="tf_efficientnet_b5_ns", type=str)
    parser.add_argument("--epochs-warmup", default=0, type=float)
    parser.add_argument("--num_cycles", default=0.5, type=float)
    parser.add_argument("--alpha", default=10, type=float)
    parser.add_argument("--sigma", default=15, type=float)
    parser.add_argument("--p", default=1.0, type=float)
    parser.add_argument("--mean", default=0.3089279, type=float)
    parser.add_argument("--std", default=0.25053555408335154, type=float)
    parser.add_argument("--focal-alpha", default=0.6, type=float)
    parser.add_argument("--focal-gamma", default=2.0, type=float)
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--n_folds", default=4, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--epochs", default=7, type=int)
    parser.add_argument("--lr", default=5.0e-5, type=float)
    parser.add_argument("--img-size", nargs="+", default=[1520, 912])
    parser.add_argument("--resize", default=512, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--apex", default="y", type=str)
    parser.add_argument("--print-freq", default=100, type=int)
    parser.add_argument("--log-freq", default=100, type=int)
    parser.add_argument("--running-interactive", default="n", type=str)
    parser.add_argument("--inference-mode", default="n", type=str)
    parser.add_argument("--model-type", default="Classifier", type=str)
    parser.add_argument("--select-cancer", default="y", type=str)

    return parser.parse_args()


def main(args):
    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    do_experiments(args, device)


if __name__ == "__main__":
    args = config()
    main(args)
