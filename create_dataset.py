#!/usr/bin/env python3
"""
Build a balanced 10k-real / 10k-fake face-image folder with maximum
deepfake-tech variety.

Usage:
    python build_face_dataset.py --out_dir ./faces20k --seed 42
"""

import argparse, subprocess, zipfile, random, shutil, os, pathlib, sys
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

# ----------------------------------------------------------------------
# 1. Edit here to add / remove sources
# ----------------------------------------------------------------------
DATASETS = [
    {
        "name": "140k",
        "slug": "xhlulu/140k-real-and-fake-faces",
        "subdirs": {"real": "real", "fake": "fake"},
        "fake_label": "stylegan2"
    },
    {
        "name": "deepfake_real",
        "slug": "manjilkarki/deepfake-and-real-images",
        "subdirs": {"real": "real", "fake": "fake"},
        "fake_label": "pggan_stylegan_mix"
    },
    {
        "name": "dfdc_f150",
        "slug": "sciarrilli/dfdc-f150",
        "subdirs": {"real": "real", "fake": "fake"},
        "fake_label": "dfdc_swaps"
    },
    {
        "name": "faceforensics_imgs",
        "slug": "greatgamedota/faceforensics",
        "subdirs": {"real": "real", "fake": "fake"},
        "fake_label": "ffpp_swaps"
    },
]

TARGET_PER_CLASS = 10_000
# ----------------------------------------------------------------------

def kaggle_download(slug: str, dest: pathlib.Path) -> pathlib.Path:
    """Download <slug> to dest/. Returns path of the zip."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / f"{slug.split('/')[-1]}.zip"
    if zip_path.exists():
        return zip_path
    print(f"Downloading {slug} …")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--quiet"],
        check=True,
    )
    return zip_path

def extract(zip_path: pathlib.Path, dest: pathlib.Path) -> pathlib.Path:
    """Unzip if needed. Returns extraction dir."""
    extract_dir = dest / zip_path.stem
    if extract_dir.exists():
        return extract_dir
    print(f"Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(path=extract_dir)
    return extract_dir

def glob_images(root: pathlib.Path, pattern: str):
    return list(root.glob(pattern)) + list(root.glob(pattern.replace("jpg", "png")))

def main(out_dir: pathlib.Path, seed: int):
    random.seed(seed)
    temp_root = out_dir / "_raw"
    real_pool, fake_pool = [], []
    fake_source_tag = {}  # path -> dataset tag

    # ------------------------------------------------------------------
    # 2. Pull sources
    # ------------------------------------------------------------------
    for ds in DATASETS:
        zip_path = kaggle_download(ds["slug"], temp_root)
        extract_dir = extract(zip_path, temp_root)
        real_dir = extract_dir / ds["subdirs"]["real"]
        fake_dir = extract_dir / ds["subdirs"]["fake"]
        real_pool += glob_images(real_dir, "**/*.jpg")
        fakes = glob_images(fake_dir, "**/*.jpg")
        fake_pool += fakes
        for fp in fakes:
            fake_source_tag[str(fp)] = ds["fake_label"]

    # sanity check
    if len(real_pool) < TARGET_PER_CLASS or len(fake_pool) < TARGET_PER_CLASS:
        print("Not enough images – add another dataset.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Sample
    # ------------------------------------------------------------------
    random.shuffle(real_pool)
    random.shuffle(fake_pool)

    # try to spread fake quota equally over sources
    per_source_quota = TARGET_PER_CLASS // len(DATASETS)
    selected_fake = []
    taken = defaultdict(int)
    for fp in fake_pool:
        tag = fake_source_tag[str(fp)]
        if taken[tag] < per_source_quota:
            selected_fake.append(fp)
            taken[tag] += 1
        if len(selected_fake) == TARGET_PER_CLASS:
            break
    # top-up if we’re short (some sets too small)
    if len(selected_fake) < TARGET_PER_CLASS:
        needed = TARGET_PER_CLASS - len(selected_fake)
        selected_fake += fake_pool[len(selected_fake) : len(selected_fake) + needed]

    selected_real = real_pool[:TARGET_PER_CLASS]

    # ------------------------------------------------------------------
    # 4. Copy to final tree + manifest
    # ------------------------------------------------------------------
    for cls in ("real", "fake"):
        (out_dir / cls).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    def copy_files(file_list, cls):
        for src in tqdm(file_list, desc=f"Copying {cls}"):
            dst = out_dir / cls / src.name
            shutil.copy(src, dst)
            manifest_rows.append(
                {"filename": dst.name, "label": cls,
                 "source": fake_source_tag.get(str(src), "n/a")}
            )

    copy_files(selected_real, "real")
    copy_files(selected_fake, "fake")

    pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)
    print("Done →", out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="faces20k", type=pathlib.Path,
                   help="destination folder")
    p.add_argument("--seed", default=42, type=int)
    args = p.parse_args()
    main(args.out_dir, args.seed)
