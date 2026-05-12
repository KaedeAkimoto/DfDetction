from pathlib import Path
# import shutil

img_dir = Path(r"F:\UserData\CODE_SPACE\DefectDetection\data\val\images")
lab_dir = Path(r"F:\UserData\CODE_SPACE\DefectDetection\data\val\labels")
pool_dir = Path(r"F:\UserData\CODE_SPACE\DefectDetection\data\val\pool")
pool_dir.mkdir(exist_ok=True)
pool = {ex.stem for ex in img_dir.iterdir()} - {ex.stem for ex in lab_dir.iterdir()}
ex_pool = {ex.stem for ex in lab_dir.iterdir()} - {ex.stem for ex in img_dir.iterdir()}
print(pool)
print(ex_pool)

for name in pool:
    lab = lab_dir / Path(f"{name}.txt")
    with lab.open('w'):
        ...
    print(f"create {lab.name}: {lab.exists()}")

# for img_stem in pool:
#     name = f"{img_stem}.jpg"
#     shutil.copy2(f"{img_dir}/{name}", f"{pool_dir}/{name}")