from pathlib import Path
import shutil

img_dir = Path(r"F:\UserData\CODE_SPACE\DefectDetection\data\train\images")
lab_dir = Path(r"F:\UserData\CODE_SPACE\DefectDetection\data\train\labels")
pool_dir = Path(r"F:\UserData\CODE_SPACE\DefectDetection\data\train\pool")
pool_dir.mkdir(exist_ok=True)
pool = {ex.stem for ex in img_dir.iterdir()} - {ex.stem for ex in lab_dir.iterdir()}
ex_pool = {ex.stem for ex in lab_dir.iterdir()} - {ex.stem for ex in img_dir.iterdir()}
print(pool)
print(ex_pool)

# for img_stem in pool:
#     name = f"{img_stem}.jpg"
#     shutil.copy2(f"{img_dir}/{name}", f"{pool_dir}/{name}")