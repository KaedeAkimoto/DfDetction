from random import random, randint
from pathlib import Path
import shutil
from tqdm import tqdm
from typing import Literal


# global var
source_pool: set[Path] = set()
train_pool  = []
test_pool   = []
val_pool    = []
left_pool   = []

# parameter
source  = Path(r"F:\UserData\CODE_SPACE\DefectDetection\augmented\img")
train   = Path(r"F:\UserData\CODE_SPACE\DefectDetection\data\test\images")
test    = Path(r"F:\UserData\CODE_SPACE\DefectDetection\data\train\images")
val     = Path(r"F:\UserData\CODE_SPACE\DefectDetection\data\val\images")
part    = (.8, .0, .2) #(train, test, val)
# train_pool | test_pool | val_pool | list() | [] 
left_give_to = train_pool 



def main():
    deamon()

    threshold = tuple(sum(part[:i+1]) for i in range(len(part)))

    it = iter(source_pool)
    for img in it:
        rd = random()
        if rd < threshold[0]:   train_pool.append(img)
        elif rd < threshold[1]: test_pool.append(img)
        elif rd < threshold[2]: val_pool.append(img)
        else: 
            left_pool.append(img)

    order: list[tuple[list, float, Path]] = sorted(
        [
            (train_pool, part[0], train), 
            (test_pool, part[1], test), 
            (val_pool, part[2], val)
        ], 
        key=lambda tri: len(tri[0]) - int(tri[1] * len(source_pool)), 
        reverse=True
    )
    
    for tri in order: sec_distribution(tri[0], tri[1])
    left_give_to.extend(it)

    print(
        "allocation: \n"
        f"train set: {len(train_pool)} imgs",
        f"test set: {len(test_pool)} imgs",
        f"valid set: {len(val_pool)} imgs",
        sep=', '
    )

    if not ask_continue("starting copy img to dataset"): print('bye.'); exit()

    for tri in order: cp(tri[0], tri[2])

    print("done.")


def deamon():
    if not all((
        source.exists(),
        train.exists(),
        test.exists(),
        val.exists()
    )): 
        print("dir not exists")
        exit()
    
    global source_pool
    source_pool = {*(
        img for img in source.iterdir() 
        if img.suffix.lower() in {
            '.jpeg', '.png', '.jpg', '.bmp', '.tiff'
    })}

    if len(source_pool) == 0:
        print("no img select")
        exit(0)

    ext = {img.suffix for img in source_pool}
    if len(ext) > 1:
        print(f"too many type: {ext}")
        exit(0)
    
    if sum(part) - 1 > 1e-8:
        print(f"cannot distribution in {part}")
        exit(0)

    print(
        "expect: \n"
        f"train set: {len(source_pool) * part[0]} imgs",
        f"test set: {len(source_pool) * part[1]} imgs",
        f"valid set: {len(source_pool) * part[2]} imgs",
        sep=', '
    )
    

def ask_continue(dec: str | None = None) -> bool:
    if dec: print(dec) 
    try: 
        while not (ans := input("continue with (y | yes) else quit: ")):
            print("invalid input. \n>>> ")

        return ans.lower() in {'y', 'yes'}
    
    except KeyboardInterrupt:
        return False
    


def sec_distribution(pool: list, part: float):
    dif = len(pool) - int(len(source_pool) * part)

    if dif > 0:
        left_pool.extend(pool.pop(randint(0, len(pool)-1)) for _ in range(dif))
    elif dif < 0:
        pool.extend(left_pool.pop(randint(0, len(left_pool)-1)) for _ in range(-dif))

def cp(pool: list[Path], dir: Path):
    invalid_file: set[tuple[Exception, Path]] = set()
    with tqdm(total=len(pool), desc=f"copy to {dir}") as pbar:
        for f in pool:
            try:
                dst_path = dir / f.name
                if dst_path.exists():   raise FileExistsError

                shutil.copy2(f, dst_path)
                pbar.set_postfix_str(f.name)
                pbar.update()
            
            except Exception as e:
                invalid_file.add((e, f))
                pbar.update()

    if invalid_file:
        print(*((idx, e.__repr__, f.name) for idx, (e, f) in enumerate(invalid_file)), sep='\n')
    


if __name__ == "__main__":
    main()

    

