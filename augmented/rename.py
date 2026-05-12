from pathlib import Path


img_path = Path(r"F:\UserData\CODE_SPACE\DefectDetection\augmented\img")

def main():
    rollback: list[tuple[Path, str]] = []
    stoped: tuple[int, Path] = (-1, Path())
    cnt = 0
    try:
        for idx, img_f in enumerate(img_path.iterdir()):
            stoped = (idx, img_f)
            rollback.append((img_f, img_f.name))
            new_name = rf"{img_f.parent}\aug_{idx}" + img_f.suffix
            print(rf"rename {img_f.parent}\{img_f.name} -> {new_name}")
            img_f.rename(new_name)

    except:
        print(f"stoped at file: {stoped}")
        
        for img, name in rollback:
            try:
                img.rename(name)
            except:
                cnt += 1
                print(f"{cnt} rollback but fail: {img}")
    
    finally:
        print(f"done")
            


if __name__ == "__main__":
    main()