'''
数据增强脚本
 * 随机灰度变化
 * 随机裁剪
 * 随机仿射变换
 * 随机色调变化
 * 多图像拼接
'''


import torch
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from typing import List, Tuple


class AdvancedDataAugmentor:
    def __init__(self, 
                 target_count=1000,  # 目标增强后的图片总数
                 output_size=(224, 224),  # 输出图片尺寸
                 seed=42):
        """
        初始化数据增强器
        
        Args:
            target_count: 目标图片数量
            output_size: 输出图片尺寸 (height, width)
            seed: 随机种子
        """
        self.target_count = target_count
        self.output_size = output_size
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def load_images(self, input_dir: str) -> List[Image.Image]:
        """加载所有图片"""
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            image_paths.extend(Path(input_dir).glob(ext))
            image_paths.extend(Path(input_dir).glob(ext.upper()))
        
        print(f"找到 {len(image_paths)} 张图片")
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"无法加载图片 {img_path}: {e}")
        
        return images
    
    def get_random_augmentation_pipeline(self) -> transforms.Compose:
        """获取随机的增强流水线"""
        # 随机选择3-5个增强操作
        all_transforms = []
        
        # 1. 随机仿射变换
        if random.random() < 0.8:  # 80%的概率应用
            angle = random.uniform(-30, 30)  # 旋转角度
            translate = (random.uniform(0, 0.2), random.uniform(0, 0.2))  # 平移
            scale = random.uniform(0.8, 1.2)  # 缩放
            shear = random.uniform(0, 20)  # 剪切
            
            all_transforms.append(
                transforms.RandomAffine(
                    degrees=abs(angle),
                    translate=translate,
                    scale=(scale, scale),
                    shear=shear,
                )
            )
        
        # 2. 随机裁剪
        if random.random() < 0.7:  # 70%的概率应用
            all_transforms.extend([
                transforms.RandomResizedCrop(
                    self.output_size,
                    scale=(0.6, 1.0),
                    ratio=(0.8, 1.2)
                )
                # transforms.ToTensor(),
                # transforms.RandomErasing(
                #     p=1.0,  # 总是执行
                #     scale=(0.2, 0.4),  # 遮挡面积比例范围
                #     ratio=(0.3, 3.3),  # 长宽比范围
                #     value=0,  # 填充黑色（像素值0）
                #     inplace=False
                # ),
                # transforms.ToPILImage()
            ])
        
        # 3. 随机水平/垂直翻转
        if random.random() < 0.5:  # 50%的概率水平翻转
            all_transforms.append(transforms.RandomHorizontalFlip(p=1.0))
        
        if random.random() < 0.3:  # 30%的概率垂直翻转
            all_transforms.append(transforms.RandomVerticalFlip(p=1.0))
        
        # 4. 颜色变换
        if random.random() < 0.7:  # 70%的概率应用
            all_transforms.append(
                transforms.ColorJitter(
                    brightness=random.uniform(0.5, 1.5),  # 亮度
                    contrast=random.uniform(0.5, 1.5),    # 对比度
                    saturation=random.uniform(0.5, 1.5),  # 饱和度
                    hue=random.uniform(0, 0.2)         # 色调
                )
            )
        
        # 5. 模糊
        if random.random() < 0.4:  # 40%的概率应用
            kernel_size = random.choice([3, 5, 7])
            all_transforms.append(
                transforms.GaussianBlur(
                    kernel_size=kernel_size,
                    sigma=random.uniform(0.1, 2.0)
                )
            )
        
        # 6. 随机拼接（将两张图片拼接成一张）
        if random.random() < 0.3 and len(all_transforms) > 0:  # 30%的概率
            # 这里我们会在apply_augmentations中单独处理
            pass
        else:
            # 如果没有拼接操作，则调整到固定大小
            all_transforms.append(transforms.Resize(self.output_size))
        
        return transforms.Compose(all_transforms)
    
    def create_mosaic(self, image1: Image.Image, image2: Image.Image) -> Image.Image:
        """创建马赛克拼接图片"""
        w, h = self.output_size
        
        # 创建新图片
        mosaic = Image.new('RGB', (w, h))
        
        # 随机选择拼接模式
        mode = random.choice(['horizontal', 'vertical', 'four_grid'])
        
        if mode == 'horizontal':
            # 水平拼接
            left_part = image1.resize((w//2, h))
            right_part = image2.resize((w//2, h))
            mosaic.paste(left_part, (0, 0))
            mosaic.paste(right_part, (w//2, 0))
            
        elif mode == 'vertical':
            # 垂直拼接
            top_part = image1.resize((w, h//2))
            bottom_part = image2.resize((w, h//2))
            mosaic.paste(top_part, (0, 0))
            mosaic.paste(bottom_part, (0, h//2))
            
        else:  # four_grid
            # 四宫格拼接
            quarter_w, quarter_h = w//2, h//2
            for i in range(2):
                for j in range(2):
                    if random.random() < 0.5:
                        part = image1.resize((quarter_w, quarter_h))
                    else:
                        part = image2.resize((quarter_w, quarter_h))
                    mosaic.paste(part, (i*quarter_w, j*quarter_h))
        
        return mosaic
    
    def apply_random_cutout(self, image: Image.Image) -> Image.Image:
        """应用随机剪切（cutout）"""
        img_array = np.array(image)
        h, w, _ = img_array.shape
        
        # 随机生成剪切区域
        cutout_h = random.randint(h//8, h//4)
        cutout_w = random.randint(w//8, w//4)
        y = random.randint(0, h - cutout_h)
        x = random.randint(0, w - cutout_w)
        
        # 用随机颜色填充剪切区域
        fill_color = (0, 0, 0)
        img_array[y:y+cutout_h, x:x+cutout_w, :] = fill_color
        
        return Image.fromarray(img_array)
    
    def apply_augmentations(self, image: Image.Image, 
                           reference_images: List[Image.Image]) -> Image.Image:
        """应用数据增强"""
        # 随机选择是否创建拼接图片
        if random.random() < 0.3 and len(reference_images) > 1:
            # 从参考图片中随机选择一张（不包括当前图片本身）
            other_images = [img for img in reference_images if img is not image]
            if other_images:
                other_image = random.choice(other_images)
                # 创建拼接图片
                image = self.create_mosaic(
                    image.resize(self.output_size),
                    other_image.resize(self.output_size)
                )
        
        # 获取增强流水线
        pipeline = self.get_random_augmentation_pipeline()
        
        # 应用增强
        augmented = pipeline(image)
        
        # 应用随机剪切（在增强之后）
        if random.random() < 0.3:
            augmented = self.apply_random_cutout(augmented)
            # 重新调整大小
            augmented = F.resize(augmented, self.output_size)
        
        return augmented
    
    def augment_dataset(self, input_dir: str, output_dir: str):
        """增强整个数据集"""
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 加载原始图片
        original_images = self.load_images(input_dir)
        if not original_images:
            print("没有找到图片！")
            return
        
        print(f"原始图片数量: {len(original_images)}")
        print(f"目标增强数量: {self.target_count}")
        
        # 计算每张原始图片需要生成的增强图片数量
        images_per_original = max(1, self.target_count // len(original_images))
        
        # 进行数据增强
        augmented_count = 0
        
        for i, img in enumerate(original_images):
            print(f"处理第 {i+1}/{len(original_images)} 张原始图片...")
            
            # 首先保存原始图片
            original_path = os.path.join(output_dir, f"original_{i:04d}.jpg")
            img.save(original_path)
            
            # 为每张原始图片生成多个增强版本
            for j in range(images_per_original):
                if augmented_count >= self.target_count:
                    break
                    
                # 应用增强
                augmented_img = self.apply_augmentations(img.copy(), original_images)
                
                # 保存增强后的图片
                save_path = os.path.join(output_dir, f"aug_{i:04d}_{j:04d}.jpg")
                augmented_img.save(save_path)
                augmented_count += 1
                
                if augmented_count % 100 == 0:
                    print(f"已生成 {augmented_count}/{self.target_count} 张增强图片")
        
        print(f"\n数据增强完成！")
        print(f"生成图片总数: {augmented_count}")
        print(f"保存路径: {output_dir}")
        
        return augmented_count
    
    def visualize_augmentations(self, image_path: str, num_examples=9):
        """可视化数据增强效果"""
        img = Image.open(image_path).convert('RGB')
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()
        
        # 显示原始图片
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # 创建参考图片列表（这里用同一张图片模拟）
        reference_images = [img] * 5
        
        # 显示8个增强后的例子
        for i in range(1, 9):
            augmented_img = self.apply_augmentations(img.copy(), reference_images)
            axes[i].imshow(augmented_img)
            axes[i].set_title(f"Augmented #{i}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


def deamon(input_, output_):
    '''确认用户参数是否合理'''
    inp, outp = Path(input_), Path(output_)
    if not inp.exists():
        print("input dir not exists")
        exit()
    if not outp.exists():
        print("outp dir not exists")
        exit()
    input_files = {*(it for it in inp.iterdir() if it.name.lower() != 'classes.txt')}
    cnt = len(input_files)
    if cnt == 0:
        print("no file select")
        exit()
    cnt = len(set(outp.iterdir()))
    if cnt != 0:
        print("output not empty")
        exit()
    
    ext = {ex.suffix for ex in input_files}
    if len(ext) > 1:
        print(f"too many types: {ext}")
        exit()
    
    print(*input_files, sep='\n')
    print(f"total {len(input_files)}")
    try:
        while not (flag := input("continue with (yes | y) else quit \n>>> ")):
            print("invalid input.")
    
    except KeyboardInterrupt:
        print("bye.")
        exit()

    if flag.lower() not in {'y', 'yes'}:
        print("bye.")
        exit()

    print("start!")


def main():
    # 配置参数
    INPUT_DIR = r"F:\UserData\CODE_SPACE\DefectDetection\DD_ori_data\original"  # 原始图片目录
    OUTPUT_DIR = r"F:\UserData\CODE_SPACE\DefectDetection\DD_ori_data\test"  # 增强后图片保存目录
    TARGET_COUNT = 1000  # 目标生成图片数量

    deamon(INPUT_DIR, OUTPUT_DIR)

    # 创建增强器
    augmentor = AdvancedDataAugmentor(
        target_count=TARGET_COUNT,
        output_size=(640, 640),  # 可以根据你的模型调整大小
        seed=42
    )
    
    # 执行数据增强
    augmentor.augment_dataset(INPUT_DIR, OUTPUT_DIR)
    
    # 可视化增强效果（可选）
    # 确保原始图片目录中有至少一张图片
    image_files = list(Path(INPUT_DIR).glob("*.jpg")) + \
                  list(Path(INPUT_DIR).glob("*.png"))
    
    if image_files:
        augmentor.visualize_augmentations(str(image_files[0]))


if __name__ == "__main__":
    main()