    import os
    import torch
    from PIL import Image
    from torchvision import transforms
    import numpy as np
    from glob import glob
    from tqdm import tqdm
    import matplotlib.pyplot as plt


    class EyeImageProcessor:
        def __init__(self, img_size=256):
            """
            初始化图像处理器
            :param img_size: 统一调整的图像尺寸
            """
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            self.img_size = img_size

        def process_pair(self, left_path, right_path):
            """
            处理左右眼图像对为6通道张量
            :return: torch.Tensor [6, H, W]
            """
            left_img = Image.open(left_path).convert('RGB')
            right_img = Image.open(right_path).convert('RGB')

            left_tensor = self.transform(left_img)
            right_tensor = self.transform(right_img)

            return torch.cat([left_tensor, right_tensor], dim=0)

        def visualize_processing(self, left_path, right_path):
            """可视化处理前后的图像对比"""
            left_img = Image.open(left_path).convert('RGB')
            right_img = Image.open(right_path).convert('RGB')

            processed = self.process_pair(left_path, right_path)

            # 反标准化函数
            def denormalize(tensor):
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                return tensor * std + mean

            # 分离左右眼
            left_processed = denormalize(processed[:3]).permute(1, 2, 0).numpy()
            right_processed = denormalize(processed[3:]).permute(1, 2, 0).numpy()

            # 绘制对比图
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes[0, 0].imshow(left_img)
            axes[0, 0].set_title('Original Left')
            axes[0, 1].imshow(right_img)
            axes[0, 1].set_title('Original Right')
            axes[1, 0].imshow(np.clip(left_processed, 0, 1))
            axes[1, 0].set_title('Processed Left')
            axes[1, 1].imshow(np.clip(right_processed, 0, 1))
            axes[1, 1].set_title('Processed Right')

            for ax in axes.flat:
                ax.axis('off')
            plt.tight_layout()
            plt.show()

        def process_directory(self, left_dir, right_dir, save_dir=None):
            """
            批量处理目录中的图像对
            :return: 处理后的张量列表 [N, 6, H, W]
            """
            left_files = sorted(glob(os.path.join(left_dir, '*.jpg')))
            right_files = sorted(glob(os.path.join(right_dir, '*.jpg')))
            assert len(left_files) == len(right_files), "左右眼图像数量不匹配"

            processed = []
            # 添加进度条
            progress_bar = tqdm(zip(left_files, right_files), total=len(left_files), desc="Processing Images")

            for left_path, right_path in progress_bar:
                try:
                    tensor_6ch = self.process_pair(left_path, right_path)
                    processed.append(tensor_6ch)

                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        # 修改文件名：去除_left/_right后缀
                        base_name = os.path.basename(left_path).replace('_left', '')
                        torch.save(tensor_6ch, os.path.join(save_dir, f'processed_{base_name}.pt'))

                except Exception as e:
                    print(f"\nError processing {left_path}: {str(e)}")
                    continue

            return torch.stack(processed) if processed else None


    # 使用示例
    if __name__ == "__main__":
        # 初始化处理器
        processor = EyeImageProcessor()

        # 路径设置
        left_dir = "C:/Users/26448/Desktop/A07_Data/A07_Data/Training_Dataset/left"
        right_dir = "C:/Users/26448/Desktop/A07_Data/A07_Data/Training_Dataset/right"
        save_dir = "C:/Users/26448/Desktop/A07_Data/A07_Data/Training_Dataset/processed"

        # 单对图像处理和可视化测试
        test_left = os.path.join(left_dir, "0_left.jpg")
        test_right = os.path.join(right_dir, "0_right.jpg")

        if os.path.exists(test_left) and os.path.exists(test_right):
            print("\n正在执行单对图像处理测试...")
            processor.visualize_processing(test_left, test_right)

            # 处理并保存单对
            tensor = processor.process_pair(test_left, test_right)
            torch.save(tensor, "test_pair.pt")
            print(f"测试文件已保存为 test_pair.pt")

        # 批量处理
        print("\n开始批量处理目录...")
        batch_data = processor.process_directory(left_dir, right_dir, save_dir)

        if batch_data is not None:
            print(f"\n处理完成！共 {len(batch_data)} 个样本")
            print(f"最终数据形状: {batch_data.shape}")

            # 保存整个批处理结果
            torch.save(batch_data, os.path.join(save_dir, "full_dataset.pt"))
            print(f"完整数据集已保存为 full_dataset.pt")
        else:
            print("\n处理失败，请检查输入路径和文件格式")