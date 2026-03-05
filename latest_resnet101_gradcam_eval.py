import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet101
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def main():
    # 1. 交互式获取测试图片名称
    image_name = input("Please enter the name of the image to test (e.g., sample1.jpg): ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 实例化 ResNet-101 模型并修改最后一层以匹配训练架构
    model = resnet101()
    num_ftrs = model.fc.in_features
    
    # 必须严格复刻 latest_train_resnet101_model.py 中定义的 fc 结构
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2)
    )
    model = model.to(device)
    
    checkpoint_path = 'resnet101_checkpoint/best_resnet101_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"Weight file not found: {checkpoint_path}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Successfully loaded model weights! Highest validation accuracy: {checkpoint['accuracy']:.4f}")

    # 3. 指定 ResNet-101 中用于生成 Grad-CAM 的目标层
    target_layers = [model.layer4[-1]]

    # 4. 自动拼接输入目录路径
    input_dir = 'Test_Grad-CAM_Images'
    img_path = os.path.join(input_dir, image_name) 
    
    if not os.path.exists(img_path):
        print(f"Error: Image '{image_name}' not found in the '{input_dir}' directory. Please check the spelling.")
        return

    # 5. 图像预处理 (必须与验证集完全一致)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    rgb_img = np.array(Image.open(img_path).convert('RGB'))
    rgb_img_float = np.float32(rgb_img) / 255
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    input_tensor = transform(Image.fromarray(rgb_img)).unsqueeze(0).to(device)

    # 6. 构建 Grad-CAM 对象并生成热力图
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # 指定我们感兴趣的类别：假设 0 是 'crocodile'
    targets = [ClassifierOutputTarget(0)] 
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # 7. 将热力图叠加到原始图像上
    resized_img = cv2.resize(rgb_img_float, (224, 224))
    visualization = show_cam_on_image(resized_img, grayscale_cam, use_rgb=True)

    # 8. 设置输出目录并保存结果
    output_dir = 'Test_Grad-CAM_Results'
    os.makedirs(output_dir, exist_ok=True) 
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Cropped Image")
    plt.imshow(resized_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("ResNet-101 Grad-CAM Heatmap")
    plt.imshow(visualization)
    plt.axis('off')
    
    plt.tight_layout()
    
    # 保存结果时使用 resnet101 前缀
    output_path = os.path.join(output_dir, f"resnet101_heatmap_{image_name}")
    plt.savefig(output_path)
    print(f"Analysis complete! Heatmap saved as {output_path}")

if __name__ == '__main__':
    main()