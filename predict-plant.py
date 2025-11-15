import os
import json
import time

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model__both_enhanced02 import DualAttentionAlexNet


def calculate_metrics(model, input_size=(1, 3, 224, 224), device='cuda', test_times=100):
    """计算模型的Params, FLOPs, Speed等指标"""
    model.eval()

    # 创建伪输入
    dummy_input = torch.randn(input_size).to(device)

    metrics = {}

    # 1. 计算参数量 (Params) - 使用PyTorch内置方法
    print("Calculating Parameters...")
    total_params = sum(p.numel() for p in model.parameters())
    metrics['Params'] = total_params
    metrics['Params_M'] = total_params / 1e6

    # 2. 估算FLOPs（近似值）
    print("Estimating FLOPs...")
    # 对于CNN，FLOPs ≈ 2 * Params（这是一个经验估算，对于精确值建议安装thop）
    metrics['FLOPs'] = total_params * 2  # 近似估算
    metrics['FLOPs_G'] = metrics['FLOPs'] / 1e9

    # 3. 计算推理速度 (Speed)
    print("Measuring inference speed...")

    # 预热
    for _ in range(10):
        _ = model(dummy_input)

    # 正式测速
    timings = []
    with torch.no_grad():
        for _ in range(test_times):
            start_time = time.time()
            _ = model(dummy_input)

            # 如果是CUDA，同步操作
            if device == 'cuda':
                torch.cuda.synchronize()

            end_time = time.time()
            timings.append((end_time - start_time) * 1000)  # 转换为毫秒

    metrics['Speed'] = sum(timings) / len(timings)
    metrics['Speed_std'] = torch.tensor(timings).std().item()  # 标准差

    return metrics


def manual_flops_calculation(model, input_size=(224, 224)):
    """手动估算FLOPs（如果需要更精确的估算）"""
    # 这是一个简化的FLOPs估算方法
    # 对于精确计算，强烈建议安装thop

    # 估算每层的FLOPs
    total_flops = 0

    # 遍历模型的所有层进行估算
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            # Conv2d FLOPs = 2 * (kernel_h * kernel_w * in_channels) * out_channels * output_h * output_w
            # 这里使用简化估算
            if hasattr(module, 'weight'):
                kernel_params = module.weight.numel()
                output_size = input_size[0] * input_size[1]  # 近似输出尺寸
                total_flops += 2 * kernel_params * output_size

        elif isinstance(module, torch.nn.Linear):
            # Linear FLOPs = 2 * in_features * out_features
            if hasattr(module, 'weight'):
                total_flops += 2 * module.weight.numel()

    return total_flops


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device for inference")

    # 图像预处理
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载测试图像
    img_path = r"D:\deep_leaning\Alexnet\data\test\Target spot\Target spot1602.jpg"
    assert os.path.exists(img_path), f"File '{img_path}' does not exist."

    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    plt.imshow(img)

    # 图像预处理
    try:
        img_tensor = data_transform(img)
        img_tensor = torch.unsqueeze(img_tensor, dim=0)
    except Exception as e:
        print(f"Error transforming image: {e}")
        return

    # 加载类别标签
    json_path = r'D:\deep_leaning\Alexnet\class_02indices.json'
    assert os.path.exists(json_path), f"Class indices file '{json_path}' does not exist."

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    num_classes = len(class_indict)
    print(f"Loaded {num_classes} classes: {class_indict}")

    # 初始化模型
    model = DualAttentionAlexNet(num_classes=num_classes).to(device)

    # 加载训练好的权重
    weights_path = r"D:\deep_leaning\Alexnet\DualAttentionAlexNet_Plant.pth"
    assert os.path.exists(weights_path), f"Model weights '{weights_path}' does not exist."

    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Successfully loaded weights from {weights_path}")
    except RuntimeError as e:
        print(f"Error loading weights: {e}")
        print("Attempting partial loading...")

        model_dict = model.state_dict()
        pretrained_dict = torch.load(weights_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        loaded_params = len(pretrained_dict)
        total_params = len(model_dict)
        print(f"Partially loaded weights: {loaded_params}/{total_params} parameters matched")

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE METRICS CALCULATION")
    print("=" * 60)

    # 计算性能指标
    metrics = calculate_metrics(model, device=device, test_times=100)

    # 打印指标结果
    print("\n📊 PERFORMANCE METRICS RESULTS:")
    print(f"• Parameters: {metrics['Params_M']:.2f} M")
    print(f"• FLOPs (estimated): {metrics['FLOPs_G']:.2f} G")
    print(f"• Inference Speed: {metrics['Speed']:.2f} ± {metrics['Speed_std']:.2f} ms")

    # 注意：准确率需要测试集数据，这里显示占位符
    print("• Accuracy: [需要测试集数据计算]")

    print("\n" + "=" * 60)
    print("INFERENCE ON SAMPLE IMAGE")
    print("=" * 60)

    # 单张图像推理
    model.eval()
    with torch.no_grad():
        try:
            output = torch.squeeze(model(img_tensor.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).item()
        except Exception as e:
            print(f"Error during prediction: {e}")
            return

    # 显示预测结果
    print("\n🎯 PREDICTION RESULT:")
    print(f"Most likely class: {class_indict[str(predict_cla)]}")
    print(f"Probability: {predict[predict_cla].item():.3f}")

    print("\nAll class probabilities:")
    for i in range(len(predict)):
        print(f"Class {i:2} ({class_indict[str(i)]:15}): {predict[i].item():.4f}")

    # 在图像上显示预测结果
    plt.title(f"Prediction: {class_indict[str(predict_cla)]} ({predict[predict_cla].item():.2f})")
    plt.axis('off')

    # 保存结果图像
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, os.path.basename(img_path))
    plt.savefig(result_path)
    print(f"\n💾 Result saved to: {result_path}")


if __name__ == '__main__':
    main()