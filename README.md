# ACHC-for-Tomato-Disease-Identification
Tomato Disease Identification
1. 研究背景与模型定位
作为全球重要的经济作物，番茄的产量和品质易受真菌、细菌、病毒等病害的影响，严重威胁农业生产和经济效益。其叶片病害（如斑枯病、轮斑病、早疫病）易导致光合效率骤降、产量损失达10%-30%，传统人工检测存在整体流程耗时、成本高昂，难以满足现代农业对病害快速响应与大规模监测的需求等问题。

本文提出一种基于改进AlexNet和融合Coordinate Attention 和Hybrid Channel-Spatial Attention多注意力机制的深度学习番茄叶片病害识别模型ACHC。首先，结合番茄病害叶片特性优化经典的AlexNet，其次，在Conv1和Conv3后嵌入Coordinate Attention模块，强化病斑空间的定位能力。接着，在Conv2和Conv4后融合Hybrid Channel-Spatial Attention模块，提高通道特征区分度。解决番茄叶片病害“局部病灶特征模糊（如早期微小锈孢子堆）、全局纹理建模效率低、相似病害区分难”的核心问题，提出的方法达到99.61%的平均测试正确率，当前提出的策略对于提升番茄种植的自动化监测与精准病害防控具有重要的参考意义。

2. ACHC 核心创新点

(1)优化AlexNet：基于对番茄病斑特征分析，移除AlexNet中冗余的Conv3层，将五层卷积压缩为四层，删除第二个全连接层（FC2），将三阶全连接结构简化为两阶，参数量减少；
(2)新型双注意力融合：创新性地整合了Hybrid Channel-Spatial Attention（混合通道-空间注意力）和CoordAttention（坐标注意力）机制，提出基于病斑显著性的动态权重融合策略，实现计算资源的动态优化分配；
(3)提升模型综合性能：通过网络架构的精简与注意力机制的创新融合，不仅优化了模型的计算效率，还增强了模型的特征提取能力与对复杂场景的适应能力。
通过逐步下采样特征图，构建多尺度特征表示，适配小麦叶片不同大小的病害区域（从早期毫米级微小病斑到后期厘米级大面积发病区域）。

效率与精度平衡：
相较于传统的Vgg16和EfficientNetB0相比，提出的方法在计算量（FLOPs）上分别降低了约99.5%和77%以上，同时在准确率和速度上均取得了显著优势。

3. 实验数据集：TDD
3.1 数据集概况
本研究基于自建番茄叶片常见病害检测数据集（Tomato Disease Detection Dataset, TDD），数据集已随项目上传至仓库的 TDD/ 文件夹，无需额外下载。

数据集名称	包含类别	图像总数	图像分辨率	数据分布（训练:测试）
TDD	 Healthy  Leaf blight Target spot  Early blight	详见数据集说明文件	统一 resize 至 256×256	3:1（通过代码自动划分）
3.2 数据集结构
仓库中 TDD/ 文件夹组织如下，图像均采集于小麦主产区田间（涵盖不同生育期、光照与拍摄角度），可直接用于模型训练/测试：

TDD/
├── Healthy/       # 健康番茄植株图像（特征：叶片平展或略微向上翘起，充满活力, 呈现出均匀、鲜亮、深绿色的色调）
├── Leaf blight/       # 番茄斑枯病叶片图像（病斑特征：叶片上出现小型、不规则的水浸状褪绿小点，随后迅速扩大）
├── Target spot/       # 番茄轮斑病叶片图像（病斑特征：叶片上出现非常小的、水浸状的圆形小斑点,病斑扩大后，会形成非常明显的同心轮纹，直径变化较大，从几毫米到超过一厘米都有可能）
└── Early blight/     # 番茄早疫病植株图像（特征：叶片上出现深褐色或黑褐色的小斑点，通常为圆形或近圆形）
4. 实验环境配置
4.1 依赖安装
推荐使用Anaconda创建虚拟环境，确保依赖版本匹配（避免兼容性问题，尤其适配PyTorch 2.5.1）：

# 1. 创建并激活虚拟环境
conda create -n ACHC python=3.12.9
conda activate ACHC

# 2. 安装PyTorch与TorchVision（需适配CUDA版本，示例为CUDA 12.1；CPU用户可替换为cpu版本）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装其他依赖库（数据处理、可视化、模型工具等）
# pip 安装（推荐，速度快、兼容性强）
pip install numpy pandas scipy matplotlib seaborn opencv-python pillow tqdm

# conda 安装（适合Anaconda环境用户）
conda install -c conda-forge numpy pandas scipy matplotlib seaborn opencv pillow tqdm
4.2 硬件要求
GPU：推荐 NVIDIA GPU（显存≥8GB，如RTX 3060/4060，支持CUDA 11.8+），训练50轮耗时约2-3小时，显存占用峰值≤6GB；
CPU：支持推理测试（单张图像推理耗时约0.5-1秒），但训练耗时显著增加（约20-25小时），不推荐用于完整训练流程。
5. 实验结果
5.1 核心指标对比（TDD数据集）
ACHC与主流深度学习模型在番茄叶片病害分类任务上的性能对比如下，模型在精度、计算效率上均表现更优，尤其对相似病害（条锈病/叶锈病）的区分能力显著提升：

模型	              分类准确率（Accuracy）	  计算量（FLOPs）	   参数量（M）     推理速度（Speed）
MobileNet-V3	    	 83.97%	                 3.76G              1.52            4.63
EfficientNet-B0	     82.27%	                 30.8               4.01            7.27
Vgg16          	     93.48%	                 1476.46            134.27          7.64
ACHC（本文）	         99.61%	                 6.81               10.57           2.42
注：1.分类准确率（Accuracy）指模型预测结果与真实标签一致的样本比例（通常为百分比）常用于衡量模型 “预测性能”，数值越高，模型的分类 / 预测能力越强，效果越优；2. 计算量（FLOPs）指模型完成一次前向传播所需的浮点运算总数，用于衡量模型 “轻量化程度”，数值越小，模型体积越小、占用内存越少，越适合边缘设备（如手机、嵌入式设备）部署；3. 模型参数量M，是衡量模型 “轻量化程度”，数值越小，模型体积越小、占用内存越少，越适合边缘设备；4. 推理速度（Speed）指模型处理单个样本的平均耗时，单位为 “毫秒（ms）”，用于衡量模型 “响应效率”，数值越小，推理速度越快，越适合实时场景（如实时检测、语音交互）。

6. 代码使用说明
6.1 模型训练
运行 train.py 脚本启动训练，支持通过参数调整训练配置，示例命令（适配TDD数据集）：

python train.py \
  --data ./TDD \
  --epochs 100 \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --save_dir ./weights \
  --device cuda:0 \
  --log_interval 10  # 每10个batch打印一次训练日志
关键参数说明：
参数名	含义	默认值
--data_dir	  TDD数据集根目录路径	./TDD
--epochs	训练轮数	100
--batch_size	批次大小（根据GPU显存调整，8/16/32）	16
--lr	初始学习率	1e-4
--save_dir	训练权重保存目录	./weights
--device	训练设备（cuda:0 或 cpu）	cuda:0
训练输出：
训练过程中，模型会自动保存验证集准确率最高的权重，文件名为 best-model.pth；
训练日志（损失值、准确率）会实时打印。
6.2 模型预测
使用训练好的权重进行单张番茄叶片图像预测，运行 predict.py 脚本，示例命令：

python predict.py \
  --image_path ./TDD/Test/Target_spot/Target_spot1602.jpg \
  --model_path ./weights/DualAttentionAlexNet_Plant.pth \
  --class_json ./class_indices.json \
  --device cuda:0
预测输出示例：
📊 PERFORMANCE METRICS RESULTS:
• Parameters: 10.57 M
• FLOPs (estimated): 6.81 G
• Inference Speed: 2.42 ± 0.15 ms

🎯 PREDICTION RESULT:
Most likely class: Target spot
Probability: 0.996
6.3 预训练权重
提供基于 TDD 数据集训练完成的最优权重，可直接用于预测或微调。除随项目仓库附带的权重外，也可通过百度网盘获取完整权重文件：

百度网盘分享： 链接: https://pan.baidu.com/s/1CHoAXlxq5zYGMwRJijomkA 提取码: rcum（复制这段内容后打开百度网盘手机 App，操作更方便） 本地权重文件：DualAttentionAlexNet_Plant.pth（若仓库内权重存在大小限制，可通过上述网盘链接获取完整版本）； 适用场景：仅针对番茄叶片的 “斑枯病、轮斑病、早疫病、健康叶片” 四类分类，若需扩展其他番茄病害，建议基于此权重微调（冻结浅层注意力模块，仅训练分类头与深层特征融合层，可减少 50% 以上训练数据量）。

7. 项目文件结构
ACHC-for-Tomato-Disease-Identification/
├── TDD/                          # 番茄病害数据集
├── data/                         # 数据处理文件夹（可选）
│   ├── train/                   # 训练集
│   ├── val/                     # 验证集
│   └── test/                    # 测试集
├── models/                       # 模型定义文件
│   ├── model.py                 # 基础AlexNet
│   ├── model_optimized.py       # 优化版AlexNet
│   ├── model_cood_attention.py  # 添加坐标注意力模型
│   ├── model_hybridcs.py         #添加通道注意力模型
│   └── model__both_enhanced02.py # ACHC模型
├── train.py                      # 训练脚本
├── predict.py                    # 预测与性能评估脚本
├── class_indices.json           # 类别索引文件
└── README.md                     # 项目说明文档
9. 已知问题与注意事项
数据集适配：当前模型与权重仅针对“斑枯病、轮斑病、早疫病、健康叶片”四类，若新增病害类别，需补充对应数据集并重新训练（建议每类样本量≥500张，确保模型泛化性）；
图像分辨率：输入图像会自动resize至256×256，若原始图像分辨率过低（<128×128），可能导致早期微小病斑特征丢失，建议输入图像分辨率≥256×256；
CUDA版本问题：若安装PyTorch时出现CUDA不兼容，可替换为CPU版本（需将所有脚本的--device改为cpu），但训练效率会大幅下降。
10. 引用与联系方式
9.1 引用方式
论文处于投刊阶段，正式发表后将更新BibTeX引用格式，当前可临时引用：

@article{hh_former_wheat_disease,
  title={
VWLM: A Novel and High Accuracy Deep Learning model for Wheat Disease Identification },
  author={[作者姓名，待发表时补充]},
  journal={[期刊名称，待录用后补充]},
  year={2026},
  note={Manuscript submitted for publication}
}
9.2 联系方式
若遇到代码运行问题或学术交流需求，请联系：

邮箱：liyan@huuc.edu.cn
GitHub Issue：直接在本仓库提交Issue，会在1-3个工作日内回复。
