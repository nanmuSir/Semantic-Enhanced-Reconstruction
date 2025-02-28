import os
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
from scipy.stats import binom

# 设置输入文件夹路径
test_dir = '/home/work/ZH/StableDiffusionReconstruction-main/image/test_img'  # GT feature
gt_images_dir = '/home/work/ZH/StableDiffusionReconstruction-main/image/test_img'# GT img
feats_dir = '/home/work/ZH/StableDiffusionReconstruction-main/image/result_img/deepseek_subj07'  # feature
images_dir = '/home/work/ZH/StableDiffusionReconstruction-main/image/result_img/deepseek_subj07'  # img

num_test =

# 定义函数提取特征相似性
# def compute_similarity(gen_features, gt_features):
#     # 计算ResNet50特征相似度
#     resnet_similarity = euclidean(gen_features.flatten(), gt_features.flatten())
#     return resnet_similarity
def pairwise_corr_all(ground_truth, predictions):
    r = np.corrcoef(ground_truth, predictions)
    r = r[:len(ground_truth), len(ground_truth):]  # rows: groundtruth, columns: predicitons
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    perf = np.mean(success_cnt) / (len(ground_truth) - 1)
    p = 1 - binom.cdf(perf * len(ground_truth) * (len(ground_truth) - 1), len(ground_truth) * (len(ground_truth) - 1),
                      0.5)
    return perf, p

# 计算MSE
def compute_mse(image1, image2):
    return mean_squared_error(image1.flatten(), image2.flatten())

# 计算MSE和ResNet50特征相似度
mse_list = []
resnet_similarity_list = []

# 加载特征文件
gen_feat_file = os.path.join(feats_dir, 'resnet50_fc.npy')
gt_feat_file = os.path.join(test_dir, 'resnet50_fc.npy')

gen_features = np.load(gen_feat_file)
gt_features = np.load(gt_feat_file)

gen_feat = gen_features.reshape((len(gen_features), -1))
gt_feat = gt_features.reshape((len(gt_features), -1))

pairwise_corr = pairwise_corr_all(gt_feat[:num_test], gen_feat[:num_test])[0]
resnet_similarity_list.append(pairwise_corr)

for i in range(num_test):
    # 计算MSE
    gen_image_path = os.path.join(images_dir, f'{i+1}.png')
    gt_image_path = os.path.join(gt_images_dir, f'{i+1}.png')

    gen_image = Image.open(gen_image_path).resize((425, 425))
    gt_image = Image.open(gt_image_path).resize((425, 425))

    gen_image = np.array(gen_image) / 255.0
    gt_image = np.array(gt_image) / 255.0

    mse = compute_mse(gen_image, gt_image)
    mse_list.append(mse)

    # # 计算ResNet50特征相似度
    # resnet_similarity = compute_similarity(gen_features[i], gt_features[i])
    # resnet_similarity_list.append(resnet_similarity)

mse_list = np.array(mse_list)
#resnet_similarity_list = np.array(resnet_similarity_list)

# 输出统计结果
print('ResNet50 Feature Similarity: {}'.format(resnet_similarity_list))
print('MSE: {}'.format(mse_list.mean()))
print('MSE min: {}, max: {}'.format(mse_list.min(), mse_list.max()))
#print('ResNet50 Feature Similarity min: {}, max: {}'.format(resnet_similarity_list.min(), resnet_similarity_list.max()))

# # 绘制直方图
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.hist(mse_list, bins=30, color='blue', alpha=0.7)
# plt.title('MSE Distribution')
# plt.xlabel('MSE')
# plt.ylabel('Frequency')
#
# plt.subplot(1, 2, 2)
# plt.hist(resnet_similarity_list, bins=30, color='green', alpha=0.7)
# plt.title('ResNet50 Feature Similarity Distribution')
# plt.xlabel('Euclidean Distance')
# plt.ylabel('Frequency')
#
# plt.tight_layout()
# plt.show()
