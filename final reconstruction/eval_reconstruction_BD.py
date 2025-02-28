import os
import numpy as np
import scipy.io as spio
import scipy as sp
from PIL import Image
from scipy.stats import pearsonr, binom
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim

# 设置输入文件夹路径
test_dir = '/home/work/ZH/StableDiffusionReconstruction-main/image/test_img'  # 替换为您的测试特征文件夹路径
feats_dir = '/home/work/ZH/StableDiffusionReconstruction-main/image/result_img/deepseek_subj01'  # 替换为您的评估特征文件夹路径
images_dir = '/home/work/ZH/StableDiffusionReconstruction-main/image/result_img/deepseek_subj01'  # 替换为您的输入图像文件夹路径

num_test =
#sub = 1  # 设置受试者编号

# 定义函数计算特征相似性
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

# 定义网络和层列表
net_list = [
    ('inceptionv3', 'avgpool'),
    ('clip', 'final'),
    ('alexnet', 2),
    ('alexnet', 5),
    ('efficientnet', 'avgpool'),
    ('swav', 'avgpool')
]

distance_fn = sp.spatial.distance.correlation
pairwise_corrs = []

for (net_name, layer) in net_list:
    test_file_name = os.path.join(test_dir, f'{net_name}_{layer}.npy')
    eval_file_name = os.path.join(feats_dir, f'{net_name}_{layer}.npy')

    gt_feat = np.load(test_file_name)
    eval_feat = np.load(eval_file_name)

    gt_feat = gt_feat.reshape((len(gt_feat), -1))
    eval_feat = eval_feat.reshape((len(eval_feat), -1))

    print(net_name, layer)
    if net_name in ['efficientnet', 'swav']:
        distances = np.array([distance_fn(gt_feat[i], eval_feat[i]) for i in range(num_test)]).mean()
        print('distance: ', distances)
    else:
        pairwise_corr = pairwise_corr_all(gt_feat[:num_test], eval_feat[:num_test])[0]
        pairwise_corrs.append(pairwise_corr)
        print('pairwise corr: ', pairwise_corr)

# 计算SSIM和PixCorr
ssim_list = []
pixcorr_list = []

for i in range(num_test):
    gen_image_path = os.path.join(images_dir, f'{i+1}.png')
    gt_image_path = os.path.join('/home/work/ZH/StableDiffusionReconstruction-main/image/test_img', f'{i+1}.png')

    gen_image = Image.open(gen_image_path).resize((425, 425))
    gt_image = Image.open(gt_image_path)

    gen_image = np.array(gen_image) / 255.0
    gt_image = np.array(gt_image) / 255.0

    pixcorr_res = np.corrcoef(gt_image.reshape(1, -1), gen_image.reshape(1, -1))[0, 1]
    pixcorr_list.append(pixcorr_res)

    gen_image = rgb2gray(gen_image)
    gt_image = rgb2gray(gt_image)

    ssim_res = ssim(gen_image, gt_image, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
    ssim_list.append(ssim_res)

ssim_list = np.array(ssim_list)
pixcorr_list = np.array(pixcorr_list)

print('PixCorr: {}'.format(pixcorr_list.mean()))
print('SSIM: {}'.format(ssim_list.mean()))
