import os
import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.psnr import PeakSignalNoiseRatio as PSNR
from pytorch_fid import fid_score
import pandas as pd

# 计算PSNR
def compute_psnr(image1, image2):
    global psnr_metric  # 声明为全局变量
    image1 = torch.tensor(image1).unsqueeze(0) if isinstance(image1, np.ndarray) else image1
    image2 = torch.tensor(image2).unsqueeze(0) if isinstance(image2, np.ndarray) else image2
    return psnr_metric(image1, image2)

# 计算SSIM
def compute_ssim(image1, image2):
    image1 = np.array(image1) if isinstance(image1, torch.Tensor) else image1
    image2 = np.array(image2) if isinstance(image2, torch.Tensor) else image2
    return ssim(image1, image2, multichannel=True)

# 计算LPIPS
def compute_lpips(image1, image2):
    global lpips_model  # 声明为全局变量
    # 将图像转换为 PyTorch tensor 并确保其形状符合要求 (BGR -> RGB, HWC -> CHW)
    image1 = torch.tensor(image1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image2 = torch.tensor(image2).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 计算 LPIPS 值
    return lpips_model(image1, image2).item()

# 计算FID
def compute_fid(real_images, fake_images):
    print("Computing FID...")
    real_dir = './real_images'
    fake_dir = './fake_images'
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    for i, img in enumerate(real_images):
        cv2.imwrite(os.path.join(real_dir, f"real_{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    for i, img in enumerate(fake_images):
        cv2.imwrite(os.path.join(fake_dir, f"fake_{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    fid_value = fid_score.calculate_fid_given_paths([real_dir, fake_dir], batch_size=1, device='cuda', dims=2048)

    # 删除临时文件夹
    for img in os.listdir(real_dir):
        os.remove(os.path.join(real_dir, img))
    for img in os.listdir(fake_dir):
        os.remove(os.path.join(fake_dir, img))
    os.rmdir(real_dir)
    os.rmdir(fake_dir)

    print(f"FID computed: {fid_value}")
    return fid_value

def compute_lse_c(real_images, fake_images):
    print("Computing LSE-C...")
    return np.mean((real_images - fake_images) ** 2)

def compute_lse_d(real_images, fake_images):
    print("Computing LSE-D...")
    return np.mean(np.abs(real_images - fake_images))

# 评估单个视频的质量（PSNR, SSIM, LPIPS, FID, LSE-C, LSE-D）
def evaluate_video_quality(real_video_path, fake_video_path):
    print(f"Evaluating: {real_video_path} vs {fake_video_path}")
    real_video = cv2.VideoCapture(real_video_path)
    fake_video = cv2.VideoCapture(fake_video_path)

    real_frame_count = int(real_video.get(cv2.CAP_PROP_FRAME_COUNT))
    fake_frame_count = int(fake_video.get(cv2.CAP_PROP_FRAME_COUNT))

    min_frame_count = min(real_frame_count, fake_frame_count)

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []  # 修改为 LPIPS
    frame_idx = 0
    real_images = []
    fake_images = []

    while True:
        ret_real, frame_real = real_video.read()
        ret_fake, frame_fake = fake_video.read()

        if ret_real and ret_fake:
            frame_real_rgb = cv2.cvtColor(frame_real, cv2.COLOR_BGR2RGB)
            frame_fake_rgb = cv2.cvtColor(frame_fake, cv2.COLOR_BGR2RGB)

            # 计算PSNR
            psnr_scores.append(compute_psnr(frame_real_rgb, frame_fake_rgb))
            # 计算SSIM
            ssim_scores.append(compute_ssim(frame_real_rgb, frame_fake_rgb))
            real_images.append(frame_real_rgb)
            fake_images.append(frame_fake_rgb)
            # 计算LPIPS
            lpips_scores.append(compute_lpips(frame_real_rgb, frame_fake_rgb))
            frame_idx += 1

        elif ret_fake:
            lpips_scores.append(compute_lpips(frame_fake_rgb, frame_fake_rgb))
        else:
            break

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_lpips = np.mean(lpips_scores)  # 修改为 LPIPS 平均值

    fid_value = compute_fid(np.array(real_images), np.array(fake_images))
    lse_c_score = compute_lse_c(np.array(real_images), np.array(fake_images))
    lse_d_score = compute_lse_d(np.array(real_images), np.array(fake_images))

    real_video.release()
    fake_video.release()

    print(f"Video evaluated. PSNR: {avg_psnr}, SSIM: {avg_ssim}, LPIPS: {avg_lpips}, FID: {fid_value}, LSE-C: {lse_c_score}, LSE-D: {lse_d_score}")
    
    return avg_psnr, avg_ssim, avg_lpips, fid_value, lse_c_score, lse_d_score

# 评估整个测试集
def evaluate_test_set(real_video_folder, fake_video_folder,evaluate_dir):
    print("Evaluating test set...")
    video_type = "_facial_dubbing_add_audio.mp4"
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []  # 修改为 LPIPS
    fid_scores = []
    lse_c_scores = []
    lse_d_scores = []

    fake_video_list = sorted(os.listdir(fake_video_folder))

    for fake_video_file in fake_video_list:
        if not fake_video_file.endswith(video_type):
            continue

        part1 = fake_video_file[:-len(video_type)]

        real_video_path = os.path.join(real_video_folder, part1 + '.mp4')
        fake_video_path = os.path.join(fake_video_folder, fake_video_file)

        if not os.path.exists(real_video_path):
            print(f"Warning: Real video {real_video_path} not found for {fake_video_path}. Skipping...")
            continue

        avg_psnr, avg_ssim, avg_lpips, fid_score, lse_c_score, lse_d_score = evaluate_video_quality(real_video_path, fake_video_path)

        psnr_scores.append(avg_psnr)
        ssim_scores.append(avg_ssim)
        lpips_scores.append(avg_lpips)  # 修改为 LPIPS
        fid_scores.append(fid_score)
        lse_c_scores.append(lse_c_score)
        lse_d_scores.append(lse_d_score)

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_lpips = np.mean(lpips_scores)  # 修改为 LPIPS
    avg_fid = np.mean(fid_scores)
    avg_lse_c = np.mean(lse_c_scores)
    avg_lse_d = np.mean(lse_d_scores)

    print(f"Final Results for Test Set:")
    print(f"PSNR: {avg_psnr}, SSIM: {avg_ssim}, LPIPS: {avg_lpips}, FID: {avg_fid}, LSE-C: {avg_lse_c}, LSE-D: {avg_lse_d}")
    data = {
        'PSNR': [avg_psnr],
        'SSIM': [avg_ssim],
        'LPIPS': [avg_lpips],
        'FID': [avg_fid],
        'LSE-C': [avg_lse_c],
        'LSE-D': [avg_lse_d]
    }
    df = pd.DataFrame(data)
    evaluate_path=os.path.join(evaluate_dir,'evaluate.csv')
    df.to_csv(evaluate_path, index=False)

# 主程序
if __name__ == "__main__":
    print("Starting evaluation...")
    # 初始化全局变量
    lpips_model = lpips.LPIPS(net='alex')  # 加载 LPIPS 模型
    psnr_metric = PSNR()  # 加载 PSNR 计算工具

    dir = os.path.dirname(os.path.abspath(__file__))

    real_video_folder = os.path.join(dir, 'asserts', 'test_data', "split_video_25fps")
    fake_video_folder = os.path.join(dir, 'asserts', 'inference_result')
    evaluate_dir=os.path.join(dir, 'asserts', 'evaluate_result')
    if not os.path.exists(evaluate_dir):
        os.makedirs(evaluate_dir)
        print(f'create: {evaluate_dir}')
    # 进行测试集的定性评估
    evaluate_test_set(real_video_folder, fake_video_folder,evaluate_dir=evaluate_dir)
