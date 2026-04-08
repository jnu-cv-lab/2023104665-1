import cv2
import numpy as np
import matplotlib.pyplot as plt
import os  # 导入os模块，用于创建文件夹

# ---------------------- 1. 读取灰度图像 ----------------------
img_path = "C:\cv-course\lab03\work3.jpg"  # 你的图片路径
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("图片读取失败，请检查路径")

h, w = img.shape
print(f"原图尺寸：{w}x{h}")

# ---------------------- 2. 下采样（缩小为1/2） ----------------------
# 方法1：不做预滤波直接缩小
img_down_no_filter = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)

# 方法2：先高斯平滑再缩小
img_blur = cv2.GaussianBlur(img, (3,3), sigmaX=1.0)
img_down_filtered = cv2.resize(img_blur, (w//2, h//2), interpolation=cv2.INTER_NEAREST)

# ---------------------- 3. 插值恢复到原尺寸 ----------------------
# 定义插值方法
methods = {
    "Nearest": cv2.INTER_NEAREST,
    "Bilinear": cv2.INTER_LINEAR,
    "Bicubic": cv2.INTER_CUBIC
}

restored = {}
for name, method in methods.items():
    restored[name] = cv2.resize(img_down_filtered, (w, h), interpolation=method)

# ---------------------- 4. 空域对比：MSE / PSNR ----------------------
def calc_mse(img1, img2):
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64))**2)

def calc_psnr(img1, img2):
    mse = calc_mse(img1, img2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255**2 / mse)

print("\n==== 空域对比（以高斯平滑后下采样的恢复结果为例） ====")
for name, rimg in restored.items():
    mse = calc_mse(img, rimg)
    psnr = calc_psnr(img, rimg)
    print(f"{name} 恢复结果：MSE={mse:.2f}, PSNR={psnr:.2f} dB")

# ---------------------- 5. 傅里叶变换（FFT） ----------------------
def fft_analysis(image):
    f = np.fft.fft2(image.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    # 归一化到0-255，方便保存为图像
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return magnitude

fft_origin = fft_analysis(img)
fft_down = fft_analysis(img_down_filtered)
fft_bilinear = fft_analysis(restored["Bilinear"])

# ---------------------- 6. DCT 分析 ----------------------
def dct_analysis(image, block_size=8):
    img_float = image.astype(np.float32) / 255.0
    dct = cv2.dct(img_float)
    total_energy = np.sum(dct**2)
    low_freq_energy = np.sum(dct[:block_size, :block_size]**2)
    ratio = low_freq_energy / total_energy
    # 归一化DCT系数，方便保存
    dct_norm = cv2.normalize(np.abs(dct), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return dct_norm, ratio

print("\n==== DCT 能量分析 ====")
dct_origin, ratio_origin = dct_analysis(img)
print(f"原图DCT低频能量占比：{ratio_origin:.4f}")

dct_results = {}
for name, rimg in restored.items():
    dct_norm, ratio = dct_analysis(rimg)
    dct_results[name] = dct_norm
    print(f"{name}恢复结果DCT低频能量占比：{ratio:.4f}")

# ---------------------- 关键步骤：保存图像到 lab03 文件夹 ----------------------
# 1. 定义保存路径（自动创建文件夹，避免报错）
save_dir = "lab03"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print(f"\n开始保存图像到 {save_dir} 文件夹...")

# 2. 保存所有处理后的图像
cv2.imwrite(os.path.join(save_dir, "original.png"), img)
cv2.imwrite(os.path.join(save_dir, "down_no_filter.png"), img_down_no_filter)
cv2.imwrite(os.path.join(save_dir, "down_filtered.png"), img_down_filtered)

for name, rimg in restored.items():
    cv2.imwrite(os.path.join(save_dir, f"restored_{name}.png"), rimg)

cv2.imwrite(os.path.join(save_dir, "fft_origin.png"), fft_origin)
cv2.imwrite(os.path.join(save_dir, "fft_down.png"), fft_down)
cv2.imwrite(os.path.join(save_dir, "fft_bilinear.png"), fft_bilinear)

cv2.imwrite(os.path.join(save_dir, "dct_origin.png"), dct_origin)
for name, dct_img in dct_results.items():
    cv2.imwrite(os.path.join(save_dir, f"dct_{name}.png"), dct_img)

print("所有图像保存完成！")