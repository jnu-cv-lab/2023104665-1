import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 确保保存目录存在
save_dir = "lab04"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ====================== 工具函数：保存图片 ======================
def save_fig(filename):
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"已保存: {path}")

# ====================== 图像显示 + 自动保存 ======================
def show_images(images, titles, cols=2, savename=None):
    n = len(images)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 4, rows * 3.5))

    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        if len(images[i].shape) == 2:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(images[i])
        plt.title(titles[i], fontsize=10)
        plt.axis('off')

    plt.tight_layout()

    # 自动保存
    if savename:
        save_fig(savename)

    plt.show()

# ====================== 第一部分 ======================
def part1_downsampling_test():
    print("===== 第一部分：下采样与混叠 =====")
    size = 512
    chessboard = np.zeros((size, size), dtype=np.uint8)
    block_size = 8
    for i in range(size):
        for j in range(size):
            if (i // block_size + j // block_size) % 2 == 0:
                chessboard[i, j] = 255

    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    chirp = np.sin(2 * np.pi * (10 * xx + 100 * xx**2))
    chirp = (chirp - chirp.min()) / (chirp.max() - chirp.min()) * 255
    chirp = chirp.astype(np.uint8)

    chess_direct = chessboard[::4, ::4]
    chirp_direct = chirp[::4, ::4]

    sigma = 1.8
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    chess_blur = cv2.GaussianBlur(chessboard, (kernel_size, kernel_size), sigma)
    chess_blur_down = chess_blur[::4, ::4]
    chirp_blur = cv2.GaussianBlur(chirp, (kernel_size, kernel_size), sigma)
    chirp_blur_down = chirp_blur[::4, ::4]

    # 保存第一部分结果
    show_images(
        [chessboard, chess_direct, chess_blur_down, chirp, chirp_direct, chirp_blur_down],
        ["原始棋盘格", "直接下采样", "滤波后下采样", "原始Chirp", "直接下采样Chirp", "滤波后Chirp"],
        cols=3,
        savename="part1_chessboard_chirp.png"
    )

# ====================== 第二部分 ======================
def part2_sigma_validation(M=4):
    print(f"===== 第二部分：σ验证（M={M}） =====")
    size = 512
    chessboard = np.zeros((size, size), dtype=np.uint8)
    block_size = 8
    for i in range(size):
        for j in range(size):
            if (i // block_size + j // block_size) % 2 == 0:
                chessboard[i, j] = 255

    sigma_list = [0.5, 1.0, 2.0, 4.0]
    theory_sigma = 0.45 * M

    results = []
    titles = []

    for sigma in sigma_list:
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        blur = cv2.GaussianBlur(chessboard, (kernel_size, kernel_size), sigma)
        down = blur[::M, ::M]
        results.append(down)
        titles.append(f"σ={sigma:.1f}")

    # 理论最优
    kernel_size = int(6 * theory_sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    blur_theory = cv2.GaussianBlur(chessboard, (kernel_size, kernel_size), theory_sigma)
    down_theory = blur_theory[::M, ::M]
    results.append(down_theory)
    titles.append(f"理论σ={theory_sigma:.2f}")

    show_images(results, titles, cols=3, savename="part2_sigma_comparison.png")

# ====================== 第三部分 ======================
def part3_adaptive_downsampling():
    print("===== 第三部分：自适应下采样 =====")
    img = cv2.imread("test_image.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        size = 512
        img = np.zeros((size, size), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (200, 200), 255, -1)
        cv2.rectangle(img, (250, 50), (450, 200), 127, -1)
        cv2.line(img, (0, 250), (512, 250), 255, 3)
        cv2.line(img, (256, 0), (256, 512), 255, 3)
        cv2.circle(img, (128, 384), 80, 255, -1)

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_mag = cv2.normalize(grad_mag, None, 0, 1, cv2.NORM_MINMAX)

    M_map = np.where(grad_mag > 0.3, 2, 4)
    sigma_map = 0.45 * M_map

    h, w = img.shape
    adaptive_down = np.zeros((h//2, w//2), dtype=np.uint8)
    uniform_down = cv2.GaussianBlur(img, (int(6*1.8+1), int(6*1.8+1)), 1.8)[::4, ::4]

    block_size = 16
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block_M = M_map[i:i+block_size, j:j+block_size].mean()
            sigma = 0.45 * block_M
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1

            block = img[i:i+block_size, j:j+block_size]
            block_blur = cv2.GaussianBlur(block, (kernel_size, kernel_size), sigma)
            down_block = block_blur[::int(block_M), ::int(block_M)]

            adaptive_down[i//2:i//2+down_block.shape[0], j//2:j//2+down_block.shape[1]] = down_block

    uniform_up = cv2.resize(uniform_down, (w, h), interpolation=cv2.INTER_NEAREST)
    uniform_error = np.abs(img - uniform_up)
    adaptive_up = cv2.resize(adaptive_down, (w, h), interpolation=cv2.INTER_NEAREST)
    adaptive_error = np.abs(img - adaptive_up)

    # 保存第三部分结果
    show_images(
        [img, grad_mag, M_map, sigma_map],
        ["原始图像", "梯度幅值", "局部M图", "局部σ图"],
        cols=2,
        savename="part3_grad_m_sigma.png"
    )

    show_images(
        [uniform_down, adaptive_down, uniform_error, adaptive_error],
        ["统一M=4", "自适应下采样", "统一采样误差", "自适应误差"],
        cols=2,
        savename="part3_downsampling_comparison.png"
    )

# ====================== 主函数 ======================
if __name__ == "__main__":
    part1_downsampling_test()
    part2_sigma_validation(M=4)
    part3_adaptive_downsampling()