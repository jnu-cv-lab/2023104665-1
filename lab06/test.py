import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ====================== 路径设置（确保图片在lab06文件夹） ======================
# 获取当前脚本所在目录（lab06文件夹）
current_dir = os.path.dirname(os.path.abspath(__file__))
img1_path = os.path.join(current_dir, "box.png")
img2_path = os.path.join(current_dir, "box_in_scene.png")

# ====================== 读取图像 ======================
img1 = cv2.imread(img1_path, 0)          # 模板图
img2 = cv2.imread(img2_path, 0)          # 场景图

# 检查是否读取成功
if img1 is None or img2 is None:
    raise FileNotFoundError(f"无法读取图片，请检查路径是否正确：\n{img1_path}\n{img2_path}")

# ====================== 任务1：ORB 特征检测 ======================
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 可视化关键点
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)

# 保存图片到lab06文件夹
cv2.imwrite(os.path.join(current_dir, "box_keypoints.png"), img1_kp)
cv2.imwrite(os.path.join(current_dir, "scene_keypoints.png"), img2_kp)

# 输出信息
print("===== 任务1 结果 =====")
print(f"box 关键点数量：{len(kp1)}")
print(f"场景关键点数量：{len(kp2)}")
print(f"描述子维度：{des1.shape[1]}")

# ====================== 任务2：ORB 特征匹配 ======================
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 显示前50个匹配
img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
cv2.imwrite(os.path.join(current_dir, "orb_matches.png"), img_match)

print("\n===== 任务2 结果 =====")
print(f"总匹配数：{len(matches)}")

# ====================== 任务3：RANSAC 剔除误匹配 ======================
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matches_mask = mask.ravel().tolist()
inliers = sum(matches_mask)
inlier_ratio = inliers / len(matches)

# 绘制内点匹配
img_ransac = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                             matchesMask=matches_mask, flags=2)
cv2.imwrite(os.path.join(current_dir, "ransac_result.png"), img_ransac)

print("\n===== 任务3 结果 =====")
print(f"单应矩阵 H：\n{H}")
print(f"总匹配数：{len(matches)}")
print(f"RANSAC 内点数：{inliers}")
print(f"内点比例：{inlier_ratio:.2f}")

# ====================== 任务4：目标定位 ======================
h, w = img1.shape
pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts, H)

# 在场景图绘制边框
img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
img2_detect = cv2.polylines(img2_color, [np.int32(dst)], True, (0,255,0), 3)
cv2.imwrite(os.path.join(current_dir, "detection_result.png"), img2_detect)

print("\n===== 任务4 结果 =====")
print("目标定位成功，绿色框已标出物体位置")

# ====================== 任务6：参数对比实验 ======================
def test_orb(nfeatures):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1_t, des1_t = orb.detectAndCompute(img1, None)
    kp2_t, des2_t = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_t = bf.match(des1_t, des2_t)
    src = np.float32([kp1_t[m.queryIdx].pt for m in matches_t]).reshape(-1,1,2)
    dst = np.float32([kp2_t[m.trainIdx].pt for m in matches_t]).reshape(-1,1,2)
    H_t, mask_t = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inl = sum(mask_t.ravel())
    ratio = inl / len(matches_t) if len(matches_t) > 0 else 0
    success = ratio > 0.5
    return len(kp1_t), len(kp2_t), len(matches_t), inl, ratio, success

print("\n===== 任务6 参数对比 =====")
for n in [500,1000,2000]:
    k1,k2,m,i,r,s = test_orb(n)
    print(f"n={n}: k1={k1}, k2={k2}, 匹配={m}, 内点={i}, 比例={r:.2f}, 定位={s}")

# ====================== 选做：SIFT 对比 ======================
sift = cv2.SIFT_create()
kp_s1, des_s1 = sift.detectAndCompute(img1, None)
kp_s2, des_s2 = sift.detectAndCompute(img2, None)

bf_s = cv2.BFMatcher(cv2.NORM_L2)
matches_s = bf_s.knnMatch(des_s1, des_s2, k=2)
good_s = []
for m,n in matches_s:
    if m.distance < 0.75*n.distance:
        good_s.append(m)

src_s = np.float32([kp_s1[m.queryIdx].pt for m in good_s]).reshape(-1,1,2)
dst_s = np.float32([kp_s2[m.trainIdx].pt for m in good_s]).reshape(-1,1,2)
H_s, mask_s = cv2.findHomography(src_s, dst_s, cv2.RANSAC, 5.0)
inl_s = sum(mask_s.ravel())
ratio_s = inl_s / len(good_s) if len(good_s) > 0 else 0

print("\n===== SIFT 结果 =====")
print(f"匹配数：{len(good_s)}, 内点：{inl_s}, 比例：{ratio_s:.2f}")

# 显示所有结果
plt.figure(figsize=(12,8))
plt.subplot(221), plt.imshow(img1_kp, cmap='gray'), plt.title('Task1: Keypoints')
plt.subplot(222), plt.imshow(img_match, cmap='gray'), plt.title('Task2: Matches')
plt.subplot(223), plt.imshow(img_ransac, cmap='gray'), plt.title('Task3: RANSAC')
plt.subplot(224), plt.imshow(img2_detect), plt.title('Task4: Detection')
plt.show()