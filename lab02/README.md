# Lab02 基于OpenCV的图像处理基础实验
实验概述
本实验基于OpenCV 4.9.0，使用C++语言完成基础图像处理任务，覆盖图像读取、信息输出、格式转换、保存与裁剪等核心操作，完整满足课程作业要求。
一、实验目的
1.  掌握OpenCV在Windows环境下的配置与基础API使用
2.  实现图像的读取、显示与基础信息提取
3.  完成彩色图像转灰度图的格式转换与结果保存
4.  实现图像区域裁剪操作，完成作业要求的NumPy等价功能
5.  掌握C++项目编译、运行与GitHub代码提交流程
二、实验环境
| 环境项 | 配置详情 |
|--------|----------|
| 操作系统 | Windows 11 |
| 开发工具 | Visual Studio 2022 / VS Code |
| 编程语言 | C++ (C++17标准) |
| OpenCV版本 | 4.9.0 |
| 编译器 | MSVC v143 |
三、实验任务与完成情况
作业要求清单（全部完成）
| 任务序号 | 任务内容 | 完成状态 |
|----------|----------|----------|
| 1 | 使用OpenCV读取一张测试图片 
| 2 | 输出图像基本信息（尺寸、通道数、数据类型） 
| 3 | 显示原图（OpenCV方式）
| 4 | 彩色图像转换为灰度图并显示 
| 5 | 保存灰度图为新文件 
| 6 | 图像简单操作（输出像素值/裁剪左上角区域并保存）
四、核心代码说明
1. 图像读取与信息输出
```cpp
// 读取本地测试图片（绝对路径确保读取成功）
string img_path = "C:/cv-course/lab02/work.png";
Mat src = imread(img_path, IMREAD_COLOR);

// 输出图像基础信息
cout << "===== 图像基本信息 =====" << endl;
cout << "图像宽度: " << src.cols << " 像素" << endl;
cout << "图像高度: " << src.rows << " 像素" << endl;
cout << "图像通道数: " << src.channels() << endl;
2. 灰度转换与保存
// 彩色图像转灰度图
Mat gray;
cvtColor(src, gray, COLOR_BGR2GRAY);

// 保存灰度图到本地
imwrite("C:/cv-course/lab02/test_gray.png", gray);
3. 图像裁剪与保存
// 裁剪图像左上角200×200区域
int crop_size = 200;
Mat crop = src(Rect(0, 0, crop_size, crop_size));

// 保存裁剪结果
imwrite("C:/cv-course/lab02/test_crop.png", crop);
五、编译与运行步骤
1. 环境配置（Visual Studio 2022）
安装 OpenCV 4.9.0，配置系统环境变量
新建空项目，添加main.cpp源文件
项目属性配置：
VC++ 目录→包含目录：添加D:\opencv\opencv\build\include
VC++ 目录→库目录：添加D:\opencv\opencv\build\x64\vc16\lib
链接器→输入→附加依赖项：添加opencv_world490d.lib
2. 运行流程
编译项目，生成可执行文件
运行程序，自动弹出原图、灰度图、裁剪图窗口
按任意键关闭窗口，自动生成test_gray.png和test_crop.png
验证输出文件，完成实验
六、实验结果展示
1. 输入原图
work.png：实验原始输入彩色图像
2. 灰度转换结果
test_gray.png：彩色图像转灰度图的输出结果，已保存至本地
3. 裁剪结果
test_crop.png：图像左上角 200×200 区域裁剪结果，已保存至本地
4. 终端输出
程序运行后终端打印图像基础信息、操作成功提示，无报错。
七、实验总结
本次实验完整实现了 OpenCV 基础图像处理的所有要求，成功完成图像读取、信息提取、格式转换、保存与裁剪操作，验证了 OpenCV 在 Windows 环境下的可用性，掌握了 C++ 项目开发与 GitHub 代码提交的完整流程。
八、提交信息
实验人：谢厚麟
学号：2023104665
