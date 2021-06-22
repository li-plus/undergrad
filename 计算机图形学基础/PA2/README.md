# 计算机图形学基础 PA2

> 2017011620  计73  李家昊

**你在 OpenGL 的环境配置中遇到了哪些问题？是怎么解决的？**

我在 Ubuntu 20.04 系统下配置环境，CMake 构建项目时会提示找不到 OpenGL，执行安装命令 `sudo apt install freeglut3-dev` 即可解决。

**结合核心代码分析使用 OpenGL 的绘制逻辑和光线投射的绘制逻辑有什么不同？**

手写光线投射时，程序员需要编写代码来生成光线、对光线与物体求交、对物体进行着色，最终才能得到渲染结果，较为繁琐。

使用 OpenGL 时，用户只需要将场景中的相机、灯光、三维物体、材质等参数告知 OpenGL，OpenGL 即可自动完成求交，生成渲染结果。

**你在完成作业的时候和哪些同学进行了怎样的讨论？是否借鉴了网上/别的同学的代码？**

没有与任何同学讨论，没有借鉴任何代码。

**实验结果**

| scene01                      | scene02                      | scene03                      | scene04                      |
| ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- |
| ![](code/output/scene01.bmp) | ![](code/output/scene02.bmp) | ![](code/output/scene03.bmp) | ![](code/output/scene04.bmp) |
| scene05                      | scene06                      | scene07                      |                              |
| ![](code/output/scene05.bmp) | ![](code/output/scene06.bmp) | ![](code/output/scene07.bmp) |                              |

