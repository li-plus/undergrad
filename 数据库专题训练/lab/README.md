# dbtrain-lab

## 总述

本仓库为 2020-2021 年春季学期数据库专题训练课程所用的实验框架，实验目标是完成一个简单数据库，并阶段性地添加新的功能，以加深同学们对数据库的理解。

## 准备工作

该实验代码由 git 管理，构建需要 cmake 和 make 工具，还需要 gcc 编译器，你可以通过系统包管理器进行安装。

以 Ubuntu 为例，使用 apt 安装依赖：

```bash
sudo apt install git make cmake g++
```

## 目录结构

目录结构如下：

1. doc: 实验文档目录

2. executable: 存放数据库可执行文件相关代码，目前有三个：

    - clear.cc, 用于清空数据库文件
    - init.cc, 用于初始化数据库文件
    - shell.cc, 一个与数据库进行交互的shell，可以运行 SQL 语句

3. src: 源代码目录

4. third_party: 第三方库，目前只包含查询解析相关的 antlr 库文件
