# 中文文档 | [English](../README.md) | [한국어](./README_KR.md)
灵活可扩展的 AI 驱动自动驾驶系统软件。

![license](https://img.shields.io/badge/license-MIT-green.svg)
![language](https://img.shields.io/badge/language-English-blue.svg)
![language](https://img.shields.io/badge/language-中文-red.svg)
![language](https://img.shields.io/badge/language-한국어-orange.svg)

[English Documentation](./docs/README_EN.md) | [한국어 문서](./docs/README_KR.md)

<img src="https://github.com/leolixingyou/FlexpAI/blob/main/docs/demo.gif" width="640" alt="Demo Video">


## 目录
- [系统概述](#系统概述)
- [系统架构设计](#系统架构设计)
- [功能模块](#功能模块)
  - [控制系统](#控制系统)
  - [规划系统](#规划系统)
  - [感知系统](#感知系统)
  - [传感系统](#传感系统)
  - [输入输出接口](#输入输出接口)
- [快速开始](#快速开始)
- [系统扩展](#系统扩展)
- [性能测试](#性能测试)
- [贡献指南](#贡献指南)
- [致谢](#致谢)
- [许可证](#许可证)

## 系统概述
这是一个面向多平台的通用自动驾驶系统框架，支持灵活配置与功能扩展。本系统旨在提供一个标准化的自动驾驶开发平台，可适配不同的硬件设施与应用场景。

## 系统架构设计
pass

## 功能模块

### 控制系统
#### 特点
- 多控制器支持
- 实时性保证
- 控制精度优化
#### 实现方式
pass

### 规划系统
#### 特点
- 多场景适配
- 动态路径规划
- 实时决策优化
#### 实现方式
pass

### 感知系统
#### 特点
- 多传感器融合
- 目标检测与跟踪
- 场景理解能力
#### 实现方式
pass

### 传感系统
#### 特点
- 传感器模块化设计
- 数据同步机制
- 故障检测与恢复
#### 实现方式
pass

### 输入输出接口
#### 特点
- 一次配置永久使用
- 标准化接口定义
- 跨平台兼容
#### 实现方式
pass

## 快速开始
### Docker Environment Setup

Run the following command to enter the docker environment:
```bash
./docker/run_container.sh
```
This will pull and run our pre-configured docker image from Docker Hub.

### Development Setup

#### Windows Users
1. Open WSL terminal
2. Run `code .` to open VS Code
3. Connect to docker container within VS Code

## Docker Image

Our official docker image is available on Docker Hub:
```bash
docker push leolixingyou/ros1_for_self-driving:carla_0.9.13
```


## 系统扩展
pass

## 性能测试
pass

## 贡献指南
如果您想为本项目做出贡献，请参考以下步骤：
1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 将您的更改推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交拉取请求

## 致谢
- 感谢所有为本项目做出贡献的开发者
- 特别感谢以下机构/个人的支持：
  - [待添加]
  - [待添加]

## 许可证
本项目采用 MIT 许可证。查看 [LICENSE](LICENSE) 获取更多信息。

---