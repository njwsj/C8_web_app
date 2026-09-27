# Docker 部署指南

本指南说明如何将 C8 选择性预测服务构建为 Docker 镜像并推送到 Docker Hub，以便在其他电脑上快速部署。

## 前置准备

### 1. 安装 Docker

**macOS:**
```bash
brew install --cask docker
# 或从官网下载 Docker Desktop: https://www.docker.com/products/docker-desktop
```

**Linux (Ubuntu/Debian):**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# 重新登录使权限生效
```

**Windows:**
下载并安装 Docker Desktop: https://www.docker.com/products/docker-desktop

### 2. 注册 Docker Hub 账号

访问 https://hub.docker.com/ 注册账号（如果还没有）。

## 构建并推送镜像

### 步骤 1: 登录 Docker Hub

```bash
docker login
# 输入你的 Docker Hub 用户名和密码
```

### 步骤 2: 构建 Docker 镜像

在项目根目录（C8_web_app）下执行：

```bash
cd /Users/zhanghj/Documents/school/code/C8_web_app

# 构建镜像（替换 your-dockerhub-username 为你的用户名）
docker build -t your-dockerhub-username/c8-prediction:latest .

# 示例：如果用户名是 zhangsan
# docker build -t zhangsan/c8-prediction:latest .
```

构建过程大约需要 5-10 分钟，取决于网络速度。

### 步骤 3: 测试镜像（可选但推荐）

```bash
# 运行容器测试
docker run -d -p 5001:5001 --name c8_test your-dockerhub-username/c8-prediction:latest

# 访问 http://localhost:5001 测试是否正常

# 查看日志
docker logs c8_test

# 停止并删除测试容器
docker stop c8_test
docker rm c8_test
```

### 步骤 4: 推送到 Docker Hub

```bash
docker push your-dockerhub-username/c8-prediction:latest
```

推送完成后，你可以在 https://hub.docker.com/r/your-dockerhub-username/c8-prediction 查看镜像。

## 在新电脑上部署

### 方式一：直接运行（独立容器）

```bash
# 拉取镜像
docker pull your-dockerhub-username/c8-prediction:latest

# 运行服务
docker run -d \
  -p 5001:5001 \
  --name c8_prediction \
  --restart unless-stopped \
  your-dockerhub-username/c8-prediction:latest

# 访问 http://localhost:5001
```

### 方式二：使用 docker-compose（推荐）

**1. 创建工作目录：**
```bash
mkdir -p ~/c8_deployment
cd ~/c8_deployment
```

**2. 创建 docker-compose.yml 文件：**
```yaml
version: '3.8'

services:
  c8_web_app:
    image: your-dockerhub-username/c8-prediction:latest
    container_name: c8_prediction_service
    ports:
      - "5001:5001"
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

**3. 启动服务：**
```bash
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

## 注意事项

### 1. MySQL 数据库配置

**当前 Dockerfile 不包含 MySQL**，有两种解决方案：

**方案 A：连接宿主机 MySQL（推荐用于生产环境）**

在新电脑上：
1. 安装并启动 MySQL
2. 运行时修改容器网络配置：

```bash
docker run -d \
  -p 5001:5001 \
  --add-host=host.docker.internal:host-gateway \
  --name c8_prediction \
  your-dockerhub-username/c8-prediction:latest
```

然后修改 `app.py` 中的 `DB_CONFIG`，将 `host` 改为 `host.docker.internal`。

**方案 B：在 Docker 中同时运行 MySQL**

下载完整的 docker-compose.yml（已包含在项目中），取消注释 MySQL 部分：

```bash
docker-compose up -d
```

这会同时启动应用和 MySQL 容器。

### 2. C8 原始数据文件

**重要：** 当前 Dockerfile 已将 `model/C8选择性.xlsx` 打包到镜像中。如果需要更新此文件：

```bash
# 重新构建镜像
docker build -t your-dockerhub-username/c8-prediction:latest .
docker push your-dockerhub-username/c8-prediction:latest
```

或者使用 volume 挂载（动态更新）：

```bash
docker run -d \
  -p 5001:5001 \
  -v /path/to/your/C8选择性.xlsx:/app/model/C8选择性.xlsx \
  --name c8_prediction \
  your-dockerhub-username/c8-prediction:latest
```

### 3. 端口冲突

如果 5001 端口被占用，可以映射到其他端口：

```bash
# 映射到 8080 端口
docker run -d -p 8080:5001 --name c8_prediction your-dockerhub-username/c8-prediction:latest

# 访问 http://localhost:8080
```

## 常用命令

```bash
# 查看运行中的容器
docker ps

# 查看所有容器（包括停止的）
docker ps -a

# 查看日志
docker logs c8_prediction
docker logs -f c8_prediction  # 实时日志

# 进入容器调试
docker exec -it c8_prediction bash

# 停止容器
docker stop c8_prediction

# 启动已停止的容器
docker start c8_prediction

# 删除容器
docker rm c8_prediction

# 删除镜像
docker rmi your-dockerhub-username/c8-prediction:latest

# 查看镜像大小
docker images | grep c8-prediction
```

## 镜像优化建议

当前镜像大小约 2-3 GB（主要是 PyTorch）。如果需要优化：

1. **使用多阶段构建**（减小镜像体积）
2. **仅安装 PyTorch CPU 版本**：
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

## 故障排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 容器启动后立即退出 | 检查日志查看错误 | `docker logs c8_prediction` |
| 无法访问服务 | 端口映射错误或防火墙 | 检查 `-p` 参数，关闭防火墙测试 |
| MySQL 连接失败 | 容器无法访问宿主机 MySQL | 使用 `--add-host=host.docker.internal:host-gateway` |
| 模型文件缺失 | 构建时未包含 model 目录 | 确认 model/ 目录在构建路径下 |

## 示例：完整部署流程

```bash
# === 在原电脑上 ===
cd /Users/zhanghj/Documents/school/code/C8_web_app
docker login
docker build -t zhangsan/c8-prediction:latest .
docker push zhangsan/c8-prediction:latest

# === 在新电脑上 ===
# 1. 安装 Docker
# 2. 安装并启动 MySQL（可选）

# 3. 拉取并运行
docker pull zhangsan/c8-prediction:latest
docker run -d \
  -p 5001:5001 \
  --name c8_prediction \
  --restart unless-stopped \
  zhangsan/c8-prediction:latest

# 4. 访问服务
open http://localhost:5001
```

## 更新镜像

当代码更新后：

```bash
# 1. 重新构建
docker build -t your-dockerhub-username/c8-prediction:latest .

# 2. 推送新版本
docker push your-dockerhub-username/c8-prediction:latest

# 3. 在部署机器上更新
docker pull your-dockerhub-username/c8-prediction:latest
docker stop c8_prediction
docker rm c8_prediction
docker run -d -p 5001:5001 --name c8_prediction your-dockerhub-username/c8-prediction:latest
```
