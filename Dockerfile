# 使用 Python 3.10 作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 所有 Python 依赖均有 manylinux 预编译 wheel，无需 gcc/g++ 编译
# （如遇个别包需要编译，再恢复安装 build-essential）

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目文件
COPY app.py .
COPY train_save.py .
COPY model/ model/
COPY templates/ templates/

# 暴露端口
EXPOSE 5001

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 启动命令
CMD ["python", "app.py"]
