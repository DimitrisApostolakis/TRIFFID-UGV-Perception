# syntax=docker/dockerfile:1
# TRIFFID UGV perception - live ROS 2 node: detector + tracker + geolocation -> GeoJSON / map API.
#
# Stages
#   base               ROS 2 + PyTorch + Python dependencies (shared)
#   runtime (default)  YOLOE-v8-L (visual supports) + SIFT appearance hints + geo-space association
#   masa               runtime + MASA-R50 fallback tracker (builds mmcv CUDA ops; slow first build)
#
# Weights and the support bank are mounted at run time, never baked in.
# CUDA 12.8 PyTorch wheels cover Turing..Blackwell (RTX 20xx-50xx); needs NVIDIA driver >= 570.
ARG ROS_DISTRO=jazzy

FROM ros:${ROS_DISTRO}-ros-base AS base
ARG ROS_DISTRO
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    YOLO_OFFLINE=1 YOLO_AUTOINSTALL=false YOLO_CONFIG_DIR=/tmp/ultralytics MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    ROS_DISTRO=${ROS_DISTRO}

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-venv python3-pip git libgl1 libglib2.0-0 \
        ros-${ROS_DISTRO}-tf2-ros-py \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
        ros-${ROS_DISTRO}-rosbag2-storage-mcap \
    && rm -rf /var/lib/apt/lists/*

# Python environment on top of ROS's Python (rclpy and messages stay visible).
# numpy stays < 2: ROS Jazzy message extensions are built against NumPy 1.x.
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:${PATH}
RUN python3 -m venv --system-site-packages /opt/venv && pip install -U "pip<25" wheel
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN --mount=type=cache,target=/root/.cache/pip pip install torch==2.7.0 torchvision==0.22.0 --index-url ${TORCH_INDEX}

COPY requirements.txt constraints.txt /tmp/req/
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/req/requirements.txt -c /tmp/req/constraints.txt


# ---------------------------------------------------------------------------------------------------
FROM base AS runtime
# YOLOE: THU-MIG fork of Ultralytics, pinned to the evaluated revision. It installs as `ultralytics`.
ARG YOLOE_REPO=https://github.com/THU-MIG/yoloe.git
ARG YOLOE_REF=40cd606cabdbe2b566d6f14a6b162c89206e9a1b
RUN --mount=type=cache,target=/root/.cache/pip pip install -c /tmp/req/constraints.txt "ultralytics @ git+${YOLOE_REPO}@${YOLOE_REF}" \
 && pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null; \
    pip install --force-reinstall --no-deps -c /tmp/req/constraints.txt opencv-python-headless

# Robot message packages (ugv_status_interfaces, gnss_converter_interfaces), built with the system toolchain.
COPY interfaces /opt/triffid_ws/src
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
 && cd /opt/triffid_ws && colcon build --merge-install --event-handlers console_direct- \
 && rm -rf build log

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY cyclonedds.xml ./
RUN pip install --no-cache-dir --no-deps . \
 && python -c "import numpy, cv2, torch; assert numpy.__version__.startswith('1.'); cv2.SIFT_create()"
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && mkdir -p /app/outputs /app/state /app/supports /models \
 && chmod 1777 /app/outputs /app/state /tmp

ENTRYPOINT ["/entrypoint.sh"]
CMD ["run", "--config", "configs/default.yaml"]


# ---------------------------------------------------------------------------------------------------
# MASA: mmcv must be compiled against this exact PyTorch/CUDA; done in a throw-away builder stage.
FROM base AS masa-build
ARG CUDA_APT=12-8
RUN apt-get update && apt-get install -y --no-install-recommends wget build-essential ninja-build \
 && wget -qO /tmp/keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i /tmp/keyring.deb && apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-${CUDA_APT} cuda-cudart-dev-${CUDA_APT} cuda-nvrtc-dev-${CUDA_APT} cuda-nvtx-${CUDA_APT} \
        cuda-profiler-api-${CUDA_APT} libcublas-dev-${CUDA_APT} libcusparse-dev-${CUDA_APT} \
        libcusolver-dev-${CUDA_APT} libcurand-dev-${CUDA_APT} \
 && rm -rf /var/lib/apt/lists/*
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0"
ARG MAX_JOBS=8
ENV CUDA_HOME=/usr/local/cuda MMCV_WITH_OPS=1 FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS}
RUN pip wheel --no-cache-dir --no-build-isolation --no-deps mmcv==2.2.0 -w /wheels


FROM runtime AS masa
COPY --from=masa-build /wheels /wheels
COPY requirements-masa.txt /tmp/req/
# --no-deps: the OpenMMLab packages would otherwise pull opencv-python over the headless build.
RUN pip install --no-cache-dir --no-deps /wheels/mmcv-*.whl mmengine==0.10.7 mmdet==3.3.0 \
 && pip install --no-cache-dir -r /tmp/req/requirements-masa.txt -c /tmp/req/constraints.txt \
 && sed -i "s/^mmcv_maximum_version = .*/mmcv_maximum_version = '2.3.0'/" \
        /opt/venv/lib/python3*/site-packages/mmdet/__init__.py \
 && rm -rf /wheels
ARG MASA_REPO=https://github.com/siyuanliii/masa.git
ARG MASA_REF=c5472b9c7615f35abdf1188cb1a0c5408fe50d66
COPY docker/patch_masa.py /tmp/patch_masa.py
RUN git clone --filter=blob:none "${MASA_REPO}" /opt/masa && git -C /opt/masa checkout -q "${MASA_REF}" \
 && python /tmp/patch_masa.py /opt/masa \
 && python -c "import mmcv.ops, mmdet; print('mmcv ops OK')"
