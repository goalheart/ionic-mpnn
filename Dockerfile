# ARG 必须在 FROM 之前
ARG http_proxy
ARG https_proxy
ARG no_proxy

FROM continuumio/miniconda3:24.5.0-0

# 使用 ${} 代替 $，避免警告
ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy}

# 为 APT 配置代理（如需）
RUN if [ -n "${http_proxy}" ]; then \
        echo "Acquire::http::Proxy \"${http_proxy}\";" > /etc/apt/apt.conf.d/99proxy; \
        echo "Acquire::https::Proxy \"${http_proxy}\";" >> /etc/apt/apt.conf.d/99proxy; \
    fi

# 安装 curl（用于调试）
RUN apt-get update && apt-get install -y curl ca-certificates && rm -rf /var/lib/apt/lists/*

# 可选：测试代理是否通（构建时取消注释）
# RUN curl -x ${http_proxy} -I https://repo.anaconda.com || echo "Proxy test failed"

WORKDIR /workspace
COPY environment.yml .
RUN conda env create -f environment.yml
RUN conda run -n ionic-mpnn pip install git+https://github.com/NREL/nfp.git
ENV PATH /opt/conda/envs/ionic-mpnn/bin:$PATH
CMD ["bash"]