FROM mambaorg/micromamba:2.0.4

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

# Set the working directory
WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER test_files /app/test_files/
COPY --chown=$MAMBA_USER:$MAMBA_USER *.py mimetypes.json /app/
