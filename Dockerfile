FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

# Install git as root
USER root
RUN apt-get update && apt-get install -y git tmux && apt-get clean && rm -rf /var/lib/apt/lists/*

# Switch back to jovyan user (default Jupyter user)
# USER jovyan

# Copy requirements file as root
COPY . /tmp/.

# Set working directory
WORKDIR /home/jovyan/work
# Ensure the src directory is in the Python path for Jupyter
ENV PYTHONPATH="/home/jovyan/work/src:${PYTHONPATH}"

# Install Jupyter Notebook
RUN pip install --no-cache-dir jupyter

# Install Python packages (project and extra requirements)
RUN pip install -U pip
RUN pip install --no-cache-dir /tmp/.
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# To run jupyter in remote development scenario with VSCode
# from https://stackoverflow.com/questions/63998873/vscode-how-to-run-a-jupyter-notebook-in-a-docker-container-over-a-remote-serve
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# Configure Jupyter
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab automatically
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]