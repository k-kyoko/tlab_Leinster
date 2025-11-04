FROM jupyter/datascience-notebook:lab-4.0.7

# 作業ディレクトリを設定
WORKDIR /home/jovyan

# 必要なパッケージ・拡張機能のインストール  7系のダウングレード：notebookで互換性を保つため
RUN pip install --no-cache-dir \
    notebook==6.4.12 \
    jupyter_contrib_nbextensions \
    jupyter_nbextensions_configurator \
    jupyterlab-lsp \
    python-lsp-server[all] \
    jupyterlab-pygments==0.3.0 \
    jupyterlab-git \
    nbdime

RUN pip uninstall -y jupyterlab-git
RUN pip uninstall -y nbdime nbdime-jupyterlab
RUN pip install jupyterlab-git


# JupyterLab 拡張機能の有効化
RUN jupyter contrib nbextension install --user && \
    jupyter nbextensions_configurator enable --user && \
    jupyter labextension install @jupyter-lsp/jupyterlab-lsp && \
    jupyter labextension install jupyterlab_code_formatter && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager && \
    jupyter labextension unlock @jupyterlab/completer-extension:base-service  && \
    jupyter labextension unlock @jupyterlab/fileeditor-extension:language-server  && \
    jupyter labextension unlock @jupyterlab/lsp-extension:settings  && \
    jupyter labextension unlock @jupyterlab/notebook-extension:language-server  && \
    jupyter labextension enable @jupyterlab/completer-extension:base-service && \
    jupyter labextension enable @jupyterlab/fileeditor-extension:language-server && \
    jupyter labextension enable @jupyterlab/lsp-extension:settings && \
    jupyter labextension enable @jupyterlab/notebook-extension:language-server

# キャッシュクリアとビルド
RUN jupyter lab clean && \
    jupyter lab build

# GWTuneの環境ファイルをコンテナにコピー
COPY work/GWTune/environment.yaml /home/jovyan/environment.yaml

# Conda環境を更新し、依存パッケージをインストール
RUN conda env update -f /home/jovyan/environment.yaml && \
    conda clean -afy

# GWTuneのsrcフォルダをPYTHONPATHに追加
ENV PYTHONPATH="${PYTHONPATH:-}:/home/jovyan/work/GWTune/src"


# gwtune環境をJupyter Notebookのカーネルに追加
RUN /opt/conda/envs/gwtune/bin/python -m ipykernel install --user --name=gwtune --display-name "Python (gwtune)"

# jupyterlab-lsp と jupyterlab_code_formatterのインストール
RUN pip install jupyterlab-lsp python-language-server jupyterlab_code_formatter python-lsp-server

# Jupyterの設定でgwtuneをデフォルトカーネルに指定
RUN mkdir -p /home/jovyan/.jupyter/lab/user-settings/@jupyterlab/notebook-extension/ && \
    echo '{ "kernelPreference": { "name": "gwtune", "language": "python" } }' \
    > /home/jovyan/.jupyter/lab/user-settings/@jupyterlab/notebook-extension/tracker.jupyterlab-settings

