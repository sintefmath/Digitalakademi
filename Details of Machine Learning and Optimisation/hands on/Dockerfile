FROM condaforge/mambaforge

RUN mamba install python=3.7

RUN mamba install -c conda-forge \
    numpy=1.20.1 \
    pandas=1.2.2 \
    scikit-learn=0.24.1 \
    scipy=1.6.0 \
    matplotlib=3.3.4 \
    seaborn=0.11.1 \
    cyipopt=1.0.3 \
    tensorflow=1.14.0 \
    jupyter ipython

RUN pip install tables