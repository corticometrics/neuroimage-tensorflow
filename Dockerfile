FROM jupyter/tensorflow-notebook

RUN conda install --quiet --yes 'nibabel=2.1.0'
