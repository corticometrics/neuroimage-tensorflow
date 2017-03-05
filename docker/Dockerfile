# docker build -t tensorflow-tensorboard-nibabel .
# docker run -it --rm -p 8888:8888 -p 6006:6006 -v ${PWD}:/notebooks/data tensorflow-tensorboard-nibabel

FROM gcr.io/tensorflow/tensorflow

RUN pip --no-cache-dir install nibabel

COPY ./start-jupyter-and-tensorboard.bash /
COPY ./neuro-example.ipynb /notebooks

CMD ["/start-jupyter-and-tensorboard.bash"]

