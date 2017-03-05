# docker build --no-cache -t tensorflow-tensorboard-nibabel .
# docker run -it --rm -p 8888:8888 -p 6006:6006 -v ~/tensorflow-test:/notebooks/data tensorflow-tensorboard-nibabel

FROM gcr.io/tensorflow/tensorflow

RUN pip --no-cache-dir install nibabel

COPY ./start-jupyter-and-tensorboard.bash /

CMD ["/start-jupyter-and-tensorboard.bash"]

