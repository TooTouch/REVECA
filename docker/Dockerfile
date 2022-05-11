FROM nvcr.io/nvidia/pytorch:20.11-py3

RUN apt-get update 
RUN apt-get install -y openjdk-11-jdk

ARG UNAME
ARG UID
ARG GID
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME
