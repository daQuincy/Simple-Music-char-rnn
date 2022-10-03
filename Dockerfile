FROM nvcr.io/nvidia/tensorflow:22.08-tf1-py3

RUN apt-get update && apt-get -y install sudo
RUN adduser --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime
RUN echo Etc/UTC > /etc/timezone

USER ubuntu
WORKDIR /home/ubuntu

RUN pip install pretty_midi beautifulsoup4
