FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu14.04

RUN sudo apt-get update && sudo apt-get -y upgrade
RUN sudo apt-get update && sudo apt-get install -y software-properties-common
RUN sudo add-apt-repository ppa:jonathonf/python-3.6
RUN sudo add-apt-repository ppa:libccd-debs/ppa
RUN sudo add-apt-repository ppa:fcl-debs/ppa
RUN sudo apt-get update &&  sudo apt-get install -y \
      python3.6 git curl \
      build-essential cmake \
      libeigen3-dev \
      libccd-dev \
      libfcl-dev \
      libassimp-dev \
      libboost-regex-dev \
      libopenscenegraph-dev \
      libqt4-dev libqt4-opengl-dev \
      coinor-libipopt-dev libxi-dev libxmu-dev freeglut3-dev libflann-dev libnlopt-dev \
      gawk \
      python3.4-dev \
      libtinyxml2-dev \
      liburdfdom-dev \
      doxygen \
      python3.6-dev unzip \
      libxrandr-dev libosmesa-dev libxinerama-dev libxcursor-dev \
      libtcmalloc-minimal4 \
      swig

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/mesa/libGL.so:/usr/lib/libtcmalloc_minimal.so.4
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin


# Compiling libraries

RUN mkdir /root/lib
WORKDIR /root/lib
RUN curl -L https://github.com/Kitware/CMake/releases/download/v3.14.0-rc3/cmake-3.14.0-rc3.tar.gz > cmake-3.14.0-rc3.tar.gz
RUN tar xzf cmake-3.14.0-rc3.tar.gz
RUN cd cmake-3.14.0-rc3; ./bootstrap; sudo make install
RUN git clone git://github.com/OctoMap/octomap.git
RUN cd octomap; git checkout tags/v1.8.1 -b v1.8.1; mkdir build; cd build; cmake ..; make -j5; sudo make install
RUN curl -L https://bitbucket.org/odedevs/ode/downloads/ode-0.16.tar.gz > ode-0.16.tar.gz
RUN tar xzf ode-0.16.tar.gz
RUN cd ode-0.16; cd build; cmake ..; make -j5; sudo make install
RUN git clone https://github.com/bulletphysics/bullet3.git
RUN cd bullet3; git checkout tags/2.88 -b 2.88; ./build_cmake_pybullet_double.sh; cd build_cmake; sudo make install
RUN curl -L https://www.roboti.us/download/mjpro150_linux.zip > /root/lib/mjpro_150_linux.zip
RUN mkdir /root/.mujoco; unzip /root/lib/mjpro_150_linux.zip -d /root/.mujoco
RUN touch /root/.mujoco/mjkey.txt
RUN curl -L http://nixos.org/releases/patchelf/patchelf-0.9/patchelf-0.9.tar.bz2 > patchelf-0.9.tar.bz2
RUN tar xf patchelf-0.9.tar.bz2
RUN cd patchelf-0.9; ./configure; make -j5; sudo make install
RUN git clone https://github.com/glfw/glfw; cd glfw; mkdir build; cd build; cmake ..; make -j5; sudo make install
RUN git clone https://github.com/dartsim/dart
RUN cd dart; git checkout tags/v6.6.0 -b v6.6.0; mkdir build; cd build; cmake ..; make -j5; sudo make install
RUN sudo ldconfig /usr/local/lib
RUN rm -rf /root/lib

# Installing pip and pipenv
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6
RUN sudo pip3 install pipenv

# Cloning domain randomization
RUN mkdir /root/work
WORKDIR /root/work
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

RUN git clone https://github.com/sharadmv/domain-randomization.git
WORKDIR /root/work/domain-randomization
RUN git pull origin master
RUN pipenv install tensorflow
RUN mkdir /root/.aws; echo '[default]\nregion = us-west-2' > ~/.aws/config

WORKDIR /root/work/
RUN git clone https://github.com/quanvuong/baselines.git

WORKDIR /root/work/domain-randomization 
RUN pipenv install -e ../baselines
