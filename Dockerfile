FROM intelaipg/intel-optimized-tensorflow:latest-mkl-py3
LABEL maintainer="Glen Ferguson, Michoel Snow, and Tara Blackburn"

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --allow-downgrades --no-install-recommends \
         build-essential \
         ca-certificates \
         cmake \
         curl \
         git \
         python-qt4 \
         sudo \
         unzip \
         vim \
         wget \
         zip &&\
     rm -rf /var/lib/apt/lists/*


RUN useradd cordcomp 
RUN usermod -aG sudo cordcomp
WORKDIR /opt
RUN chown cordcomp /opt
RUN chown cordcomp /
RUN chgrp cordcomp /
WORKDIR /home/cordcomp/cord_comp
RUN chown cordcomp /home
RUN chown cordcomp /home/cordcomp
RUN chown cordcomp /home/cordcomp/cord_comp
USER cordcomp

COPY --chown=cordcomp:cordcomp *.py /home/cordcomp/cord_comp/
COPY --chown=cordcomp:cordcomp cv_framework/model_definitions /home/cordcomp/cord_comp/model_definitions/
COPY --chown=cordcomp:cordcomp config.gin /home/cordcomp/cord_comp/
COPY --chown=cordcomp:cordcomp requirements.txt /home/cordcomp/cord_comp/

ENV PATH=$PATH:/home/cordcomp/.local/bin
RUN pip install opencv-python tqdm matplotlib scipy seaborn --user
RUN pip install /home/cordcomp/cord_comp/. --user
# This must be installed using git as the authors didn't update the tarball or pyPI.
RUN pip install git+https://github.com/raghakot/keras-vis.git --user
RUN pip install git+https://www.github.com/keras-team/keras-contrib.git --user
ENTRYPOINT ["bash"]

