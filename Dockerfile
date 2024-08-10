# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/tensorflow:19.10-py3
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update

# install necessary latex packages
RUN curl -L -o install-tl-unx.tar.gz https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2024/install-tl-unx.tar.gz
RUN zcat < install-tl-unx.tar.gz | tar xf -
RUN cd install-tl-2* && perl install-tl --no-interaction --scheme=small
ENV PATH="/usr/local/texlive/2024/bin/x86_64-linux:${PATH}"
RUN tlmgr install type1cm cm-super dvipng

ENV MY_ASTRO_DATA=/astro_data
ENV SDSS_LOCAL_SAS_MIRROR=$MY_ASTRO_DATA/sdss_mirror
ENV GAIA_TOOLS_DATA=$MY_ASTRO_DATA/gaia_mirror
ENV DUST_DIR=$MY_ASTRO_DATA/mwdust_mirror
RUN mkdir $MY_ASTRO_DATA $SDSS_LOCAL_SAS_MIRROR $GAIA_TOOLS_DATA $DUST_DIR
RUN git clone https://github.com/henrysky/astroNN_spectra_paper_figures.git
WORKDIR /workspace/astroNN_spectra_paper_figures
RUN pip install -r requirements.txt
RUN mkdir data_files figs
RUN curl --cookie zenodo-cookies.txt "https://zenodo.org/records/13290056/files/__train.h5?download=1" --output __train.h5.h5
RUN curl --cookie zenodo-cookies.txt "https://zenodo.org/records/13290056/files/_highsnr_test.h5?download=1" --output _highsnr_test.h5
RUN curl --cookie zenodo-cookies.txt "https://zenodo.org/records/13290056/files/aspcap_training.h5?download=1" --output aspcap_training.h5
