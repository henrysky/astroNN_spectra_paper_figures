
Abstract
===========

Deep learning with artificial neural networks is increasingly gaining attention, because of its potential for data-driven
astronomy. However, this methodology usually does not provide uncertainties and does not deal with incompleteness and
noise in the training data. In this work, we design a neural network for high-resolution spectroscopic analysis using
APOGEE data that mimics the methodology of standard spectroscopic analyses: stellar parameters are determined using the
full wavelength range, but individual element abundances use censored portions of the spectrum. We train this network
with a customized objective function that deals with incomplete and noisy training data and apply dropout variational
inference to derive uncertainties on our predictions. We determine parameters and abundances for 18 individual elements
at the ≈0.03dex level, even at low signal-to-noise ratio. We demonstrate that the uncertainties returned by our method
are a realistic estimate of the precision and they automatically blow up when inputs or outputs outside of the training
set are encountered, thus shielding users from unwanted extrapolation. By using standard deep-learning tools for GPU
acceleration, our method is extremely fast, allowing analysis of the entire APOGEE data set in ten minutes on a single,
low-cost GPU. We release the stellar parameters and 18 individual-element abundances with associated uncertainty for the
entire APOGEE DR14 dataset. Simultaneously, we release ``astroNN``, a well-tested, open-source python package
developed for this work, but that is also designed to be a general package for deep learning in astronomy. ``astroNN`` is
available at https://github.com/henrysky/astroNN with extensive documentation at http://astroNN.readthedocs.io.

.. contents:: **Table of Contents**
    :depth: 3

Getting Started
=================

This repository is to make sure all figures and results are reproducible by anyone easily for this paper [`arXiv:1804.08622`_][`ADS`_].

.. _arXiv:1804.08622: https://arxiv.org/abs/1808.04428
.. _ADS: https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.3255L/abstract

If Github has issue (or too slow) to load the Jupyter Notebooks, you can go
http://nbviewer.jupyter.org/github/henrysky/astroNN_spectra_paper_figures/tree/master/

To get started, this paper uses `astroNN`_ developed by us and tested with **astroNN 1.0.0** which Python 3.6 or above
and reasonable computational resource is required. Extensive documentation of v1.0.0 at
https://astronn.readthedocs.io/en/v1.0.0/ and quick start guide at https://astronn.readthedocs.io/en/v1.0.0/quick_start.html.
astroNN Apogee DR14 Stellar Parameters and Abundances data is available as `astroNN_apogee_dr14_catalog.fits`_.

Some notebooks make use of my `milkyway_plot`_ to plot on milkyway.

.. _astroNN: https://github.com/henrysky/astroNN
.. _milkyway_plot: https://github.com/henrysky/milkyway_plot

To continuum normalize arbitrary APOGEE spectrum with v1.0.0, see: https://astronn.readthedocs.io/en/v1.0.0/tools_apogee.html#continuum-normalization-of-apogee-spectra

Docker Image
----------------

If you have `Docker`_ installed, you can use the `Dockerfile`_ to build a Docker image upon Pytorch container from `NVIDIA NGC Catalog`_ with all dependencies installed and data files downloaded.

.. _NVIDIA NGC Catalog: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
.. _Dockerfile: Dockerfile
.. _Docker: https://www.docker.com/

To build the Docker image called ``astronn_spectra_paper_figures``, run the following command in the root directory of this repository:

.. code-block:: bash

    docker build -t astronn_spectra_paper_figures .

To run the Docker container with all GPU available to the container named ``testing123``, run the following command:

.. code-block:: bash
    
    docker run --gpus all --name testing123 -it -e SHELL=/bin/bash --entrypoint bash astronn_spectra_paper_figures

Then you can attach to the container by running:

.. code-block:: bash

    docker exec -it testing123 bash

Now you can run all notebooks or training script inside the container

Jupyter Notebook
------------------
-   | `Datasets_Data_Reduction.ipynb`_
    | You should check out this notebook first as it describes how to reproduce the **exactly** same datasets used in the paper
-   | `Training.ipynb`_
    | It provided the code used to train ``astroNN_0606_run001``  and ``astroNN_0617_run001``
-   | `Inference_highSNR.ipynb`_
    | It describes the inference process and result on spectra within SNR 100-200, also includes an isochrone plot and
    | jacobian analysis with ``astroNN_0617_run001``.
-   | `Inference.ipynb`_
    | It describes the inference process and result on individual visit spectra which their combined counterpart in
    | training set with ``astroNN_0617_run001``.
-   | `Inference_ApogeeBCNN.ipynb`_
    | It describes the inference process and result on spectra within SNR 100-200 with ``astroNN_0606_run001``.
-   | `Open_Globular_Cluster_Benchmark.ipynb`_
    | It describes the inference on Open/Globular Cluster.
-   | `Apogee_dr14_NN_Catalog.ipynb`_
    | It describes how to generate stellar parameters and abundances for the whole APOGEE DR14, also contains plots of abundances across MilkyWay Galaxy.
-   | `ASPCAP_Normalization.ipynb`_
    | It describes how to compile dataset with ASPCAP normalized spectra (as opposed to continuum normalization), training and testing NN on that.
-   | `Small_Data.ipynb`_
    | It describes training neural network with smaller datasets and see the performance.
-   | `nn_figure6_draw_io`_
    | Source for Figure 6 in paper for the NN model, can be opened and edited by draw.io

.. _Datasets_Data_Reduction.ipynb: Datasets_Data_Reduction.ipynb
.. _Training.ipynb: Training.ipynb
.. _Inference_highSNR.ipynb: Inference_highSNR.ipynb
.. _Inference.ipynb: Inference.ipynb
.. _Inference_ApogeeBCNN.ipynb: Inference_ApogeeBCNN.ipynb
.. _Open_Globular_Cluster_Benchmark.ipynb: Open_Globular_Cluster_Benchmark.ipynb
.. _Apogee_dr14_NN_Catalog.ipynb: Apogee_dr14_NN_Catalog.ipynb
.. _ASPCAP_Normalization.ipynb: ASPCAP_Normalization.ipynb
.. _Small_Data.ipynb: Small_Data.ipynb
.. _nn_figure6_draw_io: https://github.com/henrysky/astroNN_spectra_paper_figures/raw/master/_nn_figure6_draw_io

Neural Net Models
------------------
- ``astroNN_0606_run001`` is a trained astroNN's `ApogeeBCNN()`_ class model to infer 22 stellar parameters from APOGEE continuum normalized spectra.

- ``astroNN_0617_run001`` is a trained astroNN's `ApogeeBCNNCensored()`_ class model to infer 22 stellar parameters from APOGEE continuum normalized spectra.

- ``aspcapStar_BCNNCensored`` is a trained astroNN's `ApogeeBCNNCensored()`_ class model to infer 22 stellar parameters from APOGEE ASPCAP-normalized spectra, with exactly the same model architecture as ``astroNN_0617_run001``.

- ``small_data_fixed_****`` are trained astroNN's `ApogeeBCNNCensored()`_ class models with small dataset, with exactly the same model architecture as ``astroNN_0617_run001``.

.. _ApogeeBCNN(): http://astronn.readthedocs.io/en/latest/neuralnets/apogee_bcnn.html

.. _ApogeeBCNNCensored(): http://astronn.readthedocs.io/en/latest/neuralnets/apogee_bcnncensored.html

To load the model, open python outside ``astroNN_0606_run001`` or ``astroNN_0617_run001``

.. code-block:: python

    from astroNN.models import load_folder

    # replace the name of the NN folder you want to open
    neuralnet = load_folder('astroNN_0617_run001')
    # neuralnet is an astroNN neural network object, to learn more;
    # http://astronn.readthedocs.io/en/latest/neuralnets/basic_usage.html

    # To get what the output neurones are representing
    print(neuralnet.targetname)

astroNN Apogee DR14 Stellar Parameters and Abundances
------------------------------------------------------

`astroNN_apogee_dr14_catalog.fits`_ is compiled prediction with ``astroNN_0617_run001`` on the whole Apogee DR14. To load it with python

.. code-block:: python

    from astropy.io import fits

    f = fits.getdata("astroNN_apogee_dr14_catalog.fits")
    apogee_id = f['APOGEE_ID']  # APOGEE's apogee id
    location_id = f['LOCATION_ID']  # APOGEE DR14 location id
    ra = f['RA']  # J2000 RA
    dec = f['DEC']  # J2000 DEC

    # the order of the array is [Teff, log(g), C/H, C1/H, N/H, O/H, Na/H, Mg/H, Al/H, Si/H, P/H, S/H, K/H, Ca/H, Ti/H,
    # Ti2/H, V/H, Cr/H, Mn/H, Fe/H, Co/H, Ni/H]
    nn_prediction = f['astroNN']  # neural network prediction, contains -9999.
    nn_uncertainty = f['astroNN_error']  # neural network uncertainty, contains -9999.

.. _`astroNN_apogee_dr14_catalog.fits`: https://github.com/henrysky/astroNN_spectra_paper_figures/raw/master/astroNN_apogee_dr14_catalog.fits

Using Neural Net on arbitrary APOGEE spectra
-----------------------------------------------

To do inference on an arbitrary APOGEE spectrum,

1. Open python under the repository folder but outside the folder ``astroNN_0617_run001``
2. Copy and paste the following code to do inference with neural net in this paper on ``2M19060637+4717296``

.. code-block:: python

    from astropy.io import fits
    from astroNN.apogee import visit_spectra, apogee_continuum
    from astroNN.models import load_folder

    # the same spectrum used in figure 5
    opened_fits = fits.open(visit_spectra(dr=14, apogee='2M19060637+4717296'))
    spectrum = opened_fits[1].data
    spectrum_err = opened_fits[2].data
    spectrum_bitmask = opened_fits[3].data

    # using default continuum and bitmask values to continuum normalize
    norm_spec, norm_spec_err = apogee_continuum(spectrum, spectrum_err,
                                                bitmask=spectrum_bitmask, dr=14)

    # load neural net
    neuralnet = load_folder('astroNN_0617_run001')

    # inference, if there are multiple visits, then you should use the globally
    # weighted combined spectra (i.e. the second row)
    pred, pred_err = neuralnet.test(norm_spec)

    print(neuralnet.targetname)  # output neurons representation
    print(pred)  # prediction
    print(pred_err['total'])  # prediction uncertainty

External Data (ThePayne)
---------------------------

`ThePayne_dr14_catalog.fits`_ is compiled from the data provided in the paper https://arxiv.org/abs/1804.01530

To load it with python

.. code-block:: python

    from astropy.io import fits

    # the order is correspond to APOGEE DR14 allstar
    f = fits.getdata("ThePayne_dr14_catalog.fits")
    apogee_id = f['APOGEE_ID']  # APOGEE's apogee id
    location_id = f['LOCATION_ID']  # APOGEE DR14 location id
    ra = f['RA']  # J2000 RA
    dec = f['DEC']  # J2000 DEC

    # the order of the array is [Teff, log(g), C/H, C1/H, N/H, O/H, Na/H, Mg/H, Al/H, Si/H, P/H, S/H, K/H, Ca/H, Ti/H,
    # Ti2/H, V/H, Cr/H, Mn/H, Fe/H, Co/H, Ni/H], same as astroNN DR14 order
    payne_prediction = f['payne']  # ThePayne data, contains -9999.

    # good flag is 1, bad flag is 0
    payne_good_flag = f['good_flag']  # ThePayne quality flag

.. _`ThePayne_dr14_catalog.fits`: https://github.com/henrysky/astroNN_spectra_paper_figures/raw/master/external_data/ThePayne_dr14_catalog.fits

Authors
=================
-  | **Henry Leung** - henrysky_
   | Department of Astronomy and Astrophysics, University of Toronto
   | Contact Henry: henrysky.leung [at] utoronto.ca

-  | **Jo Bovy** - jobovy_
   | Department of Astronomy and Astrophysics, University of Toronto

.. _henrysky: https://github.com/henrysky
.. _jobovy: https://github.com/jobovy

Information on ``aj485195t4_mrt.txt`` for Open/Globular Cluster Benchmark
=============================================================================

The original header of the .txt file has been removed, the original header of the file is as follow:

::

    Title: Calibrations of Atmospheric Parameters Obtained from
           the First Year of SDSS-III Apogee Observations
    Authors: Meszaros Sz., Holtzman J., Garcia Perez A.E., Allende Prieto C.,
             Schiavon R.P., Basu S., Bizyaev D., Chaplin W.J., Chojnowski S.D.,
             Cunha K., Elsworth Y., Epstein C., Frinchaboy P.M., Garcia R.A.,
             Hearty F.R., Hekker S., Johnson J.A., Kallinger T., Koesterke L.,
             Majewski S.R., Martell S.L., Nidever D., Pinsonneault M.H.,
             O'Connell J., Shetrone M., Smith V.V., Wilson J.C., Zasowski G.
    Table: Properties of Stars Used for Validation of ASPCAP
    ================================================================================
    Byte-by-byte Description of file: aj485195t4_mrt.txt
    --------------------------------------------------------------------------------
       Bytes Format Units     Label    Explanations
    --------------------------------------------------------------------------------
       1- 18 A18    ---       2MASS    The 2MASS identifier (1)
      20- 27 A8     ---       Cluster  Cluster identifier
      29- 35 F7.2   km/s      RVel     Heliocentric radial velocity
      37- 42 F6.1   K         Teff     ASPCAP effective temperature
      44- 49 F6.1   K         TeffC    Corrected ASPCAP effective temperature
      51- 54 F4.2   [cm/s2]   logg     Log ASPCAP surface gravity
      56- 60 F5.2   [cm/s2]   loggC    Log corrected ASPCAP surface gravity
      62- 66 F5.2   [-]       [M/H]    ASPCAP metallicity
      68- 72 F5.2   [-]       [M/H]C   ASPCAP corrected metallicity
      74- 78 F5.2   [-]       [C/M]    ASPCAP carbon abundance
      80- 84 F5.2   [-]       [N/M]    ASPCAP nitrogen abundance
      86- 90 F5.2   [-]       [a/M]    ASPCAP {alpha} abundance
      92- 97 F6.1   ---       S/N      Signal-to-noise
      99-104 F6.3   mag       Jmag     2MASS J band magnitude
     106-111 F6.3   mag       Hmag     2MASS H band magnitude
     113-118 F6.3   mag       Kmag     2MASS K_s_ band magnitude
     120-124 F5.1   K       e_TeffC    The 1{sigma} error in TeffC
     126-130 F5.3   [-]     e_[M/H]C   The 1{sigma} error in [M/H]C
    --------------------------------------------------------------------------------
    Note (1): After DR10 was published we discovered that four stars had double
              entries with identical numbers in this table (those are deleted from
              this table, thus providing 559 stars). All calibration equations were
              derived with those four double entries in our tables, but because
              DR10 is already published we decided not to change the fitting
              equations in this paper. This problem does not affect the effective
              temperature correction.  The changes in the other fitting equations
              are completely negligible and have no affect in any scientific
              application.  The parameters published in DR10 are off by <1 K in
              case of the effective temperature error correction, and by < 0.001 dex
              for the metallicity, metallicity error, and surface gravity
              correction.
    --------------------------------------------------------------------------------

License
-------------
This project is licensed under the MIT License - see the `LICENSE`_ file for details

.. _LICENSE: LICENSE