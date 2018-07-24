
Abstract
===========

Deep learning using artificial neural network is getting attention from astronomers with its potential in data-driven astronomy.
However, such methodology usually do not capture uncertainty intervals and do not deal with incomplete data. In recent development driven by
the demand of uncertainties estimation from industries, Bayesian neural network with dropout variational inference offers an opportunity
to achieve such goal with a simple computationally cheap setup. In this work, we will demonstrate using dropout variational inference technique
as well as customized objective function to deal with incomplete data to infer 22 stellar parameters from APOGEE stellar spectra and APOGEE
Stellar Parameter and Chemical Abundance Pipeline
:math:`(Teff, log(g), C, CI, N, O, Na, Mg, Al, Si, P, S, K, Ca, Ti, TiII, V, Cr, Mn, Fe, Co, Ni)`
with reasonable uncertainty interval. By training on high SNR combined spectra mainly on red giant stars, the testing on
corresponding low SNR individual visit spectra shows the neural network inferred all stellar parameters with excellent
accuracy and reasonable model dependency on both prediction and uncertainty estimation. With upcoming precise Gaia parallax in Data Release 2, we suggest
this technique can also be used to infer intrinsic brightness from stellar spectra without any stellar model assumption. **astroNN**, a well tested active
developing open-source python package created for this work as well as designed to be a general package for deep learning in astronomy, is available at
https://github.com/henrysky/astroNN with extensive documentation at http://astroNN.readthedocs.io.

Getting Started
=================

This repository is to make sure all figures and results are reproducible by anyone easily for this paper. Python 3.6 or
above is required. The presence of Nvidia GPUs 900 series or above is recommended for Tensorlfow hardware acceleration.

To get started, this paper use `astroNN`_ developed by the authors and tested with **astroNN 1.0.0 (Not yet released)**. Extensive
documentation at http://astroNN.readthedocs.io and astroNN quick start guide at http://astronn.readthedocs.io/en/latest/quick_start.html

Some notebooks also make use of my `milkyway_plot`_ to plot on milkyway.

.. _astroNN: https://github.com/henrysky/astroNN
.. _milkyway_plot: https://github.com/henrysky/milkyway_plot

To continuum normalize arbitrary APOGEE spectrum, see: http://astronn.readthedocs.io/en/latest/tools_apogee.html#continuum-normalization-of-apogee-spectra

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
    | It describes training neural network with smaller datasets and see the effects.

.. _Datasets_Data_Reduction.ipynb: Datasets_Data_Reduction.ipynb
.. _Training.ipynb: Training.ipynb
.. _Inference_highSNR.ipynb: Inference_highSNR.ipynb
.. _Inference.ipynb: Inference.ipynb
.. _Inference_ApogeeBCNN.ipynb: Inference_ApogeeBCNN.ipynb
.. _Open_Globular_Cluster_Benchmark.ipynb: Open_Globular_Cluster_Benchmark.ipynb
.. _Apogee_dr14_NN_Catalog.ipynb: Apogee_dr14_NN_Catalog.ipynb
.. _ASPCAP_Normalization.ipynb: ASPCAP_Normalization.ipynb
.. _Small_Data.ipynb: Small_Data.ipynb

Nueral Net Models
------------------
- ``astroNN_0606_run001`` is a trained astroNN's `ApogeeBCNN()`_ class model to infer 22 stellar parameters from APOGEE continuum normalized spectra.

- ``astroNN_0617_run001`` is a trained astroNN's `ApogeeBCNNCensored()`_ class model to infer 22 stellar parameters from APOGEE continuum normalized spectra.

- ``aspcapStar_BCNNCensored`` is a trained astroNN's `ApogeeBCNNCensored()`_ class model to infer 22 stellar parameters from APOGEE ASPCAP-normalized spectra, with exactly the same model architecture as ``astroNN_0617_run001``..

- ``small_data_fixed_****`` are trained astroNN's `ApogeeBCNNCensored()`_ class models with small dataset, with exactly the same model architecture as ``astroNN_0617_run001``.

- ``small_data_adaptive_****`` are trained astroNN's `ApogeeBCNNCensored()`_ class models with small dataset, with adaptive model architecture as dataset gets smaller.

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

``astroNN_apogee_dr14_catalog.fits`` is compiled prediction with ``astroNN_0617_run001`` on the whole Apogee DR14. To load it with python

.. code-block:: python

    from astropy.io import fits

    f = fits.open("astroNN_apogee_dr14_catalog.fits")
    apogee_id = f[1].data['APOGEE_ID']  # APOGEE's apogee id
    location_id = f[1].data['LOCATION_ID']  # APOGEE DR14 location id
    ra = f[1].data['RA']  #J2000 RA
    dec = f[1].data['DEC']  #J2000 DEC

    # the order of the array is [Teff, log(g), C/H, C1/H, N/H, O/H, Na/H, Mg/H, Al/H, Si/H, P/H, S/H, K/H, Ca/H, Ti/H,
    # Ti2/H, V/H, Cr/H, Mn/H, Fe/H, Co/H, Ni/H]
    nn_prediction = f[1].data['astroNN']  #neural network prediction, contains -9999.
    nn_uncertainty = f[1].data['astroNN_error']  #neural network uncertainty, contains -9999.


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

    #using default continuum and bitmask values to continuum normalize
    norm_spec, norm_spec_err = apogee_continuum(spectrum, spectrum_err,
                                                bitmask=spectrum_bitmask, dr=14)

    #load neural net
    neuralnet = load_folder('astroNN_0617_run001')

    # inference
    pred, pred_err = neuralnet.test(norm_spec)

    print(neuralnet.targetname)  # output neurons representation
    print(pred)  # prediction
    print(pred_err['total'])  # prediction uncertainty

Authors
=================
-  | **Henry Leung** - henrysky_
   | Student, Department of Astronomy and Astrophysics, University of Toronto
   | Contact Henry: henrysky.leung [at] mail.utoronto.ca

-  | **Jo Bovy** - jobovy_
   | Professor, Department of Astronomy and Astrophysics, University of Toronto

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