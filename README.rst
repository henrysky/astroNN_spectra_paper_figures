
Getting Started
=================

This repository is to make sure all figures and results are reproducible by anyone easily for astroNN spectra analysis
paper.

``astroNN_0520_run001`` is a trained astroNN model to infer 24 stellar parameters and stellar intrinsic brightness from APOGEE spectra.
Please notice the trained ``astroNN_0520_run001`` model is used for TESTING  propose only to ensure the code run correctly with reasonable result.

To load the model, open python outside ``astroNN_0520_run001``

.. code-block:: python

   from astroNN.models import load_folder

   neuralnet = load_folder('astroNN_0520_run001')
   # neuralnet is an astroNN neural network object, you can access keras API via neuralnet.keras_model

   # To get what the output neurones are representing
   print(neuralnet.targetname)

`astroNN Github`_

`astroNN Documentation`_

.. _astroNN Github: https://github.com/henrysky/astroNN

.. _astroNN Documentation: http://astronn.readthedocs.io/

Some graphs require the package `mw_plot` from my `milkyway_plot Github`_

.. _milkyway_plot Github: https://github.com/henrysky/milkyway_plot

Project Abstract
==================

Deep learning using artificial neural network is getting attention from astronomers with its potential in data-driven astronomy.
However, such methodology usually do not capture uncertainty intervals and do not deal with incomplete data. In recent development driven by
the demand of uncertainties estimation from industries, Bayesian neural network with dropout variational inference offers an opportunity
to achieve such goal with a simple computationally cheap setup. In this work, we will demonstrate using dropout variational inference technique
as well as customized objective function to deal with incomplete data to infer 24 stellar parameters from APOGEE stellar spectra and APOGEE
Stellar Parameter and Chemical Abundance Pipeline
:math:`($T_{\text{eff}}$, $\log{g}$, $M$, $Alpha$, $C$, $CI$, $N$, $O$, $Na$, $Mg$, $Al$, $Si$, $P$, $S$, $K$, $Ca$, $Ti$, $TiII$, $V$, $Cr$, $Mn$, $Fe$, $Co$, $Ni$)`
with reasonable uncertainty interval. By training on high SNR combined spectra mainly on
red giant stars, the testing on corresponding low SNR individual visit spectra shows the neural network inferred all stellar parameters with excellent
accuracy and reasonable model dependency on both prediction and uncertainty estimation. With upcoming precise Gaia parallax in Data Release 2, we suggest
this technique can also be used to infer intrinsic brightness from stellar spectra without any stellar model assumption. **astroNN**, a well tested active
developing open-source python package created for this work as well as designed to be a general package for deep learning in astronomy, is available at
https://github.com/henrysky/astroNN with extensive documentation at http://astroNN.readthedocs.io.

Authors
-------------
-  | **Henry Leung** - henrysky_
   | Student, Department of Astronomy and Astrophysics, University of Toronto
   | Contact Henry: henrysky.leung [at] mail.utoronto.ca

-  | **Jo Bovy** - jobovy_
   | Professor, Department of Astronomy and Astrophysics, University of Toronto

Information on ``aj485195t4_mrt.txt``
--------------------------------------

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
.. _henrysky: https://github.com/henrysky
.. _jobovy: https://github.com/jobovy