
Getting Started
=================

This repository is to make sure all figures and results are reproducible by anyone easily for astroNN spectra analysis
paper.

``astroNN_0414_run001`` is a trained astroNN model to infer 24 stellar parameters and stellar intrinsic brightness from APOGEE spectra.
Please notice the trained ``astroNN_0414_run001`` model is used for TESTING  propose only to ensure the code run correctly with reasonable result.

To load the model, open python outside ``astroNN_0414_run001``

.. code-block:: python

   from astroNN.models import load_folder

   neuralnet = load_folder('astroNN_0414_run001')
   # neuralnet is an astroNN neural network object, you can access keras API via neuralnet.keras_model

   # To get what the output neurones are representing
   print(neuralnet.targetname)

`astroNN Github`_

`astroNN Documentation`_

.. _astroNN Github: https://github.com/henrysky/astroNN

.. _astroNN Documentation: http://astronn.readthedocs.io/

Project Abstract
==================

Deep learning using artificial neural network is getting attention from astronomers with its potential in data-driven astronomy.
However, such methodology usually do not capture uncertainty intervals and do not deal with incomplete data. In recent development driven by
the demand of uncertainties estimation from industries, Bayesian neural network with dropout variational inference offers an opportunity
to achieve such goal with a simple computationally cheap setup. In this work, we will demonstrate using dropout variational inference technique
as well as customized objective function to deal with incomplete data to infer 24 stellar parameters from APOGEE stellar spectra and APOGEE
Stellar Parameter and Chemical Abundance Pipeline
:math:`($T_{\text{eff}}$, $\log{g}$, $M$, $Alpha$, $C$, $CI$, $N$, $O$, $Na$, $Mg$, $Al$, $Si$, $P$, $S$, $K$, $Ca$, $Ti$, $TiII$, $V$, $Cr$, $Mn$, $Fe$, $Co$, $Ni$)'
with reasonable uncertainty interval. By training on high SNR combined spectra mainly on
red giant stars, the testing on corresponding low SNR individual visit spectra shows the neural network inferred all stellar parameters with excellent
accuracy and reasonable model dependency on both prediction and uncertainty estimation. With upcoming precise Gaia parallax in Data Release 2, we suggest
this technique can also be used to infer intrinsic brightness from stellar spectra without any stellar model assumption. **astroNN**, a well tested active
developing open-source python package created for this work as well as designed to be a general package for deep learning in astronomy, is available at
https://github.com/henrysky/astroNN with extensive documentation at http://astroNN.readthedocs.io.

Authors
-------------
-  | **Henry Leung** - henrysky_
   | Department of Astronomy and Astrophysics, University of Toronto
   | Contact Henry: henrysky.leung [at] mail.utoronto.ca

-  | **Jo Bovy** - jobovy_
   | Professor, Department of Astronomy and Astrophysics, University of Toronto

License
-------------
This project is licensed under the MIT License - see the `LICENSE`_ file for details

.. _LICENSE: LICENSE
.. _henrysky: https://github.com/henrysky
.. _jobovy: https://github.com/jobovy