.. TRTorch documentation master file, created by
   sphinx-quickstart on Mon May  4 13:43:16 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TRTorch
========
Ahead-of-time compilation of TorchScript / PyTorch JIT for NVIDIA GPUs
-----------------------------------------------------------------------
TRTorch is a compiler for PyTorch/TorchScript, targeting NVIDIA GPUs via NVIDIA's TensorRT Deep Learning Optimizer and Runtime.
Unlike PyTorch's Just-In-Time (JIT) compiler, TRTorch is an Ahead-of-Time (AOT) compiler, meaning that before you deploy your
TorchScript code, you go through an explicit compile step to convert a standard TorchScript program into an module targeting
a TensorRT engine. TRTorch operates as a PyTorch extention and compiles modules that integrate into the JIT runtime seamlessly.
After compilation using the optimized graph should feel no different than running a TorchScript module.
You also have access to TensorRT's suite of configurations at compile time, so you are able to specify
operating precision (FP32/FP16/INT8) and other settings for your module.

More Information / System Architecture:

* `GTC 2020 Talk <https://developer.nvidia.com/gtc/2020/video/s21671>`_

Getting Started
----------------
* :ref:`installation`
* :ref:`getting_started`
* :ref:`ptq`
* :ref:`trtorchc`
* :ref:`use_from_pytorch`
* :ref:`runtime`
* :ref:`using_dla`

.. toctree::
   :caption: Getting Started
   :maxdepth: 1
   :hidden:

   tutorials/installation
   tutorials/getting_started
   tutorials/ptq
   tutorials/trtorchc
   tutorials/use_from_pytorch
   tutorials/runtime
   tutorials/using_dla
   _notebooks/lenet

.. toctree::
   :caption: Notebooks
   :maxdepth: 1
   :hidden:

   _notebooks/lenet-getting-started
   _notebooks/ssd-object-detection-demo
   _notebooks/vgg-qat


Python API Documenation
------------------------
* :ref:`trtorch_py`

.. toctree::
   :caption: Python API Documenation
   :maxdepth: 0
   :hidden:

   py_api/trtorch
   py_api/logging

C++ API Documenation
----------------------
* :ref:`namespace_trtorch`

.. toctree::
   :caption: C++ API Documenation
   :maxdepth: 1
   :hidden:

   _cpp_api/trtorch_cpp

Contributor Documentation
--------------------------------
* :ref:`system_overview`
* :ref:`writing_converters`
* :ref:`useful_links`

.. toctree::
   :caption: Contributor Documentation
   :maxdepth: 1
   :hidden:

   contributors/system_overview
   contributors/writing_converters
   contributors/useful_links

Indices
----------------
* :ref:`supported_ops`
* :ref:`genindex`
* :ref:`search`

.. toctree::
   :caption: Indices
   :maxdepth: 1
   :hidden:

   indices/supported_ops


