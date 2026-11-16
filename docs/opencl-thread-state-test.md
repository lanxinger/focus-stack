# OpenCL worker thread regression

OpenCV keeps the OpenCL enable flag on each thread. Checking the GPU on the
main thread does not guarantee that a later worker has OpenCL enabled. Explicit
wavelet kernels need GPU-backed UMats; a CPU allocation has no `cl_mem` handle.
Each OpenCL wavelet task therefore enables and checks OpenCL before allocating
its input or temporary buffers. This covers both forward and inverse tasks.

On a host with the OpenCL runtime and GPU driver installed, run:

```sh
make run_opencl_thread_state_test
```

The test initializes the GPU on the main thread, then explicitly disables
OpenCL on a separate worker before each wavelet task. Both tasks must restore
OpenCL. Forward coefficients must match the CPU implementation within 0.01;
inverse output must match the original 8-bit image within one intensity level.
Missing OpenCL fails the test rather than skipping it.

The original implementation fails this test with a missing buffer handle.
On NVIDIA T4 / driver 580.159.03 / OpenCV 4.13.0, the patched implementation
passed with maximum forward error 0.000687 and inverse error zero. It also
completed 60 consecutive real-bracket stacks; the original binary failed on
the fifth run. Three successful original/patched real-bracket comparisons
produced pixel-identical images. These checks preserve the existing wavelet
algorithm and GPU acceleration.
