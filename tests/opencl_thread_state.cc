// Hardware regression: a GPU task must enable OpenCL on its own worker thread.
// Run on a host with OpenCL; absence is a failure, never a skipped GPU check.
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include "task_wavelet.hh"
#include "task_wavelet_opencl.hh"
#include <cstdio>
#include <exception>
#include <thread>

class Input : public focusstack::ImgTask
{
public:
  explicit Input(const cv::Mat &image)
  {
    m_result = image.clone();
    m_valid_area = cv::Rect(0, 0, image.cols, image.rows);
  }
private:
  void task() override {}
};

int main()
{
  try
  {
    cv::ocl::setUseOpenCL(true);
    if (!cv::ocl::useOpenCL()) throw std::runtime_error("GPU OpenCL required");
    std::printf("OpenCL device: %s\n", cv::ocl::Device::getDefault().name().c_str());
    std::exception_ptr error;
    std::thread worker([&]() {
      try
      {
        cv::Mat image(256, 256, CV_8U);
        cv::RNG random(42);
        random.fill(image, cv::RNG::UNIFORM, 0, 256);
        auto input = std::make_shared<Input>(image);
        auto gpu = std::make_shared<focusstack::Task_Wavelet_OpenCL>(input, false);

        // Reproduce the production state even when this thread's default is on.
        cv::ocl::setUseOpenCL(false);
        gpu->run();
        if (!cv::ocl::useOpenCL()) throw std::runtime_error("forward did not enable OpenCL");
        focusstack::Task_Wavelet cpu(input, false);
        cpu.run();
        double forward_error = cv::norm(cpu.img(), gpu->img(), cv::NORM_INF);
        if (forward_error > 0.01) throw std::runtime_error("forward differs from CPU wavelets");

        focusstack::Task_Wavelet_OpenCL inverse(gpu, true);
        cv::ocl::setUseOpenCL(false);
        inverse.run();
        if (!cv::ocl::useOpenCL()) throw std::runtime_error("inverse did not enable OpenCL");
        double inverse_error = cv::norm(image, inverse.img(), cv::NORM_INF);
        if (inverse_error > 1) throw std::runtime_error("inverse differs from original image");
        std::printf("PASS disabled-thread forward/inverse; max errors %.6f / %.0f\n",
                    forward_error, inverse_error);
      }
      catch (...) { error = std::current_exception(); }
    });
    worker.join();
    if (error) std::rethrow_exception(error);
    return 0;
  }
  catch (const std::exception &e)
  {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
