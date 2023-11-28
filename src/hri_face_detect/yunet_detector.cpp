#include <cstdlib>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "facedetectcnn.h"

// define the result_buffer_ size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x9000

namespace py = pybind11;

class YuNetDetector
{
public:
  YuNetDetector()
  {
    result_buffer_ = new unsigned char[DETECT_BUFFER_SIZE]();
  }

  ~YuNetDetector()
  {
    delete result_buffer_;
  }

  std::vector<std::vector<short>> detect(py::array_t<unsigned char> image_buffer, int width, int height, int step)
  {
    /* for a guideline on the detector usage and output decoding see:
     * https://github.com/ShiqiYu/libfacedetection/blob/fb0c773e1fbe30479e5e7c32888de41b9e818b4d/src/facedetectcnn-model.cpp#L203
     * https://github.com/ShiqiYu/libfacedetection/blob/fb0c773e1fbe30479e5e7c32888de41b9e818b4d/src/facedetectcnn.cpp#L773
    */

    py::buffer_info image_buffer_info = image_buffer.request();

    py::gil_scoped_release release;
    facedetect_cnn(result_buffer_, static_cast<unsigned char*>(image_buffer_info.ptr), width, height, step);
    py::gil_scoped_acquire acquire;

    int n_faces = result_buffer_ ? *(reinterpret_cast<int*>(result_buffer_)) : 0;
    std::vector<std::vector<short>> faces(n_faces);
    for (int i = 0; i < n_faces; ++i)
    {
      short* p = ((short*)(result_buffer_ + 4)) + 16 * size_t(i);
      faces[i].assign(p, p + 15);
    }
    return faces;
  }

private:
  unsigned char* result_buffer_;
};

PYBIND11_MODULE(yunet_detector, m)
{
  m.doc() = "YuNet face detection module";

  py::class_<YuNetDetector>(m, "YuNetDetector").def(py::init<>()).def("detect", &YuNetDetector::detect);
}
