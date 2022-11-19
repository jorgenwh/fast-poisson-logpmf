#include <inttypes.h>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "kernels.h"

namespace py = pybind11;

PYBIND11_MODULE(source_C, m) 
{
  m.doc() = "Documentation for the source_C module";

  m.def("poisson_logpmf", [](const py::array_t<int> &k, const py::array_t<double> &r)
  {
    py::buffer_info buf = k.request();
    const int *k_data = k.data();
    const double *r_data = r.data();
    const int size = k.size();

    auto ret = py::array_t<double>(buf.size);
    double *ret_data = ret.mutable_data(); 

    poisson_logpmf_np_and_np_to_np(k_data, r_data, ret_data, size);

    return ret;
  });

  m.def("poisson_logpmf", [](long k_ptr, const py::array_t<double> &r, long out_ptr)
  {
    py::buffer_info buf = r.request();
    const double *r_data = r.data();
    const int size = r.size();
    
    const int *k_data = reinterpret_cast<int *>(k_ptr);
    double *out_data = reinterpret_cast<double *>(out_ptr);

    poisson_logpmf_cp_and_np_to_cp(k_data, r_data, out_data, size);
  });

  m.def("poisson_logpmf", [](long k_ptr, long r_ptr, long out_ptr, const unsigned int size)
  {
    const int *k_data = reinterpret_cast<int *>(k_ptr);
    const double *r_data = reinterpret_cast<double *>(r_ptr);
    double *out_data = reinterpret_cast<double *>(out_ptr);

    poisson_logpmf_cp_and_cp_to_cp(k_data, r_data, out_data, size);
  });
}
