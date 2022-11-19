#include <inttypes.h>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(source_C, m) 
{
  m.doc() = "Documentation for the source_C module";
}
