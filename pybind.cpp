#include <pybind11/pybind11.h>
namespace py = pybind11;

template<typename T> class Net;

template<typename T>
static void bind_net(py::module_& m, const char* pyname) {
  using NetT = Net<T>;
  py::class_<NetT>(m, pyname)
    .def(py::init<int,int,int,int,int>(),
         py::arg("n"), py::arg("in_dim"), py::arg("out_dim"),
         py::arg("hidden_dim")=0, py::arg("hidden_layers")=0)
    .def("forward",    &NetT::forward,    py::arg("x"))
    .def("backward",   &NetT::backward)
    .def("zero_grad",  &NetT::zero_grad)
    .def("update",     &NetT::update,     py::arg("lr"));
}

PYBIND11_MODULE(myml, m) {
  bind_net<float>( m, "Net");
}