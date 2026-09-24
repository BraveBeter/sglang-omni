// SPDX-License-Identifier: Apache-2.0
// CPU-only live metadata snapshot. Fast path never invokes Python callbacks or
// releases the GIL. Unknown semantics use the original Python oracle instead.
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/autograd/python_variable.h>
#include <unordered_set>
#include <vector>
namespace py = pybind11;
struct Unsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
};
static void need(bool ok, const char *why) {
  if (!ok)
    throw Unsupported(why);
}
static py::object own(PyObject *obj) {
  if (!obj)
    throw py::error_already_set();
  return py::reinterpret_steal<py::object>(obj);
}
static PyObject *raw_attr(PyTypeObject *cls, PyObject *key) {
  need(cls->tp_mro && PyTuple_CheckExact(cls->tp_mro), "Unknown class MRO");
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(cls->tp_mro); ++i) {
    auto *base =
        reinterpret_cast<PyTypeObject *>(PyTuple_GET_ITEM(cls->tp_mro, i));
    need(PyType_Check(reinterpret_cast<PyObject *>(base)) &&
             PyDict_CheckExact(base->tp_dict),
         "Unknown class dictionary");
    PyObject *value = PyDict_GetItemWithError(base->tp_dict, key);
    if (value)
      return value;
    if (PyErr_Occurred())
      throw py::error_already_set();
  }
  return nullptr;
}
static bool matches(PyTypeObject *cls, PyObject *expected) {
  Py_ssize_t pos = 0;
  PyObject *key, *value;
  while (PyDict_Next(expected, &pos, &key, &value)) {
    PyObject *found = raw_attr(cls, key);
    if (value == Py_None ? found != nullptr : found != value)
      return false;
  }
  return true;
}
static PyObject *state_dict(PyObject *obj) {
  // CPython3.10 exact ABI binding; unlike getattr(__dict__), this never
  // allocates or executes a descriptor, and leaves Tensor instance dictionaries
  // untouched.
  PyObject **ptr = _PyObject_GetDictPtr(obj);
  return ptr ? *ptr : nullptr;
}
static bool contains_any(PyObject *state, PyObject *keys) {
  if (!state)
    return false;
  need(PyDict_CheckExact(state), "Unknown instance dictionary");
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(keys); ++i) {
    int found = PyDict_Contains(state, PyTuple_GET_ITEM(keys, i));
    if (found < 0)
      throw py::error_already_set();
    if (found)
      return true;
  }
  return false;
}
static py::object field(PyObject *dict, const char *key) {
  PyObject *value = PyDict_GetItemString(dict, key);
  need(value != nullptr, "Missing native module field");
  return py::reinterpret_borrow<py::object>(value);
}
static py::object full_name(PyObject *prefix, PyObject *key, PyObject *dot) {
  need(PyUnicode_CheckExact(key), "Unknown registration name");
  if (!PyUnicode_GET_LENGTH(prefix))
    return py::reinterpret_borrow<py::object>(key);
  auto first = own(PyUnicode_Concat(prefix, dot));
  return own(PyUnicode_Concat(first.ptr(), key));
}
static py::tuple ints(c10::IntArrayRef values) {
  py::tuple result(values.size());
  for (size_t i = 0; i < values.size(); ++i)
    result[i] = py::int_(values[i]);
  return result;
}
static py::tuple describe(PyObject *name, PyObject *value,
                          PyObject *tensor_keys) {
  need(THPVariable_CheckExact(value), "Custom tensor metadata");
  need(!contains_any(state_dict(value), tensor_keys),
       "Instance tensor metadata override");
  const at::Tensor &tensor = THPVariable_Unpack(value);
  auto *impl = tensor.unsafeGetTensorImpl();
  need(tensor.layout() == c10::kStrided &&
           (tensor.device().is_cpu() || tensor.device().is_cuda()) &&
           !impl->has_symbolic_sizes_strides(),
       "Unsupported tensor layout/device/symbolic metadata");
  try {
    // Original field order, inference RuntimeError -> None, no tensor
    // operation.
    auto &counter = impl->version_counter();
    py::object version =
        counter.enabled() ? py::cast(counter.current_version()) : py::none();
    auto pointer = own(PyLong_FromVoidPtr(tensor.data_ptr()));
    auto sizes = ints(tensor.sizes());
    auto strides = ints(tensor.strides());
    auto dtype = own(PyObject_Str(reinterpret_cast<PyObject *>(
        torch::getTHPDtype(tensor.scalar_type()))));
    auto device_obj = own(THPDevice_New(tensor.device()));
    auto device = own(PyObject_Str(device_obj.ptr()));
    return py::make_tuple(
        py::handle(name), own(PyLong_FromVoidPtr(value)), pointer, sizes,
        strides, py::int_(tensor.storage_offset()), dtype, device, version);
  } catch (const c10::Error &) {
    // No partial result escapes. Original oracle owns its native error
    // behavior.
    throw Unsupported("Native metadata unavailable");
  }
}
static py::tuple snapshot(py::object root, py::dict binding, py::object probe) {
  need(PyGILState_Check(), "GIL required");
  auto module_type =
      reinterpret_cast<PyTypeObject *>(binding["module_type"].ptr());
  auto tensor_type =
      reinterpret_cast<PyTypeObject *>(binding["tensor_type"].ptr());
  auto parameter_type =
      reinterpret_cast<PyTypeObject *>(binding["parameter_type"].ptr());
  auto type_meta = binding["type_meta"];
  auto abc_meta = binding["abc_meta"];
  auto meta_methods = binding["meta_methods"];
  auto module_methods = binding["module_methods"];
  auto reserved = binding["reserved"];
  auto method_keys = binding["method_keys"];
  auto tensor_keys = binding["tensor_keys"];
  auto dot = binding["dot"];
  auto training_name = binding["training_name"];
  need(PyType_IsSubtype(Py_TYPE(root.ptr()), module_type), "Non-module root");
  need(matches(module_type, module_methods.ptr()),
       "Native Module methods changed");
  need(matches(tensor_type, binding["tensor_methods"].ptr()) &&
           matches(parameter_type, binding["parameter_methods"].ptr()),
       "Tensor class metadata changed");
  need(matches(Py_TYPE(reinterpret_cast<PyObject *>(tensor_type)),
               binding["tensor_meta_methods"].ptr()),
       "Tensor metaclass changed");
  need(matches(reinterpret_cast<PyTypeObject *>(binding["dtype_type"].ptr()),
               binding["dtype_methods"].ptr()) &&
           matches(
               reinterpret_cast<PyTypeObject *>(binding["device_type"].ptr()),
               binding["device_methods"].ptr()),
       "Native string descriptor changed");
  py::object probe_before = py::none();
  if (!probe.is_none()) {
    need(PyList_CheckExact(probe.ptr()) && PyList_GET_SIZE(probe.ptr()) == 1,
         "Unknown GIL probe");
    probe_before =
        py::reinterpret_borrow<py::object>(PyList_GET_ITEM(probe.ptr(), 0));
  }
  struct Item {
    py::object name;
    py::object module;
  };
  std::vector<Item> stack;
  stack.push_back({py::str(""), root});
  std::unordered_set<PyTypeObject *> classes;
  std::unordered_set<PyObject *> seen_modules, seen_parameters, seen_buffers;
  py::list parameters, buffers, tail;
  size_t noise_count = 0;
  while (!stack.empty()) {
    Item current = std::move(stack.back());
    stack.pop_back();
    PyObject *child = current.module.ptr();
    auto *cls = Py_TYPE(child);
    need(PyType_IsSubtype(cls, module_type), "Nonmodule registration");
    if (!classes.count(cls)) {
      PyObject *meta = reinterpret_cast<PyObject *>(
          Py_TYPE(reinterpret_cast<PyObject *>(cls)));
      need(meta == type_meta.ptr() || meta == abc_meta.ptr(),
           "Unknown module metaclass");
      need(matches(reinterpret_cast<PyTypeObject *>(meta), meta_methods.ptr()),
           "Metaclass semantics changed");
      need(matches(cls, module_methods.ptr()),
           "Custom module method/attribute semantics");
      for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(cls->tp_mro); ++i) {
        auto *base =
            reinterpret_cast<PyTypeObject *>(PyTuple_GET_ITEM(cls->tp_mro, i));
        need(!contains_any(base->tp_dict, reserved.ptr()),
             "Class registration/training/noise descriptor");
      }
      classes.insert(cls);
    }
    PyObject *state = state_dict(child);
    need(state && PyDict_CheckExact(state),
         "Unknown module instance dictionary");
    need(!contains_any(state, method_keys.ptr()),
         "Instance traversal override");
    auto p = field(state, "_parameters"), b = field(state, "_buffers"),
         m = field(state, "_modules"), training = field(state, "training");
    need(PyDict_CheckExact(p.ptr()) && PyDict_CheckExact(b.ptr()) &&
             PyDict_CheckExact(m.ptr()) && PyBool_Check(training.ptr()),
         "Unknown registration/training metadata");
    if (!seen_modules.insert(child).second)
      continue;
    for (int kind = 0; kind < 2; ++kind) {
      PyObject *registry = kind ? b.ptr() : p.ptr();
      auto &seen = kind ? seen_buffers : seen_parameters;
      py::list &output = kind ? buffers : parameters;
      Py_ssize_t pos = 0;
      PyObject *key, *value;
      while (PyDict_Next(registry, &pos, &key, &value)) {
        if (value == Py_None)
          continue;
        need(THPVariable_CheckExact(value), "Custom registered tensor");
        if (!seen.insert(value).second)
          continue;
        auto name = full_name(current.name.ptr(), key, dot.ptr());
        output.append(describe(name.ptr(), value, tensor_keys.ptr()));
      }
    }
    tail.append(py::make_tuple(current.name, training_name, training));
    PyObject *noise = PyDict_GetItemString(state, "rand_noise");
    if (!noise)
      noise = PyDict_GetItemString(p.ptr(), "rand_noise");
    if (!noise)
      noise = PyDict_GetItemString(b.ptr(), "rand_noise");
    if (!noise)
      noise = PyDict_GetItemString(m.ptr(), "rand_noise");
    if (noise && PyType_IsSubtype(Py_TYPE(noise), tensor_type)) {
      need(THPVariable_CheckExact(noise), "Custom noise tensor");
      // Original root noise name intentionally begins with a dot.
      auto name = own(
          PyUnicode_Concat(current.name.ptr(), binding["noise_suffix"].ptr()));
      tail.append(describe(name.ptr(), noise, tensor_keys.ptr()));
      ++noise_count;
    }
    std::vector<Item> children;
    Py_ssize_t pos = 0;
    PyObject *key, *value;
    while (PyDict_Next(m.ptr(), &pos, &key, &value)) {
      if (value == Py_None)
        continue;
      children.push_back({full_name(current.name.ptr(), key, dot.ptr()),
                          py::reinterpret_borrow<py::object>(value)});
    }
    for (auto it = children.rbegin(); it != children.rend(); ++it)
      stack.push_back(std::move(*it));
  }
  py::tuple result(parameters.size() + buffers.size() + tail.size());
  size_t index = 0;
  for (auto part : {parameters, buffers, tail})
    for (py::handle value : part)
      result[index++] = value;
  py::dict stats;
  stats["modules"] = seen_modules.size();
  stats["classes"] = classes.size();
  stats["parameters"] = parameters.size();
  stats["buffers"] = buffers.size();
  stats["noise"] = noise_count;
  stats["gil_held"] = bool(PyGILState_Check());
  if (!probe.is_none())
    stats["gil_probe_unchanged"] =
        PyList_GET_ITEM(probe.ptr(), 0) == probe_before.ptr();
  return py::make_tuple(result, stats);
}
PYBIND11_MODULE(_epoch_native, m) {
  py::register_exception<Unsupported>(m, "Unsupported");
  m.def("snapshot", &snapshot, py::arg("root"), py::arg("binding"),
        py::arg("probe") = py::none());
}
