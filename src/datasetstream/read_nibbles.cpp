#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>

namespace py = pybind11;

py::tuple read_nibbles(py::buffer chunk_buffer,
                       int64_t n_bits,
                       uint64_t carry = 0,
                       int64_t carry_bits = 0) {
    py::buffer_info info = chunk_buffer.request();
    const uint8_t* data = static_cast<const uint8_t*>(info.ptr);
    size_t length = info.size;

    // Precompute mask for extracting lower bits
    const uint64_t mask_all = (n_bits < 64 ? (uint64_t(1) << n_bits) - 1 : ~uint64_t(0));

    // Reserve approximate output size
    std::vector<uint64_t> result;
    result.reserve((length * 8 + carry_bits) / n_bits + 1);

    // Local copies for speed
    uint64_t local_carry = carry;
    int64_t local_bits = carry_bits;

    for (size_t i = 0; i < length; ++i) {
        local_carry = (local_carry << 8) | data[i];
        local_bits += 8;
        // Extract as many values as possible
        while (local_bits >= n_bits) {
            int shift = local_bits - n_bits;
            // Most significant n_bits
            uint64_t value = local_carry >> shift;
            result.push_back(value);
            // Remove extracted bits with precomputed mask
            local_carry &= (shift ? ((uint64_t(1) << shift) - 1) : 0);
            local_bits -= n_bits;
        }
    }

    return py::make_tuple(std::move(result),
                          local_carry,
                          local_bits);
}

PYBIND11_MODULE(read_nibbles_cpp, m) {
    m.def("read_nibbles", &read_nibbles,
          py::arg("chunk_buffer"),
          py::arg("n_bits"),
          py::arg("carry") = 0,
          py::arg("carry_bits") = 0,
          "Optimized: Read n_bits-sized integers from a byte stream with carry state");
}
