/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/metal.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"

extern "C" int mlx_metal_is_available(bool* res) {
  try {
    *res = mlx::core::metal::is_available();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_metal_start_capture(const char* path) {
  try {
    mlx::core::metal::start_capture(std::string(path));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_metal_stop_capture(void) {
  try {
    mlx::core::metal::stop_capture();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

// ── Lesto's addition to mlx-c: registers the buffers a custom kernel wrote in place ──
#include <memory>
#include <stdexcept>
#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"
#include "mlx/c/private/vector.h"
#include "mlx/primitives.h"
#if defined(__APPLE__) && __has_include("mlx/backend/metal/device.h")
#include "mlx/backend/metal/device.h"
#define MLX_LESTO_WRITTEN_IN_PLACE_METAL 1
#endif

namespace {
// Inputs: the n written buffers, then one output of the kernel that wrote them (its "writer"). Outputs: n arrays
// sharing the buffers' memory (as Depends). On the GPU the outputs are registered with the command encoder as
// written here (set_output_array, the registration a donated slice_update makes for the buffer it writes), so MLX
// puts a barrier before any later dispatch in this encoder that reads them, and any later encoder that reads them
// waits on this encoder's fence. The writer's output is bound as an input, so this encoder also waits on the
// writer's fence when the writer ran in an earlier encoder, and a barrier comes next when it ran in this epoch.
// Nothing is dispatched; the bindings sit at slots 30 and below, which the next dispatch rebinds or never reads.
class MarkWrittenInPlace : public mlx::core::Primitive {
 public:
  explicit MarkWrittenInPlace(mlx::core::Stream stream)
      : mlx::core::Primitive(stream) {}
  void eval_cpu(
      const std::vector<mlx::core::array>& inputs,
      std::vector<mlx::core::array>& outputs) override {
    for (size_t i = 0; i < outputs.size(); ++i) {
      outputs[i].copy_shared_buffer(inputs[i]);
    }
  }
  void eval_gpu(
      const std::vector<mlx::core::array>& inputs,
      std::vector<mlx::core::array>& outputs) override {
    for (size_t i = 0; i < outputs.size(); ++i) {
      outputs[i].copy_shared_buffer(inputs[i]);
    }
#ifdef MLX_LESTO_WRITTEN_IN_PLACE_METAL
    auto& enc = mlx::core::metal::get_command_encoder(stream());
    int slot = 30;
    enc.set_input_array(inputs.back(), slot--);
    for (auto& out : outputs) {
      enc.set_output_array(out, slot--);
    }
#endif
  }
  const char* name() const override {
    return "MarkWrittenInPlace";
  }
};
} // namespace

extern "C" int mlx_lesto_written_in_place(
    mlx_vector_array* res,
    const mlx_vector_array buffers,
    const mlx_array writer_out,
    const mlx_stream s) {
  try {
    const auto& bufs = mlx_vector_array_get_(buffers);
    if (bufs.empty() || bufs.size() > 8) {
      throw std::invalid_argument("[mlx_lesto_written_in_place] 1 to 8 buffers");
    }
    std::vector<mlx::core::array> inputs(bufs.begin(), bufs.end());
    inputs.push_back(mlx_array_get_(writer_out));
    std::vector<mlx::core::Shape> shapes;
    std::vector<mlx::core::Dtype> dtypes;
    for (const auto& b : bufs) {
      shapes.push_back(b.shape());
      dtypes.push_back(b.dtype());
    }
    mlx_vector_array_set_(
        *res,
        mlx::core::array::make_arrays(
            std::move(shapes),
            dtypes,
            std::make_shared<MarkWrittenInPlace>(mlx_stream_get_(s)),
            inputs));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

// ── Lesto's addition to mlx-c: the byte limit on a command buffer off for what the calling thread encodes ──
extern "C" int mlx_lesto_commit_on_ops_limit_only(bool on) {
#ifdef MLX_LESTO_WRITTEN_IN_PLACE_METAL
  mlx::core::metal::set_commit_on_ops_limit_only(on);
#else
  (void)on;
#endif
  return 0;
}
extern "C" uint64_t mlx_lesto_command_buffers_committed(void) {
#ifdef MLX_LESTO_WRITTEN_IN_PLACE_METAL
  return mlx::core::metal::command_buffers_committed();
#else
  return 0;
#endif
}
