//   Ant Group Copyright (c) 2004-2020 All Rights Reserved.
// Copyright 2016 The TensorFlow Authors. All Rights Reserved.

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS
#include "tensorflow/core/kernels/conv_ops.h"
#include <string.h>
#include <atomic>
#include <map>
#include <vector>
#include <algorithm>
#include <limits>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/deep_conv2d.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "tensorflow/core/kernels/conv_grad_shape_utils.h"
#include "tensorflow/core/kernels/fill_functor.h"
#ifdef TENSORFLOW_USE_LIBXSMM_CONVOLUTIONS
#include "tensorflow/core/kernels/xsmm_conv2d.h"
#endif
#include "tensorflow/core/util/work_sharder.h"
#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif
#include "absl/base/dynamic_annotations.h"

namespace {

// image to column,就是把卷积窗口中的数拉成一行
// Returns in 'col_data', image patches in storage order (height, width, depth)
// extracted from image at 'input_data', which is required to be in storage
// order (batch, height, width, depth).
template <typename T>
void Im2col(const T* input_data, const int depth, const int height,
            const int width, const int filter_h, const int filter_w,
            const int pad_t, const int pad_l, const int pad_b, const int pad_r,
            const int stride_h, const int stride_w, T* col_data) {
  int height_col = (height + pad_t + pad_b - filter_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;

  int h_pad = -pad_t;
  for (int h = 0; h < height_col; ++h) {
    int w_pad = -pad_l;
    for (int w = 0; w < width_col; ++w) {
      for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
        for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            memcpy(col_data, input_data + (ih * width + iw) * depth,
                   sizeof(T) * depth);
          } else {
            // This should be simply padded with zero.
            memset(col_data, 0, sizeof(T) * depth);
          }
          col_data += depth;
        }
      }
      w_pad += stride_w;
    }
    h_pad += stride_h;
  }
}

}  // namespace


namespace tensorflow{
typedef Eigen::ThreadPoolDevice CPUDevice;

#define TF_REQUIRES(EXP, STATUS)                \
  do {                                          \
    if (!TF_PREDICT_TRUE(EXP)) return (STATUS); \
  } while (false)

Status InitConv2DParameters(const OpKernelConstruction* context,
                            Conv2DParameters* params) {
  TF_RETURN_IF_ERROR(context->GetAttr("dilations", &params->dilations));
  TF_RETURN_IF_ERROR(context->GetAttr("strides", &params->strides));
  TF_RETURN_IF_ERROR(context->GetAttr("padding", &params->padding));
  if (context->HasAttr("explicit_paddings")) {
    TF_RETURN_IF_ERROR(
        context->GetAttr("explicit_paddings", &params->explicit_paddings));
  }
  string data_format_string;
  TF_RETURN_IF_ERROR(context->GetAttr("data_format", &data_format_string));
  TF_REQUIRES(FormatFromString(data_format_string, &params->data_format),
              errors::InvalidArgument("Invalid data format"));

  const auto& strides = params->strides;
  const auto& dilations = params->dilations;
  const auto& data_format = params->data_format;

  TF_REQUIRES(dilations.size() == 4,
              errors::InvalidArgument("Sliding window dilations field must "
                                      "specify 4 dimensions"));
  TF_REQUIRES(strides.size() == 4,
              errors::InvalidArgument("Sliding window strides field must "
                                      "specify 4 dimensions"));
  const int64 stride_n = GetTensorDim(strides, data_format, 'N');
  const int64 stride_c = GetTensorDim(strides, data_format, 'C');
  const int64 stride_h = GetTensorDim(strides, data_format, 'H');
  const int64 stride_w = GetTensorDim(strides, data_format, 'W');
  TF_REQUIRES(
      stride_n == 1 && stride_c == 1,
      errors::InvalidArgument("Current implementation does not yet support "
                              "strides in the batch and depth dimensions."));
  TF_REQUIRES(stride_h > 0 && stride_w > 0,
              errors::InvalidArgument(
                  "Row and column strides should be larger than 0."));

  const int64 dilation_n = GetTensorDim(dilations, data_format, 'N');
  const int64 dilation_c = GetTensorDim(dilations, data_format, 'C');
  const int64 dilation_h = GetTensorDim(dilations, data_format, 'H');
  const int64 dilation_w = GetTensorDim(dilations, data_format, 'W');
  TF_REQUIRES(
      dilation_n == 1 && dilation_c == 1,
      errors::InvalidArgument("Current implementation does not yet support "
                              "dilations in the batch and depth dimensions."));
  TF_REQUIRES(
      dilation_h > 0 && dilation_w > 0,
      errors::InvalidArgument("Dilated rates should be larger than 0."));

  TF_RETURN_IF_ERROR(CheckValidPadding(params->padding,
                                       params->explicit_paddings,
                                       /*num_dims=*/4, data_format));

  return Status::OK();
}


template <typename Device, typename T>
struct LaunchGeneric {
  void operator()(OpKernelContext* ctx, const Tensor& input,
                  const Tensor& filter, int row_stride, int col_stride,
                  int row_dilation, int col_dilation, const Padding& padding,
                  const std::vector<int64>& explicit_paddings, Tensor* output,
                  TensorFormat data_format) {
    CHECK(data_format == FORMAT_NHWC) << "Generic conv implementation only "
                                         "supports NHWC tensor format for now.";
    if (filter.dim_size(0) == 1 && filter.dim_size(1) == 1 && row_stride == 1 &&
        col_stride == 1 && (padding == SAME || padding == VALID)) {
      // For 1x1 kernel, the 2D convolution is reduced to matrix
      // multiplication.
      //
      // TODO(vrv): We should be able to call SpatialConvolution
      // and it will produce the same result, but doing so
      // led to NaNs during training.  Using matmul instead for now.
      int conv_width = 1;  // Width for the convolution step.
      for (int i = 0; i < 3; ++i) {
        conv_width *= output->dim_size(i);
      }

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      functor::MatMulConvFunctor<Device, T>()(
          ctx->eigen_device<Device>(),
          output->shaped<T, 2>({conv_width, filter.dim_size(3)}),
          input.shaped<T, 2>({conv_width, filter.dim_size(2)}),
          filter.shaped<T, 2>({filter.dim_size(2), filter.dim_size(3)}),
          dim_pair);
    } else if (filter.dim_size(0) == input.dim_size(1) &&
               filter.dim_size(1) == input.dim_size(2) && row_dilation == 1 &&
               col_dilation == 1 && padding == VALID) {
      // If the input data and filter have the same height/width,
      // the 2D convolution is reduced to matrix multiplication.
      const int k =  // Length of reduction dimension.
          filter.dim_size(0) * filter.dim_size(1) * filter.dim_size(2);

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      functor::MatMulConvFunctor<Device, T>()(
          ctx->eigen_device<Device>(),
          output->shaped<T, 2>({input.dim_size(0), filter.dim_size(3)}),
          input.shaped<T, 2>({input.dim_size(0), k}),
          filter.shaped<T, 2>({k, filter.dim_size(3)}), dim_pair);
    } else {
      if (padding == EXPLICIT) {
        functor::SpatialConvolution<Device, T>()(
            ctx->eigen_device<Device>(), output->tensor<T, 4>(),
            input.tensor<T, 4>(), filter.tensor<T, 4>(), row_stride, col_stride,
            row_dilation, col_dilation, static_cast<int>(explicit_paddings[2]),
            static_cast<int>(explicit_paddings[3]),
            static_cast<int>(explicit_paddings[4]),
            static_cast<int>(explicit_paddings[5]));
      } else {
        functor::SpatialConvolution<Device, T>()(
            ctx->eigen_device<Device>(), output->tensor<T, 4>(),
            input.tensor<T, 4>(), filter.tensor<T, 4>(), row_stride, col_stride,
            row_dilation, col_dilation, BrainPadding2EigenPadding(padding));
      }
    }
  }
};

template <typename Device, typename T>
class LaunchDeepConvOp {
 public:
  static bool Run(OpKernelContext* ctx, const Tensor& input,
                  const Tensor& filter, int batch, int input_rows,
                  int input_cols, int in_depth, int filter_rows,
                  int filter_cols, int pad_rows, int pad_cols, int out_rows,
                  int /*out_cols*/, int /*out_depth*/, int /*dilation_rows*/,
                  int /*dilation_cols*/, int /*stride_rows*/,
                  int /*stride_cols*/, Tensor* /*output*/,
                  TensorFormat /*data_format*/) {
    return false;
  }
};

// Conditionally launches DeepConv operation based on convolution parameters.
template <>
class LaunchDeepConvOp<CPUDevice, float> {
 public:
  static bool Run(OpKernelContext* ctx, const Tensor& input,
                  const Tensor& filter, int batch, int input_rows,
                  int input_cols, int in_depth, int filter_rows,
                  int filter_cols, int pad_rows, int pad_cols, int out_rows,
                  int out_cols, int out_depth, int dilation_rows,
                  int dilation_cols, int stride_rows, int stride_cols,
                  Tensor* output, TensorFormat data_format) {
    if (data_format != FORMAT_NHWC || dilation_rows != 1 ||
        dilation_cols != 1 ||
        !CanUseDeepConv2D(stride_rows, stride_cols, filter_rows, filter_cols,
                          in_depth, out_depth, out_rows, out_cols)) {
      return false;
    }

    Conv2DArgs args;
    args.batch = batch;
    args.in_rows = input_rows;
    args.in_cols = input_cols;
    args.in_depth = in_depth;
    args.filter_rows = filter_rows;
    args.filter_cols = filter_cols;
    args.pad_rows = pad_rows;
    args.pad_cols = pad_cols;
    args.out_rows = out_rows;
    args.out_cols = out_cols;
    args.out_depth = out_depth;

    auto input_ptr = input.template flat<float>().data();
    auto filter_ptr = filter.template flat<float>().data();
    auto output_ptr = output->template flat<float>().data();

    functor::DeepConv2D<CPUDevice, float>()(ctx, args, input_ptr, filter_ptr,
                                            output_ptr);
    return true;
  }
};


#ifdef TENSORFLOW_USE_LIBXSMM_CONVOLUTIONS
template <typename Device, typename T>
class LaunchXsmmConvOp {
 public:
  static bool Run(OpKernelContext* ctx, const Tensor& input,
                  const Tensor& filter, int batch, int input_rows,
                  int input_cols, int in_depth, int filter_rows,
                  int filter_cols, int pad_rows, int pad_cols, int out_rows,
                  int out_cols, int out_depth, int stride_rows, int stride_cols,
                  int dilation_rows, int dilation_cols, Tensor* output,
                  TensorFormat data_format) {
    return false;
  }
};

template <>
class LaunchXsmmConvOp<CPUDevice, float> {
 public:
  static bool Run(OpKernelContext* ctx, const Tensor& input,
                  const Tensor& filter, int batch, int input_rows,
                  int input_cols, int in_depth, int filter_rows,
                  int filter_cols, int pad_rows, int pad_cols, int out_rows,
                  int out_cols, int out_depth, int dilation_rows,
                  int dilation_cols, int stride_rows, int stride_cols,
                  Tensor* output, TensorFormat data_format) {
    auto num_threads =
        ctx->device()->tensorflow_cpu_worker_threads()->num_threads;
    // See libxsmm_dnn.h for this struct definition.
    libxsmm_dnn_conv_desc desc;
    desc.N = batch;
    desc.C = in_depth;
    desc.H = input_rows;
    desc.W = input_cols;
    desc.K = out_depth;
    desc.R = filter_rows;
    desc.S = filter_cols;
    desc.u = stride_rows;
    desc.v = stride_cols;
    desc.pad_h = pad_rows;
    desc.pad_w = pad_cols;
    desc.pad_h_in = 0;
    desc.pad_w_in = 0;
    desc.pad_h_out = 0;
    desc.pad_w_out = 0;
    desc.threads = num_threads;
    desc.algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
    desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NHWC;
    desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
    desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
    desc.options = LIBXSMM_DNN_CONV_OPTION_OVERWRITE;
    desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
    desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    if (dilation_rows != 1 || dilation_cols != 1 ||
        !CanUseXsmmConv2D(desc, data_format)) {
      return false;
    }

    auto input_ptr = input.template flat<float>().data();
    auto filter_ptr = filter.template flat<float>().data();
    auto output_ptr = output->template flat<float>().data();

    bool success = functor::XsmmFwdConv2D<CPUDevice, float>()(
        ctx, desc, input_ptr, filter_ptr, output_ptr);
    return success;
  }
};
#endif

template <typename T>
struct LaunchConv2DOp<CPUDevice, T> {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& input, const Tensor& filter, int row_dilation,
                  int col_dilation, int row_stride, int col_stride,
                  const Padding& padding,
                  const std::vector<int64>& explicit_paddings, Tensor* output,
                  TensorFormat data_format) {
    if (data_format != FORMAT_NHWC) {
      ctx->SetStatus(errors::Unimplemented(
          "The Conv2D op currently only supports the NHWC tensor format on the "
          "CPU. The op was given the format: ",
          ToString(data_format)));
      return;
    }
    const int64 in_depth = GetTensorDim(input, data_format, 'C');
    OP_REQUIRES(ctx, in_depth == filter.dim_size(2),
                errors::Unimplemented(
                    "The Conv2D op currently does not support grouped "
                    "convolutions on the CPU. A grouped convolution was "
                    "attempted to be run because the input depth of ",
                    in_depth, " does not match the filter input depth of ",
                    filter.dim_size(2)));

    for (int64 explicit_padding : explicit_paddings) {
      if (!FastBoundsCheck(explicit_padding, std::numeric_limits<int>::max())) {
        ctx->SetStatus(errors::InvalidArgument("filter too large"));
        return;
      }
    }
    LaunchGeneric<CPUDevice, T>()(ctx, input, filter, row_stride, col_stride,
                                  row_dilation, col_dilation, padding,
                                  explicit_paddings, output, data_format);
  }
};



Status ComputeINT64Conv2DDimension(const Conv2DParameters& params,
                              const Tensor& input, const Tensor& filter,
                              Conv2DDimensions* dimensions) {
  // Check that 2D convolution input and filter have exactly 4 dimensions.
  TF_REQUIRES(input.dims() == 4,
              errors::InvalidArgument("input must be 4-dimensional",
                                      input.shape().DebugString()));
  TF_REQUIRES(filter.dims() == 4,
              errors::InvalidArgument("filter must be 4-dimensional: ",
                                      filter.shape().DebugString()));
  for (int i = 0; i < 3; i++) {
    TF_REQUIRES(
        FastBoundsCheck(filter.dim_size(i), std::numeric_limits<int>::max()),
        errors::InvalidArgument("filter too large"));
  }

  // The last dimension for input is in_depth. Check that it is the same as the
  // filter's in_depth or it is evenly divisible by filter's in_depth.
  const int64 in_depth_raw = GetTensorDim(input, params.data_format, 'C');
  const int64 patch_depth_raw = filter.dim_size(2);
  TF_REQUIRES(FastBoundsCheck(in_depth_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Input depth too large"));
  TF_REQUIRES(FastBoundsCheck(patch_depth_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Patch depth too large"));
  const int in_depth = static_cast<int>(in_depth_raw);
  const int patch_depth = static_cast<int>(patch_depth_raw);
  TF_REQUIRES(patch_depth > 0,
              errors::InvalidArgument(
                  "filter depth must be stricly positive, got ", patch_depth));
  TF_REQUIRES(in_depth % patch_depth == 0,
              errors::InvalidArgument(
                  "input depth must be evenly divisible by filter depth: ",
                  in_depth, " vs ", patch_depth));

  // The last dimension for filter is out_depth.
  const int out_depth = static_cast<int>(filter.dim_size(3));

  // The second dimension for input is rows/height.
  // The first dimension for filter is rows/height.
  const int64 input_rows_raw = GetTensorDim(input, params.data_format, 'H');
  TF_REQUIRES(FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Input rows too large"));
  // 强制类型转换
  const int input_rows = static_cast<int>(input_rows_raw);
  const int filter_rows = static_cast<int>(filter.dim_size(0));

  // The third dimension for input is columns/width.
  // The second dimension for filter is columns/width.
  const int64 input_cols_raw = GetTensorDim(input, params.data_format, 'W');
  TF_REQUIRES(FastBoundsCheck(input_cols_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Input cols too large"));
  const int input_cols = static_cast<int>(input_cols_raw);
  const int filter_cols = static_cast<int>(filter.dim_size(1));

  // The first dimension for input is batch.
  const int64 batch_raw = GetTensorDim(input, params.data_format, 'N');
  TF_REQUIRES(FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("batch is too large"));
  const int batch = static_cast<int>(batch_raw);

  // Take the stride and dilation from the second and third dimensions only (we
  // do not support striding or dilation on the batch or depth dimension).
  const int stride_rows = GetTensorDim(params.strides, params.data_format, 'H');
  const int stride_cols = GetTensorDim(params.strides, params.data_format, 'W');
  const int dilation_rows =
      GetTensorDim(params.dilations, params.data_format, 'H');
  const int dilation_cols =
      GetTensorDim(params.dilations, params.data_format, 'W');

  int64 pad_rows_before, pad_rows_after, pad_cols_before, pad_cols_after;
  if (params.padding == Padding::EXPLICIT) {
    GetExplicitPaddingForDim(params.explicit_paddings, params.data_format, 'H',
                             &pad_rows_before, &pad_rows_after);
    GetExplicitPaddingForDim(params.explicit_paddings, params.data_format, 'W',
                             &pad_cols_before, &pad_cols_after);
  }

  // Compute windowed output sizes for rows and columns.
  int64 out_rows = 0, out_cols = 0;
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerboseV2(
      input_rows, filter_rows, dilation_rows, stride_rows, params.padding,
      &out_rows, &pad_rows_before, &pad_rows_after));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerboseV2(
      input_cols, filter_cols, dilation_cols, stride_cols, params.padding,
      &out_cols, &pad_cols_before, &pad_cols_after));

  dimensions->batch = batch;
  dimensions->input_rows = input_rows;
  dimensions->input_cols = input_cols;
  dimensions->in_depth = in_depth;
  dimensions->filter_rows = filter_rows;
  dimensions->filter_cols = filter_cols;
  dimensions->patch_depth = patch_depth;
  dimensions->out_depth = out_depth;
  dimensions->stride_rows = stride_rows;
  dimensions->stride_cols = stride_cols;
  dimensions->dilation_rows = dilation_rows;
  dimensions->dilation_cols = dilation_cols;
  dimensions->out_rows = out_rows;
  dimensions->out_cols = out_cols;
  dimensions->pad_rows_before = pad_rows_before;
  dimensions->pad_rows_after = pad_rows_after;
  dimensions->pad_cols_before = pad_cols_before;
  dimensions->pad_cols_after = pad_cols_after;

  return Status::OK();
}

Status INT64ConvBackpropExtractAndVerifyDimension(
    StringPiece label, const TensorShape& input_shape,
    const TensorShape& filter_shape, const TensorShape& output_shape,
    const gtl::ArraySlice<int64>& dilations, const std::vector<int64>& strides,
    Padding padding, int64 padding_before, int64 padding_after, int spatial_dim,
    int filter_spatial_dim, ConvBackpropSpatialDimension* dim) {
  dim->input_size = input_shape.dim_size(spatial_dim);
  dim->filter_size = filter_shape.dim_size(filter_spatial_dim);
  dim->output_size = output_shape.dim_size(spatial_dim);
  dim->stride = strides[spatial_dim];
  dim->dilation = dilations[spatial_dim];
  int64 out_size = 0;
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerboseV2(
      dim->input_size, dim->filter_size, dim->dilation, dim->stride, padding,
      &out_size, &padding_before, &padding_after));
  if (dim->output_size != out_size) {
    return errors::InvalidArgument(
        label, ": Size of out_backprop doesn't match computed: ", "actual = ",
        dim->output_size, ", computed = ", out_size,
        " spatial_dim: ", spatial_dim, " input: ", dim->input_size,
        " filter: ", dim->filter_size, " output: ", dim->output_size,
        " stride: ", dim->stride, " dilation: ", dim->dilation);
  }

  int64 effective_filter_size = (dim->filter_size - 1) * dim->dilation + 1;
  dim->expanded_output_size = (dim->output_size - 1) * dim->stride + 1;
  const auto padded_out_size = dim->input_size + effective_filter_size - 1;
  dim->pad_before = effective_filter_size - 1 - padding_before;
  dim->pad_after =
      padded_out_size - dim->expanded_output_size - dim->pad_before;
  VLOG(2) << label << ": expanded_out = " << dim->expanded_output_size
          << ", effective_filter_size = " << effective_filter_size
          << ", padded_out = " << padded_out_size
          << ", pad_before = " << dim->pad_before
          << ", pad_after = " << dim->pad_after
          << ", dilation = " << dim->dilation << ", strides = " << dim->stride;
  return Status::OK();
}


template <typename Device, class T>
struct LaunchXsmmBackwardFilter {
  bool operator()(OpKernelContext* context, const Device& d,
                  typename TTypes<T, 4>::ConstTensor input_backward,
                  typename TTypes<T, 4>::Tensor kernel,
                  typename TTypes<T, 4>::ConstTensor output_backward,
                  int input_rows, int input_cols, int row_stride,
                  int col_stride, int pad_h, int pad_w,
                  TensorFormat data_format) const {
    return false;
  }
};


Status INT64ConvBackpropComputeDimensionsV2(
    StringPiece label, int num_spatial_dims, const TensorShape& input_shape,
    const TensorShape& filter_shape, const TensorShape& out_backprop_shape,
    const gtl::ArraySlice<int64>& dilations, const std::vector<int64>& strides,
    Padding padding, absl::Span<const int64> explicit_paddings,
    TensorFormat data_format, ConvBackpropDimensions* dims) {
  // The + 2 in the following line is for the batch and feature dimensions.
  const int num_dims = num_spatial_dims + 2;
  if (input_shape.dims() != num_dims) {
    return errors::InvalidArgument(label, ": input must be ", num_dims,
                                   "-dimensional");
  }
  if (filter_shape.dims() != num_dims) {
    return errors::InvalidArgument(label, ": filter must be ", num_dims,
                                   "-dimensional");
  }
  if (out_backprop_shape.dims() != num_dims) {
    return errors::InvalidArgument(label, ": out_backprop must be ", num_dims,
                                   "-dimensional");
  }
  // 获取batch_size在data_format中的索引
  int batch_dim = GetTensorBatchDimIndex(num_dims, data_format);
  // dims_batch_size 为batch_size值
  dims->batch_size = input_shape.dim_size(batch_dim);
  if (dims->batch_size != out_backprop_shape.dim_size(batch_dim)) {
    return errors::InvalidArgument(
        label, ": input and out_backprop must have the same batch size",
        "input batch: ", dims->batch_size,
        "outbackprop batch: ", out_backprop_shape.dim_size(batch_dim),
        " batch_dim: ", batch_dim);
  }

  int feature_dim = GetTensorFeatureDimIndex(num_dims, data_format);
  dims->in_depth = input_shape.dim_size(feature_dim);
  // The input and output feature dimensions are the second last and last
  // dimensions of the filter Tensor.
  VLOG(2) << "input vs filter_in depth " << dims->in_depth << " "
          << filter_shape.dim_size(num_dims - 2);
  if (filter_shape.dim_size(num_dims - 2) <= 0) {
    return errors ::InvalidArgument(
        label, ": filter depth must be strictly greated than zero");
  }
  if (dims->in_depth % filter_shape.dim_size(num_dims - 2)) {
    return errors::InvalidArgument(
        label, ": input depth must be evenly divisible by filter depth");
  }
  dims->out_depth = filter_shape.dim_size(num_dims - 1);
  if (dims->out_depth != out_backprop_shape.dim_size(feature_dim)) {
    return errors::InvalidArgument(
        label, ": filter and out_backprop must have the same out_depth");
  }
  dims->spatial_dims.resize(num_spatial_dims);
  // 填充数据
  for (int i = 0; i < num_spatial_dims; ++i) {
    int image_dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
    int64 padding_before = -1, padding_after = -1;
    if (padding == EXPLICIT) {
      padding_before = explicit_paddings[2 * image_dim];
      padding_after = explicit_paddings[2 * image_dim + 1];
    }
    TF_RETURN_IF_ERROR(INT64ConvBackpropExtractAndVerifyDimension(
        label, input_shape, filter_shape, out_backprop_shape, dilations,
        strides, padding, padding_before, padding_after, image_dim, i,
        &dims->spatial_dims[i]));
  }
  return Status::OK();
}

template <typename T>
struct LaunchConv2DBackpropFilterOp<CPUDevice, T> {
  void operator()(OpKernelContext* ctx,
                  const Tensor& out_backprop, const Tensor& input,
                  int row_dilation, int col_dilation, int row_stride,
                  int col_stride, const Padding& padding,
                  const std::vector<int64>& explicit_paddings,
                  Tensor* filter_backprop, TensorFormat data_format) {
    std::vector<int64> dilations(4, 1);
    dilations[GetTensorDimIndex(data_format, 'H')] = row_dilation;
    dilations[GetTensorDimIndex(data_format, 'W')] = col_dilation;

    std::vector<int64> strides(4, 1);
    strides[GetTensorDimIndex(data_format, 'H')] = row_stride;
    strides[GetTensorDimIndex(data_format, 'W')] = col_stride;
    TensorShape filter_shape = filter_backprop->shape();

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(
        ctx, INT64ConvBackpropComputeDimensionsV2(
                 "INT64Conv2DBackpropFilter", /*num_spatial_dims=*/2, input.shape(),
                 filter_shape, out_backprop.shape(), dilations, strides,
                 padding, explicit_paddings, data_format, &dims));

    int64 padding_top = -1, padding_bottom = -1;
    int64 padding_left = -1, padding_right = -1;
    if (padding == EXPLICIT) {
      GetExplicitPaddingForDim(explicit_paddings, data_format, 'H',
                               &padding_top, &padding_bottom);
      GetExplicitPaddingForDim(explicit_paddings, data_format, 'W',
                               &padding_left, &padding_right);
    }
    int64 expected_out_rows, expected_out_cols;
    // The function is guaranteed to succeed because we checked the output and
    // padding was valid earlier.
    TF_CHECK_OK(GetWindowedOutputSizeVerboseV2(
        dims.spatial_dims[0].input_size, dims.spatial_dims[0].filter_size,
        row_dilation, row_stride, padding, &expected_out_rows, &padding_top,
        &padding_bottom));
    DCHECK_EQ(dims.spatial_dims[0].output_size, expected_out_rows);
    TF_CHECK_OK(GetWindowedOutputSizeVerboseV2(
        dims.spatial_dims[1].input_size, dims.spatial_dims[1].filter_size,
        col_dilation, col_stride, padding, &expected_out_cols, &padding_left,
        &padding_right));
    DCHECK_EQ(dims.spatial_dims[1].output_size, expected_out_cols);

    const CPUDevice& d = ctx->eigen_device<CPUDevice>();

    // WARNING: Need to swap row/col, padding_top/padding_left, and
    // padding_bottom/padding_right when calling Eigen. Eigen expects tensors
    // in NWHC format, but Tensorflow uses NHWC.

    auto filter_backprop_t = filter_backprop->tensor<T, 4>();
    auto input_t = input.tensor<T, 4>();
    auto out_backprop_t = out_backprop.tensor<T, 4>();

    if (padding != EXPLICIT) {
      // If padding was not explicitly defined, Eigen spatial convolution
      // backward filter will infer correct forward paddings from input tensors.
      filter_backprop_t.device(d) = Eigen::SpatialConvolutionBackwardKernel(
          input_t, out_backprop_t, filter_backprop_t.dimension(1),
          filter_backprop_t.dimension(0), col_stride, row_stride, col_dilation,
          row_dilation);

    } else {
      // Otherwise we have to explicitly pad the input, before passing it to
      // spatial convolution backward filter.
      Eigen::array<std::pair<int, int>, 4> paddings;
      paddings[0] = {0, 0};
      paddings[1] = {padding_top, padding_bottom};
      paddings[2] = {padding_left, padding_right};
      paddings[3] = {0, 0};

      auto padded_t = input_t.pad(paddings, T(0));

      // TODO(ezhulenev): Pass explicit paddings to Eigen spatial backward
      // convolution and do not rely on tensor padding expression.
      filter_backprop_t.device(d) = Eigen::SpatialConvolutionBackwardKernel(
          padded_t, out_backprop_t, filter_backprop_t.dimension(1),
          filter_backprop_t.dimension(0), col_stride, row_stride, col_dilation,
          row_dilation);
    }
  }
};

// Returns in 'im_data' (assumes to be zero-initialized) image patch in storage
// order (height, width, depth), constructed from patches in 'col_data', which
// is required to be in storage order (out_height * out_width, filter_height,
// filter_width, in_depth).  Implementation by Yangqing Jia (jiayq).
template <typename T>
void Col2im(const T* col_data, const int depth, const int height,
            const int width, const int filter_h, const int filter_w,
            const int pad_t, const int pad_l, const int pad_b, const int pad_r,
            const int stride_h, const int stride_w, T* im_data) {
  int height_col = (height + pad_t + pad_b - filter_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;
  int h_pad = -pad_t;
  for (int h = 0; h < height_col; ++h) {
    int w_pad = -pad_l;
    for (int w = 0; w < width_col; ++w) {
      T* im_patch_data = im_data + (h_pad * width + w_pad) * depth;
      for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
        for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            // TODO(andydavis) Vectorize this loop (if compiler does not).
            for (int i = 0; i < depth; ++i) {
              im_patch_data[i] += col_data[i];
            }
          }
          im_patch_data += depth;
          col_data += depth;
        }
        // Jump over remaining number of depth.
        im_patch_data += depth * (width - filter_w);
      }
      w_pad += stride_w;
    }
    h_pad += stride_h;
  }
}

// Computes backprop input using Eigen::SpatialConvolutionBackwardInput on CPU
template <typename Device, typename T>
struct LaunchConv2DBackpropInputOpImpl {
  void operator()(OpKernelContext* ctx,
                  const Tensor& out_backprop, const Tensor& filter,
                  int row_dilation, int col_dilation, int row_stride,
                  int col_stride, const Padding& padding,
                  const std::vector<int64>& explicit_paddings,
                  Tensor* in_backprop, TensorFormat data_format) {
    std::vector<int64> strides(4, 1);
    std::vector<int64> dilations(4, 1);

    auto input_h = GetTensorDimIndex(data_format, 'H');
    auto input_w = GetTensorDimIndex(data_format, 'W');
    strides[input_h] = row_stride;
    strides[input_w] = col_stride;
    dilations[input_h] = row_dilation;
    dilations[input_w] = col_dilation;

    const TensorShape& input_shape = in_backprop->shape();
    const TensorShape& filter_shape = filter.shape();

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(
        ctx, INT64ConvBackpropComputeDimensionsV2(
                 "INT64Conv2DBackpropInput", /*num_spatial_dims=*/2, input_shape,
                 filter_shape, out_backprop.shape(), dilations, strides,
                 padding, explicit_paddings, data_format, &dims));

    int64 padding_top = -1, padding_bottom = -1;
    int64 padding_left = -1, padding_right = -1;
    if (padding == EXPLICIT) {
      GetExplicitPaddingForDim(explicit_paddings, data_format, 'H',
                               &padding_top, &padding_bottom);
      GetExplicitPaddingForDim(explicit_paddings, data_format, 'W',
                               &padding_left, &padding_right);
    }

    int64 expected_out_rows, expected_out_cols;
    // The function is guaranteed to succeed because we checked the output and
    // padding was valid earlier.
    TF_CHECK_OK(GetWindowedOutputSizeVerboseV2(
        dims.spatial_dims[0].input_size, dims.spatial_dims[0].filter_size,
        row_dilation, row_stride, padding, &expected_out_rows, &padding_top,
        &padding_bottom));
    DCHECK_EQ(dims.spatial_dims[0].output_size, expected_out_rows);

    TF_CHECK_OK(GetWindowedOutputSizeVerboseV2(
        dims.spatial_dims[1].input_size, dims.spatial_dims[1].filter_size,
        col_dilation, col_stride, padding, &expected_out_cols, &padding_left,
        &padding_right));
    DCHECK_EQ(dims.spatial_dims[1].output_size, expected_out_cols);


    auto in_backprop_t = in_backprop->tensor<T, 4>();
    auto out_backprop_t = out_backprop.tensor<T, 4>();
    auto filter_t = filter.tensor<T, 4>();

    // WARNING: Need to swap row/col, padding_top/padding_left, and
    // padding_bottom/padding_right when calling Eigen. Eigen expects tensors
    // in NWHC format, but Tensorflow uses NHWC.

    if (padding != EXPLICIT) {
      // If padding was not explicitly defined, Eigen spatial convolution
      // backward input will infer correct forward paddings from input tensors.
      functor::SpatialConvolutionBackwardInputFunc<Device, T>()(
          ctx->eigen_device<Device>(), in_backprop_t, filter_t, out_backprop_t,
          col_stride, row_stride, col_dilation, row_dilation);
    } else {
      functor::SpatialConvolutionBackwardInputWithExplicitPaddingFunc<Device,
                                                                      T>()(
          ctx->eigen_device<Device>(), in_backprop_t, filter_t, out_backprop_t,
          in_backprop_t.dimension(2) + (padding_left + padding_right),
          in_backprop_t.dimension(1) + (padding_top + padding_bottom),
          col_stride, row_stride, col_dilation, row_dilation, padding_top,
          padding_left);
    }
  }
};

#ifdef TENSORFLOW_USE_LIBXSMM_CONVOLUTIONS
template <typename Device, class T>
struct LaunchXsmmBackwardInputConvolution {
  bool operator()(OpKernelContext* context, const Device& d,
                  typename TTypes<T, 4>::Tensor input_backward,
                  typename TTypes<T, 4>::ConstTensor kernel,
                  typename TTypes<T, 4>::ConstTensor output_backward,
                  int input_rows, int input_cols, int row_stride,
                  int col_stride, int pad_h, int pad_w,
                  TensorFormat data_format) const {
    return false;
  }
};
#endif


// Computes backprop input using Eigen::SpatialConvolutionBackwardInput on CPU.
template <typename T>
struct LaunchConv2DBackpropInputOp<CPUDevice, T> {
  void operator()(OpKernelContext* ctx,
                  const Tensor& out_backprop, const Tensor& filter,
                  int row_dilation, int col_dilation, int row_stride,
                  int col_stride, const Padding& padding,
                  const std::vector<int64>& explicit_paddings,
                  Tensor* in_backprop, TensorFormat data_format) {
    LaunchConv2DBackpropInputOpImpl<CPUDevice, T> launcher;
    launcher(ctx, out_backprop, filter,
             row_dilation, col_dilation, row_stride, col_stride, padding,
             explicit_paddings, in_backprop, data_format);
  }
};

template <typename T>
struct Conv2DCustomBackpropInputMatMulFunctor {
  using MatrixMap = Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using ConstMatrixMap = Eigen::Map<
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  void operator()(OpKernelContext* ctx, const T* out_data, const T* filter_data,
                  const int filter_total_size, const int output_image_size,
                  const int dims_out_depth, T* im2col_buf) {
    // Compute gradient into 'im2col_buf'.
    MatrixMap C(im2col_buf, output_image_size, filter_total_size);

    ConstMatrixMap A(out_data, output_image_size, dims_out_depth);
    ConstMatrixMap B(filter_data, filter_total_size, dims_out_depth);

    C.noalias() = A * B.transpose();
  }
};



// 定义op接口
REGISTER_OP("INT64Conv2D")
    .Input("input: int64")
    .Input("filter: int64")
    .Output("output: int64")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn(shape_inference::Conv2DShapeWithExplicitPadding);


// 注册OP
REGISTER_OP("INT64Conv2DBackpropFilter")
    .Input("input: int64")
    .Input("filter_sizes: int64")
    .Input("out_backprop: int64")
    .Output("output: int64")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    });


// 注册op操作
REGISTER_OP("INT64Conv2DBackpropInput")
    .Input("input_sizes: int64")
    .Input("filter: int64")
    .Input("out_backprop: int64")
    .Output("output: int64")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    });



// 定义运算,继承自BinaryOp
template <typename Device, typename T>
class INT64Conv2DOp : public BinaryOp<T> {
 public:
 //被explicit修饰的构造函数，不能发生隐式的类型转换；
  explicit INT64Conv2DOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter = context->input(1);

    // 从参数、输入和滤波器张量推断的卷积维度
    Conv2DDimensions dimensions;
    OP_REQUIRES_OK(context,
                   ComputeINT64Conv2DDimension(params_, input, filter, &dimensions));

    TensorShape out_shape = ShapeFromFormat(
        params_.data_format, dimensions.batch, dimensions.out_rows,
        dimensions.out_cols, dimensions.out_depth);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "Conv2D: in_depth = " << dimensions.in_depth
            << ", patch_depth = " << dimensions.patch_depth
            << ", input_cols = " << dimensions.input_cols
            << ", filter_cols = " << dimensions.filter_cols
            << ", input_rows = " << dimensions.input_rows
            << ", filter_rows = " << dimensions.filter_rows
            << ", stride_rows = " << dimensions.stride_rows
            << ", stride_cols = " << dimensions.stride_cols
            << ", dilation_rows = " << dimensions.dilation_rows
            << ", dilation_cols = " << dimensions.dilation_cols
            << ", out_depth = " << dimensions.out_depth;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

#ifdef TENSORFLOW_USE_LIBXSMM_CONVOLUTIONS
    if (params_.padding != EXPLICIT &&
        LaunchXsmmConvOp<Device, T>::Run(
            context, input, filter, dimensions.batch, dimensions.input_rows,
            dimensions.input_cols, dimensions.in_depth, dimensions.filter_rows,
            dimensions.filter_cols, dimensions.pad_rows_before,
            dimensions.pad_cols_before, dimensions.out_rows,
            dimensions.out_cols, dimensions.out_depth, dimensions.dilation_rows,
            dimensions.dilation_cols, dimensions.stride_rows,
            dimensions.stride_cols, output, params_.data_format)) {
      return;
    }
#endif

    if (params_.padding != EXPLICIT &&
        LaunchDeepConvOp<Device, T>::Run(
            context, input, filter, dimensions.batch, dimensions.input_rows,
            dimensions.input_cols, dimensions.in_depth, dimensions.filter_rows,
            dimensions.filter_cols, dimensions.pad_rows_before,
            dimensions.pad_cols_before, dimensions.out_rows,
            dimensions.out_cols, dimensions.out_depth, dimensions.dilation_rows,
            dimensions.dilation_cols, dimensions.stride_rows,
            dimensions.stride_cols, output, params_.data_format)) {
      return;
    }

    launcher_(context, use_cudnn_, cudnn_use_autotune_, input, filter,
              dimensions.dilation_rows, dimensions.dilation_cols,
              dimensions.stride_rows, dimensions.stride_cols, params_.padding,
              params_.explicit_paddings, output, params_.data_format);
  }

 private:
  Conv2DParameters params_;
  bool use_cudnn_;
  bool cudnn_use_autotune_;

  LaunchConv2DOp<Device, T> launcher_;

  TF_DISALLOW_COPY_AND_ASSIGN(INT64Conv2DOp);
};

// 求参数的导数
template <typename Device, class T>
class INT64Conv2DCustomBackpropFilterOp : public OpKernel {
 public:
  explicit INT64Conv2DCustomBackpropFilterOp(OpKernelConstruction* context)
      : OpKernel(context) {
      // 解析参数
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                errors::InvalidArgument(
                    "Conv2DCustomBackpropFilterOp only supports NHWC."));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[3] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES(context, strides_[1] > 0 && strides_[2] > 0,
                errors::InvalidArgument(
                    "Row and column strides should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("explicit_paddings", &explicit_paddings_));
    OP_REQUIRES_OK(context, CheckValidPadding(padding_, explicit_paddings_,
                                              /*num_dims=*/4, data_format_));
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
    OP_REQUIRES(context, dilations_.size() == 4,
                errors::InvalidArgument("Sliding window dilations field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, (dilations_[0] == 1 && dilations_[3] == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilations in the batch and depth dimensions."));
    if (std::is_same<Device, CPUDevice>::value) {
      // TODO(yangzihao): Add a CPU implementation for dilated convolution.
      OP_REQUIRES(context, (dilations_[1] == 1 && dilations_[2] == 1),
                  errors::InvalidArgument(
                      "Current libxsmm and customized CPU implementations do "
                      "not yet support dilation rates larger than 1."));
      dilations_ = {1, 1, 1, 1};
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0); // 前向传播中，上一层传入卷积层的输入
    const Tensor& filter_sizes = context->input(1); //卷积核形状
    const Tensor& out_backprop = context->input(2); // 反向传播中，上一层传入卷积层的误差
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(filter_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DCustomBackpropFilter: filter_sizes input must be 1-dim, "
            "not ",
            filter_sizes.dims()));
    TensorShape filter_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                filter_sizes.vec<int64>(), &filter_shape));

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(
        context,
        INT64ConvBackpropComputeDimensionsV2(
            "INT64Conv2DCustomBackpropFilter", /*num_spatial_dims=*/2, input.shape(),
            filter_shape, out_backprop.shape(), dilations_, strides_, padding_,
            explicit_paddings_, data_format_, &dims));

    Tensor* filter_backprop;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));

    // If there is nothing to compute, return.
    if (filter_shape.num_elements() == 0) {
      return;
    }

    int64 pad_top, pad_bottom;
    int64 pad_left, pad_right;
    if (padding_ == Padding::EXPLICIT) {
      pad_top = explicit_paddings_[2];
      pad_bottom = explicit_paddings_[3];
      pad_left = explicit_paddings_[4];
      pad_right = explicit_paddings_[5];
    }
    OP_REQUIRES_OK(
        context,
        GetWindowedOutputSizeVerbose(
            dims.spatial_dims[0].input_size, dims.spatial_dims[0].filter_size,
            dims.spatial_dims[0].stride, padding_,
            &dims.spatial_dims[0].output_size, &pad_top, &pad_bottom));
    OP_REQUIRES_OK(
        context,
        GetWindowedOutputSizeVerbose(
            dims.spatial_dims[1].input_size, dims.spatial_dims[1].filter_size,
            dims.spatial_dims[1].stride, padding_,
            &dims.spatial_dims[1].output_size, &pad_left, &pad_right));
#if defined TENSORFLOW_USE_LIBXSMM_CONVOLUTIONS && \
    defined TENSORFLOW_USE_LIBXSMM_BACKWARD_CONVOLUTIONS
    if (pad_left == pad_right && pad_top == pad_bottom) {
      if (LaunchXsmmBackwardFilter<Device, T>()(
              context, context->eigen_device<Device>(), input.tensor<T, 4>(),
              filter_backprop->tensor<T, 4>(), out_backprop.tensor<T, 4>(),
              dims.spatial_dims[0].input_size, dims.spatial_dims[1].input_size,
              static_cast<int>(dims.spatial_dims[0].stride),
              static_cast<int>(dims.spatial_dims[1].stride),
              static_cast<int>(pad_top), static_cast<int>(pad_left),
              data_format_)) {
        return;
      }
    }
#endif

    // The total dimension size of each kernel.
    const int filter_total_size = dims.spatial_dims[0].filter_size *
                                  dims.spatial_dims[1].filter_size *
                                  dims.in_depth;
    OP_REQUIRES(
        context,
        filter_total_size * dims.out_depth == filter_backprop->NumElements(),
        errors::InvalidArgument(
            "filter_size does not have enough elements, requested ",
            filter_total_size * dims.out_depth, ", got ",
            filter_backprop->NumElements()));

    // The output image size is the spatial size of the output.
    const int output_image_size =
        dims.spatial_dims[0].output_size * dims.spatial_dims[1].output_size;

    // Shard 'batch' images into 'shard_size' groups of images to be fed
    // into the parallel matmul. Calculate 'shard_size' by dividing the L3 cache
    // size ('target_working_set_size') by the matmul size of an individual
    // image ('work_unit_size').

    // TODO(andydavis)
    // *) Get L3 cache size from device at runtime (30MB is from ivybridge).
    // *) Consider reducing 'target_working_set_size' if L3 is shared by
    //    other concurrently running tensorflow ops.
    const size_t target_working_set_size = (30LL << 20) / sizeof(T);

    const size_t size_A = output_image_size * filter_total_size;

    const size_t size_B = output_image_size * dims.out_depth;

    const size_t size_C = filter_total_size * dims.out_depth;

    const size_t work_unit_size = size_A + size_B + size_C;

    OP_REQUIRES(
        context, work_unit_size != 0,
        errors::InvalidArgument(
            "Work size for convolution would be 0, which is not acceptable"));

    const size_t shard_size =
        (target_working_set_size + work_unit_size - 1) / work_unit_size;

    Tensor col_buffer;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DataTypeToEnum<T>::value,
                       TensorShape({static_cast<int64>(shard_size),
                                    static_cast<int64>(output_image_size),
                                    static_cast<int64>(filter_total_size)}),
                       &col_buffer));

    // The input offset corresponding to a single input image.
    const int input_offset = dims.spatial_dims[0].input_size *
                             dims.spatial_dims[1].input_size * dims.in_depth;
    // The output offset corresponding to a single output image.
    const int output_offset = dims.spatial_dims[0].output_size *
                              dims.spatial_dims[1].output_size * dims.out_depth;

    const T* input_data = input.template flat<T>().data();
    T* col_buffer_data = col_buffer.template flat<T>().data();
    const T* out_backprop_data = out_backprop.template flat<T>().data();
    T* filter_backprop_data = filter_backprop->template flat<T>().data();

    typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>,
                             Eigen::Unaligned>
        TensorMap;
    typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                             Eigen::Unaligned>
        ConstTensorMap;

    TensorMap C(filter_backprop_data, filter_total_size, dims.out_depth);
    C.setZero();

    // Initialize contraction dims (we need to transpose 'A' below).
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_dims;
    contract_dims[0].first = 0;
    contract_dims[0].second = 0;

    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

    for (int image_id = 0; image_id < dims.batch_size; image_id += shard_size) {
      const int shard_limit =
          std::min(static_cast<int>(shard_size),
                   static_cast<int>(dims.batch_size) - image_id);

      auto shard = [&input_data, &col_buffer_data, &dims, &pad_top, &pad_left,
                    &pad_bottom, &pad_right, &input_offset,
                    &size_A](int64 start, int64 limit) {
        for (int shard_id = start; shard_id < limit; ++shard_id) {
          const T* input_data_shard = input_data + shard_id * input_offset;
          T* col_data_shard = col_buffer_data + shard_id * size_A;

          // When we compute the gradient with respect to the filters, we need
          // to do im2col to allow gemm-type computation.
          Im2col<T>(
              input_data_shard, dims.in_depth, dims.spatial_dims[0].input_size,
              dims.spatial_dims[1].input_size, dims.spatial_dims[0].filter_size,
              dims.spatial_dims[1].filter_size, pad_top, pad_left, pad_bottom,
              pad_right, dims.spatial_dims[0].stride,
              dims.spatial_dims[1].stride, col_data_shard);
        }
      };
      Shard(worker_threads.num_threads, worker_threads.workers, shard_limit,
            size_A, shard);

      ConstTensorMap A(col_buffer_data, output_image_size * shard_limit,
                       filter_total_size);
      ConstTensorMap B(out_backprop_data, output_image_size * shard_limit,
                       dims.out_depth);

      // Gradient with respect to filter.
      C.device(context->eigen_cpu_device()) += A.contract(B, contract_dims);

      input_data += input_offset * shard_limit;
      out_backprop_data += output_offset * shard_limit;
    }
  }

 private:
  std::vector<int64> dilations_;
  std::vector<int64> strides_;
  Padding padding_;
  std::vector<int64> explicit_paddings_;
  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(INT64Conv2DCustomBackpropFilterOp);
};

// 定义运算
template <typename Device, class T>
class INT64Conv2DBackpropInputOp : public OpKernel {
 public:
  explicit INT64Conv2DBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    // 解析数据格式
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));

    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    int stride_n = GetTensorDim(strides_, data_format_, 'N');
    int stride_c = GetTensorDim(strides_, data_format_, 'C');
    int stride_h = GetTensorDim(strides_, data_format_, 'H');
    int stride_w = GetTensorDim(strides_, data_format_, 'W');
    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES(context, stride_h > 0 && stride_w > 0,
                errors::InvalidArgument(
                    "Row and column strides should be larger than 0."));

    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
    OP_REQUIRES(context, dilations_.size() == 4,
                errors::InvalidArgument("Sliding window dilations field must "
                                        "specify 4 dimensions"));
    int dilation_n = GetTensorDim(dilations_, data_format_, 'N');
    int dilation_c = GetTensorDim(dilations_, data_format_, 'C');
    int dilation_h = GetTensorDim(dilations_, data_format_, 'H');
    int dilation_w = GetTensorDim(dilations_, data_format_, 'W');
    OP_REQUIRES(context, (dilation_n == 1 && dilation_c == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilations in the batch and depth dimensions."));
    OP_REQUIRES(
        context, dilation_h > 0 && dilation_w > 0,
        errors::InvalidArgument("Dilated rates should be larger than 0."));

    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("explicit_paddings", &explicit_paddings_));
    OP_REQUIRES_OK(context, CheckValidPadding(padding_, explicit_paddings_,
                                              /*num_dims=*/4, data_format_));

    if (std::is_same<Device, CPUDevice>::value ||
        std::is_same<T, int64>::value) {
      OP_REQUIRES(
          context, data_format_ == FORMAT_NHWC,
          errors::InvalidArgument("INT64Conv2DBackpropInputOp CPU "
                                  "only supports NHWC data format."));

      // TODO(yangzihao): Add a CPU implementation for dilated convolution.
      OP_REQUIRES(
          context, (dilation_h == 1 && dilation_w == 1),
          errors::InvalidArgument(
              "INT64Conv2DBackpropInputOp CPU  not yet support "
              "dilation rates larger than 1."));
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_sizes = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input_sizes.shape()),
        errors::InvalidArgument(
            "INT64Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
            input_sizes.dims()));
    TensorShape input_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                input_sizes.vec<int64>(), &input_shape));

    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

    // If there is nothing to compute, return.
    if (input_shape.num_elements() == 0) {
      return;
    }

    // For now we take the stride from the second and third dimensions only (we
    // do not support striding on the batch or depth dimension).
    const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
    const int stride_cols = GetTensorDim(strides_, data_format_, 'W');
    const int dilation_rows = GetTensorDim(dilations_, data_format_, 'H');
    const int dilation_cols = GetTensorDim(dilations_, data_format_, 'W');

    VLOG(2) << "INT64Conv2DBackpropInput:"
            << " input: " << input_shape.DebugString()
            << " filter:" << filter.shape().DebugString()
            << " out_backprop: " << out_backprop.shape().DebugString()
            << " strides: [" << stride_rows << ", " << stride_cols << "]"
            << " dilations: [" << dilation_rows << ", " << dilation_cols << "]";

    LaunchConv2DBackpropInputOp<Device, T> launch;
    launch(context, out_backprop, filter,
           dilation_rows, dilation_cols, stride_rows, stride_cols, padding_,
           explicit_paddings_, in_backprop, data_format_);
  }

 private:
  std::vector<int64> dilations_;
  std::vector<int64> strides_;
  TensorFormat data_format_;
  Padding padding_;
  std::vector<int64> explicit_paddings_;


  TF_DISALLOW_COPY_AND_ASSIGN(INT64Conv2DBackpropInputOp);
};

template <typename Device, class T>
class INT64Conv2DCustomBackpropInputOp : public OpKernel {
 public:
  explicit INT64Conv2DCustomBackpropInputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                errors::InvalidArgument(
                    "Conv2DCustomBackpropInputOp only supports NHWC."));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(
        context, (strides_[0] == 1 && strides_[3] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES(context, strides_[1] > 0 && strides_[2] > 0,
                errors::InvalidArgument(
                    "Row and column strides should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
    OP_REQUIRES(context, dilations_.size() == 4,
                errors::InvalidArgument("Sliding window dilations field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, (dilations_[0] == 1 && dilations_[3] == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilations in the batch and depth dimensions."));
    // TODO(yangzihao): Add a CPU implementation for dilated convolution.
    OP_REQUIRES(context, (dilations_[1] == 1 && dilations_[2] == 1),
                errors::InvalidArgument(
                    "Current libxsmm and customized CPU implementations do "
                    "not yet support dilation rates larger than 1."));
    OP_REQUIRES_OK(context,
                   context->GetAttr("explicit_paddings", &explicit_paddings_));
    OP_REQUIRES_OK(context, CheckValidPadding(padding_, explicit_paddings_,
                                              /*num_dims=*/4, data_format_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_sizes = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& out_backprop = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
            input_sizes.dims()));
    TensorShape input_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                input_sizes.vec<int64>(), &input_shape));

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(context,
                   INT64ConvBackpropComputeDimensionsV2(
                       "INT64Conv2DCustomBackpropInput", /*num_spatial_dims=*/2,
                       input_shape, filter.shape(), out_backprop.shape(),
                       /*dilations=*/{1, 1, 1, 1}, strides_, padding_,
                       explicit_paddings_, data_format_, &dims));

    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

    // If there is nothing to compute, return.
    if (input_shape.num_elements() == 0) {
      return;
    }

// TODO(ezhulenev): Remove custom kernel and move XSMM support to
// LaunchConv2DBackpropInputOp functor.
#if defined TENSORFLOW_USE_LIBXSMM_CONVOLUTIONS && \
    defined TENSORFLOW_USE_LIBXSMM_BACKWARD_CONVOLUTIONS
    int64 pad_top, pad_bottom;
    int64 pad_left, pad_right;
    OP_REQUIRES_OK(
        context,
        GetWindowedOutputSizeVerbose(
            dims.spatial_dims[0].input_size, dims.spatial_dims[0].filter_size,
            dims.spatial_dims[0].stride, padding_,
            &dims.spatial_dims[0].output_size, &pad_top, &pad_bottom));
    OP_REQUIRES_OK(
        context,
        GetWindowedOutputSizeVerbose(
            dims.spatial_dims[1].input_size, dims.spatial_dims[1].filter_size,
            dims.spatial_dims[1].stride, padding_,
            &dims.spatial_dims[1].output_size, &pad_left, &pad_right));

    if (pad_left == pad_right && pad_top == pad_bottom) {
      if (LaunchXsmmBackwardInputConvolution<Device, T>()(
              context, context->eigen_device<Device>(),
              in_backprop->tensor<T, 4>(), filter.tensor<T, 4>(),
              out_backprop.tensor<T, 4>(), dims.spatial_dims[0].input_size,
              dims.spatial_dims[1].input_size,
              static_cast<int>(dims.spatial_dims[0].stride),
              static_cast<int>(dims.spatial_dims[1].stride),
              static_cast<int>(pad_top), static_cast<int>(pad_left),
              data_format_)) {
        return;
      }
    }
#else
    int64 pad_top, pad_bottom;
    int64 pad_left, pad_right;
#endif
    if (padding_ == Padding::EXPLICIT) {
      pad_top = explicit_paddings_[2];
      pad_bottom = explicit_paddings_[3];
      pad_left = explicit_paddings_[4];
      pad_right = explicit_paddings_[5];
    }
    OP_REQUIRES_OK(
        context,
        GetWindowedOutputSizeVerbose(
            dims.spatial_dims[0].input_size, dims.spatial_dims[0].filter_size,
            dims.spatial_dims[0].stride, padding_,
            &dims.spatial_dims[0].output_size, &pad_top, &pad_bottom));
    OP_REQUIRES_OK(
        context,
        GetWindowedOutputSizeVerbose(
            dims.spatial_dims[1].input_size, dims.spatial_dims[1].filter_size,
            dims.spatial_dims[1].stride, padding_,
            &dims.spatial_dims[1].output_size, &pad_left, &pad_right));

    // The total dimension size of each kernel.
    const int filter_total_size = dims.spatial_dims[0].filter_size *
                                  dims.spatial_dims[1].filter_size *
                                  dims.in_depth;
    // The output image size is the spatial size of the output.
    const int output_image_size =
        dims.spatial_dims[0].output_size * dims.spatial_dims[1].output_size;

    // TODO(andydavis) Get L2/L3 cache sizes from device.
    const size_t l2_cache_size = 256LL << 10;
    const size_t l3_cache_size = 30LL << 20;

    // Use L3 cache size as target working set size.
    const size_t target_working_set_size = l3_cache_size / sizeof(T);

    // Calculate size of matrices involved in MatMul: C = A x B.
    const size_t size_A = output_image_size * dims.out_depth;

    const size_t size_B = filter_total_size * dims.out_depth;

    const size_t size_C = output_image_size * filter_total_size;

    const size_t work_unit_size = size_A + size_B + size_C;

    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

    // Calculate per-thread work unit size.
    const size_t thread_work_unit_size =
        work_unit_size / worker_threads.num_threads;

    // Set minimum per-thread work unit size to size of L2 cache.
    const size_t min_thread_work_unit_size = l2_cache_size / sizeof(T);

    // Use parallel tensor contractions if there is no batching, or if the
    // minimum per-thread work unit size threshold has been exceeded.
    // Otherwise, revert to multiple single-threaded matmul ops running in
    // parallel to keep all threads busy.
    // TODO(andydavis) Explore alternatives to branching the code in this way
    // (i.e. run multiple, parallel tensor contractions in another thread pool).
    const bool use_parallel_contraction =
        dims.batch_size == 1 ||
        thread_work_unit_size >= min_thread_work_unit_size;

    OP_REQUIRES(
        context, work_unit_size > 0,
        errors::InvalidArgument("input, filter_sizes and out_backprop tensors "
                                "must all have at least 1 element"));

    const size_t shard_size =
        use_parallel_contraction
            ? 1
            : (target_working_set_size + work_unit_size - 1) / work_unit_size;

    Tensor col_buffer;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DataTypeToEnum<T>::value,
                       TensorShape({static_cast<int64>(shard_size),
                                    static_cast<int64>(output_image_size),
                                    static_cast<int64>(filter_total_size)}),
                       &col_buffer));

    // The input offset corresponding to a single input image.
    const int input_offset = dims.spatial_dims[0].input_size *
                             dims.spatial_dims[1].input_size * dims.in_depth;
    // The output offset corresponding to a single output image.
    const int output_offset = dims.spatial_dims[0].output_size *
                              dims.spatial_dims[1].output_size * dims.out_depth;

    const T* filter_data = filter.template flat<T>().data();
    T* col_buffer_data = col_buffer.template flat<T>().data();
    const T* out_backprop_data = out_backprop.template flat<T>().data();

    auto in_backprop_flat = in_backprop->template flat<T>();
    T* input_backprop_data = in_backprop_flat.data();
    in_backprop_flat.device(context->eigen_device<Device>()) =
        in_backprop_flat.constant(T(0));

    if (use_parallel_contraction) {
      typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>,
                               Eigen::Unaligned>
          TensorMap;
      typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                               Eigen::Unaligned>
          ConstTensorMap;

      // Initialize contraction dims (we need to transpose 'B' below).
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_dims;
      contract_dims[0].first = 1;
      contract_dims[0].second = 1;

      for (int image_id = 0; image_id < dims.batch_size; ++image_id) {
        // Compute gradient into col_buffer.
        TensorMap C(col_buffer_data, output_image_size, filter_total_size);

        ConstTensorMap A(out_backprop_data + output_offset * image_id,
                         output_image_size, dims.out_depth);
        ConstTensorMap B(filter_data, filter_total_size, dims.out_depth);

        C.device(context->eigen_cpu_device()) = A.contract(B, contract_dims);

        Col2im<T>(
            col_buffer_data, dims.in_depth, dims.spatial_dims[0].input_size,
            dims.spatial_dims[1].input_size, dims.spatial_dims[0].filter_size,
            dims.spatial_dims[1].filter_size, pad_top, pad_left, pad_bottom,
            pad_right, dims.spatial_dims[0].stride, dims.spatial_dims[1].stride,
            input_backprop_data);

        input_backprop_data += input_offset;
      }
    } else {
      for (int image_id = 0; image_id < dims.batch_size;
           image_id += shard_size) {
        const int shard_limit =
            std::min(static_cast<int>(shard_size),
                     static_cast<int>(dims.batch_size) - image_id);

        auto shard = [&context, &dims, &pad_top, &pad_left, &pad_bottom,
                      &pad_right, &output_image_size, &filter_total_size,
                      &input_backprop_data, &col_buffer_data,
                      &out_backprop_data, &filter_data, &input_offset,
                      &output_offset, &size_C](int64 start, int64 limit) {
          for (int shard_id = start; shard_id < limit; ++shard_id) {
            T* im2col_buf = col_buffer_data + shard_id * size_C;
            T* input_data = input_backprop_data + shard_id * input_offset;
            const T* out_data = out_backprop_data + shard_id * output_offset;

            Conv2DCustomBackpropInputMatMulFunctor<T>()(
                context, out_data, filter_data, filter_total_size,
                output_image_size, dims.out_depth, im2col_buf);

            Col2im<T>(im2col_buf, dims.in_depth,
                      dims.spatial_dims[0].input_size,
                      dims.spatial_dims[1].input_size,
                      dims.spatial_dims[0].filter_size,
                      dims.spatial_dims[1].filter_size, pad_top, pad_left,
                      pad_bottom, pad_right, dims.spatial_dims[0].stride,
                      dims.spatial_dims[1].stride, input_data);
          }
        };
        Shard(worker_threads.num_threads, worker_threads.workers, shard_limit,
              work_unit_size, shard);

        input_backprop_data += input_offset * shard_limit;
        out_backprop_data += output_offset * shard_limit;
      }
    }
  }

 private:
  std::vector<int64> dilations_;
  std::vector<int64> strides_;
  Padding padding_;
  std::vector<int64> explicit_paddings_;
  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(INT64Conv2DCustomBackpropInputOp);
};

// 注册到Tensorflow系统中
REGISTER_KERNEL_BUILDER(Name("INT64Conv2D").Device(DEVICE_CPU), INT64Conv2DOp<CPUDevice, int64>);
REGISTER_KERNEL_BUILDER(Name("INT64Conv2DBackpropFilter").Device(DEVICE_CPU), INT64Conv2DCustomBackpropFilterOp<CPUDevice, int64>);
#define DEFAULT_CPU_OP INT64Conv2DCustomBackpropInputOp
// 注册OP
REGISTER_KERNEL_BUILDER(Name("INT64Conv2DBackpropInput").Device(DEVICE_CPU),DEFAULT_CPU_OP<CPUDevice, int64>);
REGISTER_KERNEL_BUILDER(Name("INT64Conv2DBackpropInput").Device(DEVICE_CPU).Label("custom"),INT64Conv2DCustomBackpropInputOp<CPUDevice, int64>);
REGISTER_KERNEL_BUILDER(Name("INT64Conv2DBackpropInput").Device(DEVICE_CPU).Label("eigen_tensor"),INT64Conv2DBackpropInputOp<CPUDevice, int64>);
}





