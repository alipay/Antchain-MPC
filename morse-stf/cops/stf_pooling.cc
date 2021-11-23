//   Ant Group Copyright (c) 2004-2020 All Rights Reserved.
// Copyright 2016 The TensorFlow Authors. All Rights Reserved.

#define EIGEN_USE_THREADS
#include <vector>
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/eigen_pooling.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/pooling_ops_common.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/register_types.h"


namespace tensorflow{

typedef Eigen::ThreadPoolDevice CPUDevice;

// A helper class to manage sizes and shapes for pooling operations.
struct INT64PoolParameters {
  // Updates context->status if there is an invalid input.
 INT64PoolParameters(OpKernelContext* context, const std::vector<int64>& ksize,
                 const std::vector<int64>& stride, Padding padding,
                 TensorFormat data_format, const TensorShape& tensor_in_shape);

  // Returns the shape of the output for "forward" pooling operations.
  TensorShape forward_output_shape();

  int depth;

  int tensor_in_cols;
  int tensor_in_rows;
  int tensor_in_batch;

  int window_rows;
  int window_cols;
  int depth_window;

  int row_stride;
  int col_stride;
  int depth_stride;

  int64 out_height;
  int64 out_width;
  int out_depth;

  int64 pad_rows;
  int64 pad_cols;
  int pad_depth;

  TensorFormat data_format;
};
INT64PoolParameters::INT64PoolParameters(OpKernelContext* context,
                               const std::vector<int64>& ksize,
                               const std::vector<int64>& stride,
                               Padding padding, TensorFormat data_format,
                               const TensorShape& tensor_in_shape) {
  // For maxpooling, tensor_in should have 2 spatial dimensions.
  // Note: the total number of dimensions could be 4 for NHWC, NCHW,
  // or 5 for NCHW_VECT_C.
  OP_REQUIRES(context,
              GetTensorSpatialDims(tensor_in_shape.dims(), data_format) == 2,
              errors::InvalidArgument(
                  "tensor_in_shape must have 2 spatial dimensions. ",
                  tensor_in_shape.dims(), " ", data_format));

  this->data_format = data_format;
  depth = GetTensorDim(tensor_in_shape, data_format, 'C') *
          (data_format == FORMAT_NCHW_VECT_C ? 4 : 1);
  tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, 'W');
  tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, 'H');
  tensor_in_batch = GetTensorDim(tensor_in_shape, data_format, 'N');
  window_rows = GetTensorDim(ksize, data_format, 'H');
  window_cols = GetTensorDim(ksize, data_format, 'W');
  depth_window = GetTensorDim(ksize, data_format, 'C');
  row_stride = GetTensorDim(stride, data_format, 'H');
  col_stride = GetTensorDim(stride, data_format, 'W');
  depth_stride = GetTensorDim(stride, data_format, 'C');

  // We only support 2D pooling across width/height and depthwise
  // pooling, not a combination.
  OP_REQUIRES(context,
              (depth_window == 1 || (window_rows == 1 && window_cols == 1)),
              errors::Unimplemented(
                  "MaxPooling supports exactly one of pooling across depth "
                  "or pooling across width/height."));

  if (depth_window == 1) {
    OP_REQUIRES_OK(
        context, GetWindowedOutputSize(tensor_in_rows, window_rows, row_stride,
                                       padding, &out_height, &pad_rows));
    OP_REQUIRES_OK(
        context, GetWindowedOutputSize(tensor_in_cols, window_cols, col_stride,
                                       padding, &out_width, &pad_cols));
    pad_depth = 0;
    out_depth = depth;
  } else {
    // Our current version of depthwise max pooling does not support
    // any padding, and expects the depth_window to equal the
    // depth_stride (no overlapping).
    OP_REQUIRES(
        context, depth % depth_window == 0,
        errors::Unimplemented("Depthwise max pooling requires the depth "
                              "window to evenly divide the input depth"));
    OP_REQUIRES(
        context, depth_stride == depth_window,
        errors::Unimplemented("Depthwise max pooling requires the depth "
                              "window to equal the depth stride"));

    // The current version of depthwise max is only implemented on CPU.
    OP_REQUIRES(context,
                (DeviceType(static_cast<Device*>(context->device())
                                ->attributes()
                                .device_type()) == DeviceType(DEVICE_CPU)),
                errors::Unimplemented("Depthwise max pooling is currently "
                                      "only implemented for CPU devices."));

    pad_depth = 0;
    out_depth = depth / depth_window;
  }
}

TensorShape INT64PoolParameters::forward_output_shape() {
  if (depth_window == 1) {
    // Spatial pooling
    return ShapeFromFormat(data_format, tensor_in_batch, out_height, out_width,
                           depth);
  } else {
    // Depthwise pooling
    return TensorShape(
        {tensor_in_batch, tensor_in_rows, tensor_in_cols, out_depth});
  }
}

template <typename Device, typename T>
void INT64SpatialAvgPool(OpKernelContext* context, Tensor* output,
                    const Tensor& input, const INT64PoolParameters& params,
                    const Padding& padding) {
  typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      ConstEigenMatrixMap;
  typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      EigenMatrixMap;

  auto in_flat = input.flat<T>();
  auto out_flat = output->flat<T>();

  auto shard = [&params, &in_flat, &out_flat](int64 start, int64 limit) {
    // Calculate indices for this shards chunk of work.
    const int64 input_image_size =
        params.tensor_in_rows * params.tensor_in_cols * params.depth;
    const int64 output_image_size =
        params.out_width * params.out_height * params.depth;
    const int64 shard_batch_size = limit - start;

    ConstEigenMatrixMap in_mat(
        in_flat.data() + start * input_image_size, params.depth,
        params.tensor_in_cols * params.tensor_in_rows * shard_batch_size);
    EigenMatrixMap out_mat(
        out_flat.data() + start * output_image_size, params.depth,
        params.out_width * params.out_height * shard_batch_size);
    Eigen::Matrix<T, Eigen::Dynamic, 1> out_count(out_mat.cols());
    out_count.setZero();

    // Initializes output to zero.
    out_mat.setZero();

    // The following code basically does the following:
    // 1. Flattens the input and output tensors into two dimensional arrays.
    //    tensor_in_as_matrix:
    //      depth by (tensor_in_cols * tensor_in_rows * tensor_in_batch)
    //    output_as_matrix:
    //      depth by (out_width * out_height * tensor_in_batch)
    //
    // 2. Walks through the set of columns in the flattened
    // tensor_in_as_matrix,
    //    and updates the corresponding column(s) in output_as_matrix with the
    //    average value.
    for (int b = 0; b < shard_batch_size; ++b) {
      for (int h = 0; h < params.tensor_in_rows; ++h) {
        for (int w = 0; w < params.tensor_in_cols; ++w) {
          // (h_start, h_end) * (w_start, w_end) is the range that the input
          // vector projects to.
          const int hpad = h + params.pad_rows;
          const int wpad = w + params.pad_cols;
          const int h_start =
              (hpad < params.window_rows)
                  ? 0
                  : (hpad - params.window_rows) / params.row_stride + 1;
          const int h_end =
              std::min<int>(hpad / params.row_stride + 1, params.out_height);
          const int w_start =
              (wpad < params.window_cols)
                  ? 0
                  : (wpad - params.window_cols) / params.col_stride + 1;
          const int w_end =
              std::min<int>(wpad / params.col_stride + 1, params.out_width);
          const int in_offset =
              (b * params.tensor_in_rows + h) * params.tensor_in_cols + w;
          Eigen::DSizes<Eigen::DenseIndex, 2> in_indices(0, in_offset);
          for (int ph = h_start; ph < h_end; ++ph) {
            for (int pw = w_start; pw < w_end; ++pw) {
              const int out_offset =
                  (b * params.out_height + ph) * params.out_width + pw;
              out_mat.col(out_offset) += in_mat.col(in_offset);
              out_count(out_offset) += T(1);
            }
          }
        }
      }
    }

    DCHECK_GT(out_count.minCoeff(), T(0));
    out_mat.array().rowwise() /= out_count.transpose().array();
  };

  const int64 work_unit_size =
      params.tensor_in_rows * params.tensor_in_cols * params.depth;
  // NOTE: Constants in calculation below were estimated based on benchmarking.
  // Nanoseconds/work_unit for benchmarks ranged from 0.01 to 0.001, and
  // so the factor 0.01 (i.e. 1/100) with a max of 10000, was chosen to limit
  // the work unit cost to an operating range in which it empirically performed
  // best.
  const int64 work_unit_cost = std::max(int64{10000}, work_unit_size / 100);
  const DeviceBase::CpuWorkerThreads& worker_threads =
      *(context->device()->tensorflow_cpu_worker_threads());
  Shard(worker_threads.num_threads, worker_threads.workers,
        params.tensor_in_batch, work_unit_cost, shard);
}

// 定义op接口
REGISTER_OP("INT64AvgPool")
    .Input("value: int64")
    .Output("output: int64")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .SetShapeFn(shape_inference::AvgPoolShape);

//定义AvgPool求梯度的接口
REGISTER_OP("SumPoolGrad")
    .Input("orig_input_shape: int64")
    .Input("grad: int64")
    .Output("output: int64")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    });

// 定义运算
template <typename Device, typename T>
class INT64AvgPoolingOp : public UnaryOp<T> {
 public:
  explicit INT64AvgPoolingOp(OpKernelConstruction* context) : UnaryOp<T>(context) {
    string data_format;
    // check data_format
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(
        context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument("Default AvgPoolingOp only supports NHWC ",
                                "on device type ",
                                DeviceTypeString(context->device_type())));
    //check kernel size
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    // check strides
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    // check padding
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    INT64PoolParameters params{context,  ksize_,       stride_,
                          padding_, data_format_, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }
    OP_REQUIRES(context, params.depth_window == 1,
                errors::Unimplemented("Non-spatial pooling is not "
                                      "yet supported. Volunteers? :)"));

    // For avgpooling, tensor_in should have 4 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, params.forward_output_shape(), &output));

    // compute avg
    INT64SpatialAvgPool<Device, T>(context, output, tensor_in, params, padding_);
  }

 private:
  std::vector<int64> ksize_;
  std::vector<int64> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

// The operation to compute AvgPool gradients.
// It takes two inputs:
//   - The original input tensor shape
//   - Backprop tensor for output
// It produces one output: backprop tensor for input.
template <typename Device, class T>
class SumPoolingGradOp : public OpKernel {
 public:
  explicit SumPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    // check data_format
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(
        context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument("Default AvgPoolingGradOp only supports NHWC ",
                                "on device type ",
                                DeviceTypeString(context->device_type())));
    // check kernel size
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    // check strides
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    // check padding
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in_shape = context->input(0);
    const Tensor& out_backprop = context->input(1);

    // For avg_pooling, tensor_in_shape should have 1 dimension, and 4 elements.
    OP_REQUIRES(
        context,
        tensor_in_shape.dims() == 1 && tensor_in_shape.NumElements() == 4,
        errors::InvalidArgument("out_backprop must be 1-dimensional and 4 "
                                "elements"));
    // For avg_pooling, out_backprop should have 4 dimensions.
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));
    // 解析out_backprop 参数
    const int64 out_backprop_batch = out_backprop.dim_size(0);
    const int64 out_backprop_rows = out_backprop.dim_size(1);
    const int64 out_backprop_cols = out_backprop.dim_size(2);
    const int64 out_backprop_depth = out_backprop.dim_size(3);
    TensorShape output_shape;
    //tensor_in_shape.NumElements() =4
    auto shape_vec = tensor_in_shape.vec<int64>();
    for (int64 i = 0; i < tensor_in_shape.NumElements(); ++i) {
      output_shape.AddDim(shape_vec(i));
    }
    // 确定row 和col
    const int64 in_rows = output_shape.dim_size(1);
    const int64 in_cols = output_shape.dim_size(2);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    output->flat<T>().setZero();
    if (output_shape.num_elements() == 0) {
      return;
    }
    // 解析ksize和stride参数
    const int window_rows = ksize_[1];
    const int window_cols = ksize_[2];
    const int depth_window = ksize_[3];
    // row/col方向上的移动步长
    const int row_stride = stride_[1];
    const int col_stride = stride_[2];
    // We (will) use different code for spatial pooling and
    // non-spatial pooling.
    //
    // Spatial pooling is when depth_window = 1
    OP_REQUIRES(context, depth_window == 1,
                errors::Unimplemented("Non-spatial pooling is not "
                                      "yet supported. Volunteers? :)"));

    int64 out_height, out_width, pad_rows, pad_cols;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(in_rows, window_rows, row_stride,
                                         padding_, &out_height, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(in_cols, window_cols, col_stride,
                                         padding_, &out_width, &pad_cols));

    const T* out_backprop_ptr = out_backprop.flat<T>().data();
    T* input_backprop_ptr = output->flat<T>().data();

    auto shard = [context, out_backprop_ptr, input_backprop_ptr,
                  out_backprop_rows, out_backprop_cols, out_backprop_depth,
                  in_rows, in_cols, window_rows, window_cols, row_stride,
                  col_stride, pad_rows, pad_cols](int64 start, int64 limit) {
      for (int64 b = start; b < limit; ++b) {
        for (int64 r = 0; r < out_backprop_rows; ++r) {
          // Calculates row broadcast size.  For SAME padding, current
          // index could be in the padding area, and r*row_stride +
          // window_rows could be beyond the input tensor's boundary. In
          // such cases, change the starting index and reduce the
          // broadcast size.
          int rindex, rsize;
          OP_REQUIRES_OK(context,
                         GetBroadcastSize(r, in_rows, window_rows, row_stride,
                                          pad_rows, &rindex, &rsize));
          for (int64 c = 0; c < out_backprop_cols; ++c) {
            // Calculates col broadcast size.  For SAME padding, current
            // index could be in the padding area, and c*col_stride +
            // window_cols could be beyond the input tensor's boundary. In
            // such cases, change the starting index and reduce the
            // broadcast size.
            int cindex, csize;
            OP_REQUIRES_OK(context,
                           GetBroadcastSize(c, in_cols, window_cols, col_stride,
                                            pad_cols, &cindex, &csize));

            // std::cout << "cindex: " << cindex << " csize: " << csize << std::endl;
            // 涉及到除法，这里需要将类型指定为float
//            float divide_coeff(1.0 / (rsize * csize));
//            std::cout << "raw divide_coeff" << divide_coeff << std::endl;
            int divide_coeff=1;
//            std::cout << "int divide_coeff" << divide_coeff << std::endl;
//            std::cout << divide_coeff << std::endl;
            int64 output_index =
                (b * out_backprop_rows + r) * out_backprop_cols + c;
            for (int64 r_dst = rindex; r_dst < rindex + rsize; ++r_dst) {
              for (int64 c_dst = cindex; c_dst < cindex + csize; ++c_dst) {
                int64 input_index = (b * in_rows + r_dst) * in_cols + c_dst;
                const T* output_offset =
                    out_backprop_ptr + output_index * out_backprop_depth;
                T* input_offset =
                    input_backprop_ptr + input_index * out_backprop_depth;
                for (int64 d = 0; d < out_backprop_depth; ++d) {
                  *input_offset += *output_offset * divide_coeff;
                  ++output_offset;
                  ++input_offset;
                }
              }
            }
          }
        }
      }
    };

    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());
    const int64 shard_cost =
        window_rows * window_cols * depth_window * in_rows * in_rows * in_cols;
    Shard(worker_threads.num_threads, worker_threads.workers,
          out_backprop_batch, shard_cost, shard);
  }

 private:
  std::vector<int64> ksize_;
  std::vector<int64> stride_;
  Padding padding_;
  TensorFormat data_format_;
};
// 注册op
REGISTER_KERNEL_BUILDER(Name("INT64AvgPool").Device(DEVICE_CPU), INT64AvgPoolingOp<CPUDevice, int64>);
// 注册反向传播op
REGISTER_KERNEL_BUILDER(Name("SumPoolGrad").Device(DEVICE_CPU),SumPoolingGradOp<CPUDevice, int64>);

}