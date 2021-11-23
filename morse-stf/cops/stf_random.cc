//   Ant Group Copyright (c) 2004-2020 All Rights Reserved.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "./inc/rbg.h"
using namespace tensorflow;

#define SEED_LEN 16
#define CTX_LEN 784





REGISTER_OP("Rint64")
    .Input("shape: int32")
    .Input("step: int32")   //使用时把一个Variable赋值给这个step, 然后不用更新这个variable本函数也会每次重新执行
    .Output("z: int64")
    .SetShapeFn(
    [](::tensorflow::shape_inference::InferenceContext* c)
     {
      shape_inference::ShapeHandle _shape;
      c->MakeShapeFromShapeTensor(0, &_shape);
      c->set_output(0, _shape);    //set the shape of output 0
      return tensorflow::Status::OK();
    }
    );

REGISTER_OP("GetSeed")
    //.Attr("shape: list(int)")
    .Input("id: int32")
    .Output("seed: int64")    // of shape [SEED_LEN]
    ;

REGISTER_OP("Rint64FromSeed")
    .Input("shape: int32")
    .Input("seed: int64")    //of shape [SEED_LEN]
    .Output("z: int64")    // of shape [CTX_LEN]
    .SetShapeFn(
    [](::tensorflow::shape_inference::InferenceContext* c)
     {
      shape_inference::ShapeHandle _shape;
      c->MakeShapeFromShapeTensor(0, &_shape);
      c->set_output(0, _shape);    //set the shape of output 0
      return tensorflow::Status::OK();
    })
    ;



#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;

class Rint64OP : public OpKernel {
   private:
  long long seed[16];
  int seednum=16;
  unsigned char ctx[784];
 public:
  explicit Rint64OP(OpKernelConstruction* context) : OpKernel(context) {
  rbg_getseed(this->seed, this->seednum);
  rbg_seeded_instantiate(this->ctx, this->seed);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& shape = context->input(0);
    auto _shape = shape.flat<int32>();

    int len=1;
    for (int i=0;i<shape.dim_size(0); i++)
        len *=  _shape(i);


    tensorflow::TensorShape output_shape;
    tensorflow::TensorShapeUtils::MakeShape((const int *) shape.data(), shape.dim_size(0), &output_shape);


        // Create an output tensor
    Tensor* output_tensor = NULL;
//    int _shape[this->shape.size()];
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));
    //rbg_nonseeded((long long *) output_tensor->data(), len);
    rbg_seeded_random( this->ctx, (long long *) output_tensor->data(), len);

  }
};



class GetSeedOP : public OpKernel {
 public:
  explicit GetSeedOP(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {

    int shape[] = {SEED_LEN};
    tensorflow::TensorShape output_shape;
    tensorflow::TensorShapeUtils::MakeShape((const int *) shape, 1, &output_shape);
    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));
    rbg_getseed((long long *) output_tensor->data(), SEED_LEN);


  }
};




class Rint64FromSeedOP : public OpKernel {
   private:
  unsigned char ctx[CTX_LEN];
  int init_flag=1;
 public:
  explicit Rint64FromSeedOP(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& shape = context->input(0);
    const Tensor& seed = context->input(1);
    auto _shape = shape.flat<int32>();
    auto _seed = seed.flat<int64>();

    int len=1;
    for (int i=0;i<shape.dim_size(0); i++)
        len *=  _shape(i);

    tensorflow::TensorShape output_shape;
    tensorflow::TensorShapeUtils::MakeShape((const int *) shape.data(), shape.dim_size(0), &output_shape);



    Tensor* output_tensor = NULL;

    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));

    if (this->init_flag==1)
    {
        rbg_seeded_instantiate((unsigned char *) this->ctx,  (long long *) _seed.data());
        this->init_flag=0;
    }

    rbg_seeded_random( (unsigned char *) this->ctx, (long long *) output_tensor->data(), len);


  }
};



REGISTER_KERNEL_BUILDER(Name("Rint64").Device(DEVICE_CPU), Rint64OP);
REGISTER_KERNEL_BUILDER(Name("GetSeed").Device(DEVICE_CPU), GetSeedOP);
REGISTER_KERNEL_BUILDER(Name("Rint64FromSeed").Device(DEVICE_CPU), Rint64FromSeedOP);

