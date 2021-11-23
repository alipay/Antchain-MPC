//   Ant Group Copyright (c) 2004-2020 All Rights Reserved.


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "./inc/matrix_mul_vector.h"


using namespace tensorflow;

//#define PlainModuleBits 64
//#define PolyModule 4096
//#define GaloisKeySize 22
//#define CoefModuleBits 183 //6 * 20 + 3 * 21;
//#define CrypLen 93696 //PolyModule * CoefModuleBits / 8;

//REGISTER_OP("GenKey")
//    .Input("x: int64")
//    .Output("y: int64")
//    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
//      c->set_output(0, c->input(0));
//      //c->set_output(0, c->input(1));
//      return tensorflow::Status::OK();
//    });
REGISTER_OP("GenKey")
    .Input("seed: int32")
    .Output("sk: uint8")
    .Output("pk: uint8")
    .Output("gk: uint8")
    ;


REGISTER_OP("Enc")
    .Input("pk: uint8")
    .Input("plain: int64")
    .Output("cipher: uint8")
    ;

REGISTER_OP("Dec")
    .Input("sk: uint8")
    .Input("plain_size: int64")
    .Input("cipher: uint8")
    .Output("plain: int64")
    ;

REGISTER_OP("MatMulVec")
    .Input("pk: uint8")
    .Input("gk: uint8")
    .Input("mat: int64")
    .Input("vec_in: uint8")
    .Output("vec_out: uint8")
    ;

REGISTER_OP("MatMulVecToShare")
    .Input("pk: uint8")
    .Input("gk: uint8")
    .Input("mat: int64")
    .Input("vec_in: uint8")
    .Output("share_vec_out: int64")
    .Output("vec_out: uint8")
    ;

REGISTER_OP("VecMulVec")
    .Input("pk: uint8")
    .Input("vec_plain: int64")
    .Input("vec_cipher: uint8")
    .Output("vec_out: uint8")
    ;

REGISTER_OP("CipherToShare")
    .Input("share_size: int64")
    .Input("pk: uint8")
    .Input("cipher_in: uint8")
    .Output("cipher_out: uint8")
    .Output("share_out: int64")
    ;



class GenKeyOP : public OpKernel {
private:
  int init_flag=1;
  std::vector<uint8_t> v_sk;
  std::vector<uint8_t> v_pk;
  std::vector<uint8_t> v_gk;
  tensorflow::TensorShape sk_shape;
  tensorflow::TensorShape pk_shape;
  tensorflow::TensorShape gk_shape;
 public:
  explicit GenKeyOP(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    //const Tensor& shape = context->input(0);


    if (this->init_flag == 1)
    {
        this->init_flag=0;
        ////std::cout<< "v_sk=" << v_sk <<std::endl;
        morse::mv_gen_key(v_sk, v_pk, v_gk);

        //std::cout<<"v_sk size="<<v_sk.size()<<"byte"<<std::endl;
        //std::cout<<"v_pk size="<<v_pk.size()<<"byte"<<std::endl;
        //std::cout<<"v_gk size="<<v_gk.size()<<"byte"<<std::endl;

        int _sk_shape[] = {(int) v_sk.size()};
        int _pk_shape[] = {(int) v_pk.size()};
        int _gk_shape[] = {(int) v_gk.size()};

        tensorflow::TensorShapeUtils::MakeShape((const int *) _sk_shape, 1, &(sk_shape));
        tensorflow::TensorShapeUtils::MakeShape((const int *) _pk_shape, 1, &(pk_shape));
        tensorflow::TensorShapeUtils::MakeShape((const int *) _gk_shape, 1, &(gk_shape));

    }

        // Create an output tensor
        Tensor* sk = NULL;
        Tensor* pk = NULL;
        Tensor* gk = NULL;
//        context->allocate_output(0, this->sk_shape, &sk);
//        context->allocate_output(1, this->pk_shape, &pk);
//        auto Status = context->allocate_output(2, this->gk_shape, &gk);

        //OP_REQUIRES_OK(context, Status);
        OP_REQUIRES_OK(context, context->allocate_output(0, sk_shape,
                                                         &sk));
        OP_REQUIRES_OK(context, context->allocate_output(1, pk_shape,
                                                         &pk));
        OP_REQUIRES_OK(context, context->allocate_output(2, gk_shape,
                                                         &gk));


        auto skdata = sk->data();
        auto pkdata = pk->data();
        auto gkdata = gk->data();

        memcpy(skdata, v_sk.data(), v_sk.size());
        memcpy(pkdata, v_pk.data(), v_pk.size());
        memcpy(gkdata, v_gk.data(), v_gk.size());



  }
};




class EncOP : public OpKernel {
 public:
  explicit EncOP(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& pk = context->input(0);
    const Tensor& plain = context->input(1);

    int pk_size = pk.shape().dim_size(0);
    int plain_size = plain.shape().dim_size(0);


    std::vector<uint8_t> v_pk(pk_size);
    std::vector<uint64_t> v_plain(plain_size);
    std::vector<uint8_t> v_cipher;

    memcpy(v_pk.data(), pk.data(), pk_size);
    memcpy(v_plain.data(), plain.data(), plain_size*sizeof(uint64));

    morse::mv_encrypt_vector(v_pk, v_plain,
                      v_cipher);


    ////std::cout<<"v_pk size="<<v_pk.size()<<"byte"<<std::endl;
    ////std::cout<<"v_plain size="<<v_plain.size()<<std::endl;
    ////std::cout<<"v_cipher size="<<v_cipher.size()<<"byte"<<std::endl;

    int _cipher_shape[] = {(int) v_cipher.size()};

    tensorflow::TensorShape cipher_shape;
    tensorflow::TensorShapeUtils::MakeShape((const int *) _cipher_shape, 1, &cipher_shape);

    // Create an output tensor
    Tensor* cipher = NULL;

    OP_REQUIRES_OK(context, context->allocate_output(0, cipher_shape,
                                                     &cipher));
    memcpy(cipher->data(), v_cipher.data(), v_cipher.size());

  }
};

class DecOP : public OpKernel {
 public:
  explicit DecOP(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& sk = context->input(0);
    const Tensor& plain_size = context->input(1);
    const Tensor& cipher = context->input(2);

    int sk_size = sk.shape().dim_size(0);
    int cipher_size = cipher.shape().dim_size(0);
    auto _plain_size = plain_size.flat<int64>()(0);
    //std::cout<<"line 148"<<std::endl;

    std::vector<uint8_t> v_sk(sk_size);
    std::vector<uint8_t> v_cipher(cipher_size);
    std::vector<uint64_t> v_plain(_plain_size);


    memcpy(v_sk.data(), sk.data(), sk_size);
    memcpy(v_cipher.data(), cipher.data(), cipher_size);

    //std::cout<<"v_sk size="<<v_sk.size()<<"byte"<<std::endl;
    //std::cout<<"v_cipher size="<<v_cipher.size()<<"byte"<<std::endl;
    morse::mv_decrypt_vector(v_sk, v_cipher,
                  (size_t) _plain_size, v_plain);
    //std::cout<<"v_plain size="<<v_plain.size()<<std::endl;

    ////std::cout<<v_plain[0]<<v_plain[1]<<v_plain[2]<<v_plain[3]<<v_plain[4]<<std::endl;


    int _plain_shape[] = {(int) v_plain.size()};

    tensorflow::TensorShape plain_shape;
    tensorflow::TensorShapeUtils::MakeShape((const int *) _plain_shape, 1, &plain_shape);

    // Create an output tensor
    Tensor* plain = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, plain_shape,
                                                     &plain));

    memcpy(plain->data(), v_plain.data(), v_plain.size()*sizeof(int64));


  }
};

class MatMulVecOP : public OpKernel {
 public:
  explicit MatMulVecOP(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {
    const Tensor& pk = context->input(0);
    const Tensor& gk = context->input(1);
    const Tensor& mat = context->input(2);
    const Tensor& vec_in = context->input(3);

    int pk_size = pk.shape().dim_size(0);
    int gk_size = gk.shape().dim_size(0);
    int row_num = mat.shape().dim_size(0);
    int col_num = mat.shape().dim_size(1);
    int vec_in_size = vec_in.shape().dim_size(0);
    //int vec_in_size = vec_in.shape().dim_size(0);
    //std::cout<<"row_num="<<row_num<<std::endl;
    //std::cout<<"col_num="<<col_num<<std::endl;



    std::vector<uint8_t> v_pk(pk_size);
    std::vector<uint8_t> v_gk(gk_size);
    std::vector<uint64_t> v_mat(row_num*col_num);
    std::vector<std::vector<uint64_t>> vv_mat(row_num, std::vector<uint64_t>(col_num));
    std::vector<uint8_t> v_vec_in(vec_in_size);
    std::vector<uint8_t> v_vec_out;

    memcpy(v_pk.data(), pk.data(), pk_size);
    memcpy(v_gk.data(), gk.data(), gk_size);
    memcpy(v_mat.data(), mat.data(), row_num*col_num*sizeof(uint64));

    //std::cout<<"v_mat size="<<v_mat.size()<<std::endl;

    for (int i=0; i<row_num; i++)
    {
        memcpy(vv_mat[i].data(), & (v_mat[i*col_num]), col_num*sizeof(uint64));
        //std::cout<<"i="<<i<<", vv_mat[i] size="<<vv_mat[i].size()<<std::endl;
    }

    memcpy(v_vec_in.data(), vec_in.data(), vec_in_size);

    //morse::mv_encrypt_vector(v_pk, v_plain, v_cipher);
    //std::cout<<"v_pk size="<<v_pk.size()<<"byte"<<std::endl;
    //std::cout<<"v_gk size="<<v_gk.size()<<"byte"<<std::endl;
    //std::cout<<"vv_mat size="<<vv_mat.size()<<","<<vv_mat[0].size()<<std::endl;
    //std::cout<<"v_vec_in size="<<v_vec_in.size()<<"byte"<<std::endl;


    auto status = morse::mv_matrix_mul_vector(v_pk, v_gk, vv_mat, v_vec_in, v_vec_out);
    //std::cout<<status.error_message()<<std::endl;

    //std::cout<<"v_pk size="<<v_pk.size()<<"byte"<<std::endl;
    //std::cout<<"v_vec_out size="<<v_vec_out.size()<<"byte"<<std::endl;

    int _out_shape[] = {(int) v_vec_out.size()};

    tensorflow::TensorShape out_shape;
    tensorflow::TensorShapeUtils::MakeShape((const int *) _out_shape, 1, &out_shape);

    // Create an output tensor
    Tensor* vec_out = NULL;

    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                     &vec_out));
    memcpy(vec_out->data(), v_vec_out.data(), v_vec_out.size());

  }
};


class MatMulVecToShareOP : public OpKernel {
 public:
  explicit MatMulVecToShareOP(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& pk = context->input(0);
    const Tensor& gk = context->input(1);
    const Tensor& mat = context->input(2);
    const Tensor& vec_in = context->input(3);

    int pk_size = pk.shape().dim_size(0);
    int gk_size = gk.shape().dim_size(0);
    int row_num = mat.shape().dim_size(0);
    int col_num = mat.shape().dim_size(1);
    int vec_in_size = vec_in.shape().dim_size(0);
    //int vec_in_size = vec_in.shape().dim_size(0);
    //std::cout<<"row_num="<<row_num<<std::endl;
    //std::cout<<"col_num="<<col_num<<std::endl;



    std::vector<uint8_t> v_pk(pk_size);
    std::vector<uint8_t> v_gk(gk_size);
    std::vector<uint64_t> v_mat(row_num*col_num);
    std::vector<std::vector<uint64_t>> vv_mat(row_num, std::vector<uint64_t>(col_num));
    std::vector<uint8_t> v_vec_in(vec_in_size);
    std::vector<uint64_t> v_share_vec_out;
    std::vector<uint8_t> v_vec_out;

    memcpy(v_pk.data(), pk.data(), pk_size);
    memcpy(v_gk.data(), gk.data(), gk_size);
    memcpy(v_mat.data(), mat.data(), row_num*col_num*sizeof(uint64));

    for (int i=0; i<row_num; i++)
    {
        memcpy(vv_mat[i].data(), & (v_mat[i*col_num]), col_num*sizeof(uint64));
    }

    memcpy(v_vec_in.data(), vec_in.data(), vec_in_size);

    //morse::mv_encrypt_vector(v_pk, v_plain, v_cipher);
    //std::cout<<"v_pk size="<<v_pk.size()<<"byte"<<std::endl;
    //std::cout<<"v_gk size="<<v_gk.size()<<"byte"<<std::endl;
    //std::cout<<"vv_mat size="<<vv_mat.size()<<","<<vv_mat[0].size()<<std::endl;
    //std::cout<<"v_vec_in size="<<v_vec_in.size()<<"byte"<<std::endl;


    auto status = morse::mv_matrix_mul_vector_to_share(v_pk, v_gk, vv_mat, v_vec_in, v_vec_out, v_share_vec_out);
    //std::cout<<status.error_message()<<std::endl;
// ---------------------------------------------- koko -----------------

    ////std::cout<<"v_pk size="<<v_pk.size()<<"byte"<<std::endl;
    ////std::cout<<"v_plain size="<<v_plain.size()<<std::endl;
    ////std::cout<<"v_cipher size="<<v_cipher.size()<<"byte"<<std::endl;
    ////std::cout<<"v_vec_out size="<<v_vec_out.size()<<"byte"<<std::endl;
    ////std::cout<<"v_share_vec_out="<<v_share_vec_out.size()<<std::endl;

    int _out_shape[] = {(int) v_vec_out.size()};
    int _share_out_shape[] = {(int) v_share_vec_out.size()};

    tensorflow::TensorShape out_shape;
    tensorflow::TensorShapeUtils::MakeShape((const int *) _out_shape, 1, &out_shape);

    tensorflow::TensorShape share_out_shape;
    tensorflow::TensorShapeUtils::MakeShape((const int *) _share_out_shape, 1, &share_out_shape);

    // Create an output tensor
    Tensor* vec_out = NULL;
    Tensor* share_vec_out = NULL;

    OP_REQUIRES_OK(context, context->allocate_output(0, share_out_shape,
                                                     &share_vec_out));
    OP_REQUIRES_OK(context, context->allocate_output(1, out_shape,
                                                     &vec_out));


    //////std::cout<<"line 371:"<<"v_vec_out.size()="<<v_vec_out.size()<<std::endl;

    memcpy(vec_out->data(), v_vec_out.data(), v_vec_out.size());

    ////std::cout<<"line 375:"<<"v_share_vec_out.size()="<<v_share_vec_out.size()<<std::endl;

    memcpy(share_vec_out->data(), v_share_vec_out.data(), v_share_vec_out.size()*sizeof(int64));

    ////std::cout<<"line 379"<<std::endl;

  }
};




class VecMulVecOP : public OpKernel {
 public:
  explicit VecMulVecOP(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {
    const Tensor& pk = context->input(0);
    const Tensor& vec_plain = context->input(1);
    const Tensor& vec_cipher = context->input(2);

    int pk_size = pk.shape().dim_size(0);
    int vec_plain_size = vec_plain.shape().dim_size(0);
    int vec_cipher_size = vec_cipher.shape().dim_size(0);




    std::vector<uint8_t> v_pk(pk_size);
    std::vector<uint64_t> v_vec_plain(vec_plain_size);
    std::vector<uint8_t> v_vec_cipher(vec_cipher_size);
    std::vector<uint8_t> v_vec_out;

    memcpy(v_pk.data(), pk.data(), pk_size);
    memcpy(v_vec_plain.data(), vec_plain.data(), vec_plain_size*sizeof(uint64));
    memcpy(v_vec_cipher.data(), vec_cipher.data(), vec_cipher_size);


    auto status = morse::mv_vector_mul_vector(v_pk, v_vec_plain, v_vec_cipher, v_vec_out);


    int _out_shape[] = {(int) v_vec_out.size()};

    tensorflow::TensorShape out_shape;
    tensorflow::TensorShapeUtils::MakeShape((const int *) _out_shape, 1, &out_shape);

    // Create an output tensor
    Tensor* vec_out = NULL;

    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                     &vec_out));
    memcpy(vec_out->data(), v_vec_out.data(), v_vec_out.size());

  }
};


class CipherToShareOP : public OpKernel {
 public:
  explicit CipherToShareOP(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& share_size = context->input(0);
    const Tensor& pk = context->input(1);
    const Tensor& cipher_in = context->input(2);

    int pk_size = pk.shape().dim_size(0);
    int cipher_in_size = cipher_in.shape().dim_size(0);
    auto _share_size = share_size.flat<int64>()(0);
    //std::cout<<"line 148"<<std::endl;

    std::vector<uint8_t> v_pk(pk_size);
    std::vector<uint8_t> v_cipher_in(cipher_in_size);
    std::vector<uint8_t> v_cipher_out;
    std::vector<uint64_t> v_share_out;


    memcpy(v_pk.data(), pk.data(), pk_size);
    memcpy(v_cipher_in.data(), cipher_in.data(), cipher_in_size);


    auto status = morse::mv_cipher_to_share((size_t) _share_size, v_pk,
                              v_cipher_in, v_cipher_out, v_share_out);

    //std::cout<<"status="<<status.ToString()<<std::endl;
    //std::cout<<"size of v_cipher_out="<<v_cipher_out.size()<<std::endl;
    //std::cout<<"size of v_share_out="<<v_share_out.size()<<std::endl;

    int _cipher_out_shape[] = {(int) v_cipher_out.size()};
    int _share_out_shape[] = {(int) v_share_out.size()};


    tensorflow::TensorShape cipher_out_shape;
    tensorflow::TensorShapeUtils::MakeShape((const int *) _cipher_out_shape, 1, &cipher_out_shape);

    tensorflow::TensorShape share_out_shape;
    tensorflow::TensorShapeUtils::MakeShape((const int *) _share_out_shape, 1, &share_out_shape);

    // Create an output tensor
    Tensor* cipher_out = NULL;
    Tensor* share_out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, cipher_out_shape,
                                                     &cipher_out));
    OP_REQUIRES_OK(context, context->allocate_output(1, share_out_shape,
                                                     &share_out));

    memcpy(cipher_out->data(), v_cipher_out.data(), v_cipher_out.size());
    memcpy(share_out->data(), v_share_out.data(), v_share_out.size()*sizeof(int64));


  }
};







REGISTER_KERNEL_BUILDER(Name("GenKey").Device(DEVICE_CPU), GenKeyOP);
REGISTER_KERNEL_BUILDER(Name("Enc").Device(DEVICE_CPU), EncOP);
REGISTER_KERNEL_BUILDER(Name("Dec").Device(DEVICE_CPU), DecOP);
REGISTER_KERNEL_BUILDER(Name("MatMulVec").Device(DEVICE_CPU), MatMulVecOP);
REGISTER_KERNEL_BUILDER(Name("MatMulVecToShare").Device(DEVICE_CPU), MatMulVecToShareOP);
REGISTER_KERNEL_BUILDER(Name("VecMulVec").Device(DEVICE_CPU), VecMulVecOP);
REGISTER_KERNEL_BUILDER(Name("CipherToShare").Device(DEVICE_CPU), CipherToShareOP);

