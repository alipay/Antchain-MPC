/*
 * TPC.inl
 */
#define SOFTMAX_ITER_NUM 16
//#define TEST_SOFTMAX true
#define TEST_SIGMOID true

#pragma once

#include "TPC.h"

#include <bitset>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

#include "../gpu/bitwise.cuh"
#include "../gpu/convolution.cuh"
#include "../gpu/conv.cuh"
#include "../gpu/DeviceData.h"
#include "../gpu/functors.cuh"
#include "../gpu/matrix.cuh"
#include "../gpu/gemm.cuh"
#include "../gpu/StridedRange.cuh"
#include "../globals.h"
#include "Precompute.h"
#include "../util/functors.h"
#include "../util/Profiler.h"

extern Precompute PrecomputeObject;
extern Profiler comm_profiler;
extern Profiler func_profiler;

// Functors

struct tpc_convex_comb_functor {
    const int party;
    tpc_convex_comb_functor(int _party) : party(_party) {}
    
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple t) {
        // b, c share, d share
        if (thrust::get<0>(t) == 1) {
            switch(party) {
                case TPC<uint64_t>::PARTY_A: // doesn't really matter what type TPC is templated at here
                    thrust::get<2>(t) = 1 - thrust::get<1>(t);
                    break;
                case TPC<uint64_t>::PARTY_B:
                    thrust::get<2>(t) = -thrust::get<1>(t);
                    break;
            }
        } else {
            thrust::get<2>(t) = thrust::get<1>(t);
        }
    }
};

// Prototypes

template<typename T>
void localMatMul(const TPC<T> &a, const TPC<T> &b, const TPC<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b);

template<typename T, typename I, typename I2, typename I3>
void localConvolution(TPC<T, I> &im, TPC<T, I2> &filters, DeviceData<T, I3> &out,
        size_t imageWidth, size_t imageHeight, size_t filterSize, size_t Din, size_t Dout,
        size_t stride, size_t padding);

template<typename T, typename I, typename I2>
void carryOut(TPC<T, I> &p, TPC<T, I> &g, int k, TPC<T, I2> &out);

template<typename T, typename I, typename I2>
void getPowers(TPC<T, I> &in, DeviceData<T, I2> &pow);

template<typename T, typename I, typename I2, typename Functor>
void taylorSeries(TPC<T, I> &in, TPC<T, I2> &out,
        double a0, double a1, double a2,
        Functor fn);


template<typename T, typename U, typename I, typename I2>
void convex_comb(TPC<T, I> &a, TPC<T, I> &c, DeviceData<U, I2> &b);

// TPC class implementation 

template<typename T, typename I>
TPCBase<T, I>::TPCBase(DeviceData<T, I> *a) : 
                shareA(a) {}

template<typename T, typename I>
void TPCBase<T, I>::set(DeviceData<T, I> *a) {
    shareA = a;
}




template<typename T, typename I>
size_t TPCBase<T, I>::size() const {
    return shareA->size();
}

template<typename T, typename I>
void TPCBase<T, I>::zero() {
    shareA->zero();
};

template<typename T, typename I>
void TPCBase<T, I>::fill(T val) {
    shareA->fill(partyNum == PARTY_A ? val : 0);
}

template<typename T, typename I>
void TPCBase<T, I>::setPublic(std::vector<double> &v) {
    std::vector<T> shifted_vals;
    for (double f : v) {
        shifted_vals.push_back((T) (f * (1 << FLOAT_PRECISION)));
    }

    switch (partyNum) {
        case PARTY_A:
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), shareA->begin());
            break;
        case PARTY_B:
            shareA->zero();
            break;
    }
};

template<typename T, typename I>
DeviceData<T, I> *TPCBase<T, I>::getShare(int i) {
    switch (i) {
        case 0:
            return shareA;
        default:
            return nullptr;
    }
}

template<typename T, typename I>
const DeviceData<T, I> *TPCBase<T, I>::getShare(int i) const {
    switch (i) {
        case 0:
            return shareA;
        default:
            return nullptr;
    }
}

template<typename T, typename I>
TPCBase<T, I> &TPCBase<T, I>::operator+=(const T rhs) {
    if (partyNum == PARTY_A) {
        *shareA += rhs;
    }
    return *this;
}

template<typename T, typename I>
TPCBase<T, I> &TPCBase<T, I>::operator-=(const T rhs) {
    if (partyNum == PARTY_A) {
        *shareA -= rhs;
    }
    return *this;
}

template<typename T, typename I>
TPCBase<T, I> &TPCBase<T, I>::operator*=(const T rhs) {
    *shareA *= rhs;
    return *this;
}

template<typename T, typename I>
TPCBase<T, I> &TPCBase<T, I>::operator>>=(const T rhs) {
    *shareA >>= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCBase<T, I> &TPCBase<T, I>::operator+=(const DeviceData<T, I2> &rhs) {
    if (partyNum == PARTY_A) {
        *shareA += rhs;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCBase<T, I> &TPCBase<T, I>::operator-=(const DeviceData<T, I2> &rhs) {
    if (partyNum == PARTY_A) {
        *shareA -= rhs;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCBase<T, I> &TPCBase<T, I>::operator*=(const DeviceData<T, I2> &rhs) {
    *shareA *= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCBase<T, I> &TPCBase<T, I>::operator^=(const DeviceData<T, I2> &rhs) {
    if (partyNum == PARTY_A) {
        *shareA ^= rhs;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCBase<T, I> &TPCBase<T, I>::operator&=(const DeviceData<T, I2> &rhs) {
    *shareA &= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCBase<T, I> &TPCBase<T, I>::operator>>=(const DeviceData<T, I2> &rhs) {
    *shareA >>= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCBase<T, I> &TPCBase<T, I>::operator<<=(const DeviceData<T, I2> &rhs) {
    *shareA <<= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCBase<T, I> &TPCBase<T, I>::operator+=(const TPCBase<T, I2> &rhs) {
    *shareA += *rhs.getShare(0);
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCBase<T, I> &TPCBase<T, I>::operator-=(const TPCBase<T, I2> &rhs) {
    *shareA -= *rhs.getShare(0);
    return *this;
}

//--------added----------------
template<typename T, typename I>
template<typename I2>
TPCBase<T, I> TPCBase<T, I>::operator+(const TPCBase<T, I2> &rhs) 
{
    //DeviceData<T, I> dd(this->size());
    //auto dd = std::make_shared<DeviceData<T, I>>(this->size());
    DeviceData<T, I> * dd=new DeviceData<T, I>(this->size());
    TPCBase<T, I> r(dd);
    r.zero();
    r += *this;
    r += rhs;
    return r;
}

// template<typename T, typename I>
// template<typename I2>
//TPC<T, I> TPC<T, I>::operator+(const TPCBase<T, I2> &rhs) 
// template<typename T>
// TPC<T, BufferIterator<T> > TPC<T, BufferIterator<T> >::operator+(const TPC<T, BufferIterator<T> >  &rhs)
// {
//     //DeviceData<T, I> dd(this->size());
//     //auto dd = std::make_shared<DeviceData<T, I>>(this->size());
//     //DeviceData<T, I> * dd=new DeviceData<T, I>(this->size());
//     TPC<T, BufferIterator<T>> r(this->size());
//     r.zero();
//     r += *this;
//     r += rhs;
//     return r;
// }

template<typename T, typename I>
template<typename I2>
TPCBase<T, I> TPCBase<T, I>::operator-(const TPCBase<T, I2> &rhs) {
    //DeviceData<T, I> dd(this->size());
    //auto dd = std::make_shared<DeviceData<T, I>>(this->size());
    DeviceData<T, I> * dd=new DeviceData<T, I>(this->size());
    TPCBase<T, I> r(dd);
    r.zero();
    r += *this;
    r -= rhs;
    return r;
}

template<typename T, typename I>
template<typename I2>
TPCBase<T, I> TPCBase<T, I>::operator*(const TPCBase<T, I2> &rhs) {
    //DeviceData<T, I> dd(this->size());
    //auto dd = std::make_shared<DeviceData<T, I>>(this->size());
    DeviceData<T, I> * dd=new DeviceData<T, I>(this->size());
    TPCBase<T, I> r(& (*dd));
    r.zero();
    r += *this;
    r *= rhs;
    return r;
}
//--------------------



template<typename T, typename I>
template<typename I2>
TPCBase<T, I> &TPCBase<T, I>::operator*=(const TPCBase<T, I2> &rhs) {

    size_t size = rhs.size();
    TPC<T> x(size), y(size), z(size);
    PrecomputeObject.getBeaverTriples<T, TPC<T> >(x, y, z);
    DeviceData<T> e(size), f(size), temp(size);

    *x.getShare(0) += *this->getShare(0); 
    *y.getShare(0) += *rhs.getShare(0);
    reconstruct(x, e); reconstruct(y, f);
    *x.getShare(0) -= *this->getShare(0);
    *y.getShare(0) -= *rhs.getShare(0);
    
    this->zero();
    *this += z;

    temp.zero();
    temp += f;
    temp -= *y.getShare(0);
    temp *= e;
    *this += temp;

    temp.zero();
    temp -= *x.getShare(0);
    temp *= f;
    *this += temp;
 
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCBase<T, I> &TPCBase<T, I>::operator^=(const TPCBase<T, I2> &rhs) {
    *shareA ^= *rhs.getShare(0);
    return *this;
}

template<typename T, typename I>
template<typename I2>
TPCBase<T, I> &TPCBase<T, I>::operator&=(const TPCBase<T, I2> &rhs) {

    size_t size = rhs.size();
    TPC<T> x(size), y(size), z(size);
    PrecomputeObject.getBooleanBeaverTriples<T, TPC<T> >(x, y, z);
    DeviceData<T> e(size), f(size), temp(size);

    *x.getShare(0) ^= *this->getShare(0); 
    *y.getShare(0) ^= *rhs.getShare(0);
    reconstruct(x, e); reconstruct(y, f);
    *x.getShare(0) ^= *this->getShare(0);
    *y.getShare(0) ^= *rhs.getShare(0);
    
    this->zero();
    *this ^= z;

    temp.zero();
    temp ^= f;
    temp ^= *y.getShare(0);
    temp &= e;
    *this ^= temp;

    temp.zero();
    temp ^= *x.getShare(0);
    temp &= f;
    *this ^= temp;
 
    return *this;
}

//TO_BE_DONE
template<typename T, typename I>
int TPCBase<T, I>::otherParty(int party) {
	switch(party) {
        case PARTY_A:
            return PARTY_B;
        default: // PARTY_B
            return PARTY_A;
    }	
}

template<typename T, typename I>
int TPCBase<T, I>::numShares() {
    return 1;
}

template<typename T, typename I>
TPC<T, I>::TPC(DeviceData<T, I> *a) : TPCBase<T, I>(a) {}





template<typename T, typename I>
TPC<T, I>::TPC(TPCBase<T, I> x): TPCBase<T, I>(x.getShare(0)) {
}
// {
//     *this->getShare(0) = *x.getShare(0); 
//     //return *this;
// }





template<typename T>
TPC<T, BufferIterator<T> >::TPC(DeviceData<T> *a) :
    TPCBase<T, BufferIterator<T> >(a) {}



template<typename T>
TPC<T, BufferIterator<T> >::TPC(TPCBase<T, BufferIterator<T> > x) :
    TPCBase<T, BufferIterator<T> >(x.getShare(0)) {
       // *this->getShare(0) = *x.getShare(0); 
    //return *this;
}



template<typename T>
TPC<T, BufferIterator<T> >::TPC(size_t n) :
    _shareA(n),
    TPCBase<T, BufferIterator<T> >(&_shareA) {}



template<typename T>
TPC<T, BufferIterator<T> >::TPC(std::initializer_list<double> il, bool convertToFixedPoint) :
    _shareA(il.size()),
    TPCBase<T, BufferIterator<T> >(&_shareA) {

    std::vector<T> shifted_vals;
    for (double f : il) {
        if (convertToFixedPoint) {
            shifted_vals.push_back((T) (f * (1 << FLOAT_PRECISION)));
        } else {
            shifted_vals.push_back((T) f);
        }
    }

    switch (partyNum) {
        case TPC<T>::PARTY_A:
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), _shareA.begin());
            break;
        case TPC<T>::PARTY_B:
            // nothing
            break;
    }
}











template<typename T>
void TPC<T, BufferIterator<T> >::resize(size_t n) {
    _shareA.resize(n);
}

template<typename T, typename I>
void dividePublic(TPC<T, I> &a, T denominator) {

    TPC<T> r(a.size()), rPrime(a.size());
    PrecomputeObject.getDividedShares<T, TPC<T> >(r, rPrime, denominator, a.size()); 
    a -= rPrime;
    
    DeviceData<T> reconstructed(a.size());
    reconstruct(a, reconstructed);
    reconstructed /= denominator;

    a.zero();
    a += r;
    a += reconstructed;
}


template<typename T, typename I>
void dividePublic_free(TPC<T, I> &a, T denominator) {
    DeviceData<T> x(a.size());
    x.zero();
    if (partyNum == TPC<T>::PARTY_A)
    {
        x += *a.getShare(0);
        x /= denominator;
    }
    else
    {
        x -= *a.getShare(0);
        x /= denominator;
        x *= (-1);
    }
    a.zero();
    a += x;
}

template<typename T, typename I, typename I2>
void dividePublic(TPC<T, I> &a, DeviceData<T, I2> &denominators) {

    assert(denominators.size() == a.size() && "TPC dividePublic powers size mismatch");

    TPC<T> r(a.size()), rPrime(a.size());
    PrecomputeObject.getDividedShares<T, I2, TPC<T> >(r, rPrime, denominators, a.size()); 

    a -= rPrime;

    DeviceData<T> reconstructed(a.size());
    reconstruct(a, reconstructed);
    reconstructed /= denominators;

    a.zero();
    a += r;
    a += reconstructed;
}

/*
#define MAX_BITS 30

template<typename T, typename I>
void dividePublic(TPC<T, I> &a, size_t denominator) {

    //printf("dividepublic denominator %d\n", denominator);
    //fflush(stdout);
    //assert((double)denominator > 1.0 && "dividePublic got denominator < 1.0");
    
    if (denominator == 1) return;

    double divisor = 1.0 / denominator;
    divisor *= 1 << MAX_BITS;

    std::bitset<MAX_BITS> divisorBits = (T) divisor;

    DeviceData<T> temp(a.size());
    switch (partyNum) {
        case TPC<T>::PARTY_A:

            temp += *a.getShare(0);
            a.zero();
            
            for (int i = MAX_BITS - 1; i >= 0; i--) {
                temp >>= 1;

                if (divisorBits[i]) {
                    *a.getShare(0) += temp;
                }
            }

            break;
        case TPC<T>::PARTY_B:

            temp += *a.getShare(0);
            a.zero();

            temp *= (T)-1;

            for (int i = MAX_BITS - 1; i >= 0; i--) {
                temp >>= 1;

                if (divisorBits[i]) {
                    temp *= (T)-1;
                    *a.getShare(0) += temp;
                    temp *= (T)-1;
                }
            }

            break;
    }
}

template<typename T>
struct divisor_bit_functor {

    divisor_bit_functor() {}
    __host__ __device__ T operator()(const T &x) const {
        return (T)((1.0 / x) * (1 << MAX_BITS));
    }
};

template<typename T>
struct secureml_truncation_functor {

    T scalar;
    int bit;

    secureml_truncation_functor(T _scalar, int _bit) : scalar(_scalar), bit(_bit) {}
    __host__ __device__ T operator()(const T &divisorBits, const T &value) const {
        return ((divisorBits >> bit) & 1) ? scalar * value : 0;
    }
};

template<typename T>
struct filter_by_val_functor {

    T target_val;

    filter_by_val_functor(T _target) : target_val(_target) {}

    __host__ __device__ T operator()(const T &x, const T &val) const {
        return (val == target_val) ? x : 0;
    }
};

template<typename T, typename I, typename I2>
void dividePublic(TPC<T, I> &a, DeviceData<T, I2> &denominators) {

//    for (int i = 0; i < denominators.size(); i++) {
//        assert((double)denominators[i] > 1.0 && "dividePublic got denominator < 1.0");
//    }

    DeviceData<T> recon_a(a.size());
    reconstruct(a, recon_a);

    DeviceData<T> divisorBits(denominators.size());
    thrust::transform(denominators.begin(), denominators.end(), divisorBits.begin(), divisor_bit_functor<T>());

    DeviceData<T> temp(a.size());
    DeviceData<T> intermediateTruncations(a.size());

    switch (partyNum) {
        case TPC<T>::PARTY_A:

            temp += *a.getShare(0);
            a.zero();

            // Add temp unshifted back into share in cases where denominator is 1
            thrust::transform(temp.begin(), temp.end(), denominators.begin(), a.getShare(0)->begin(), filter_by_val_functor<T>(1));

            for (int i = MAX_BITS - 1; i >= 0; i--) {
                temp >>= 1;

                intermediateTruncations.zero();
                thrust::transform(divisorBits.begin(), divisorBits.end(), temp.begin(), intermediateTruncations.begin(), secureml_truncation_functor<T>(1, i));

                *a.getShare(0) += intermediateTruncations;
            }

            break;

        case TPC<T>::PARTY_B:

            temp += *a.getShare(0);
            a.zero();

            thrust::transform(temp.begin(), temp.end(), denominators.begin(), a.getShare(0)->begin(), filter_by_val_functor<T>(1));

            temp *= (T)-1;

            for (int i = MAX_BITS - 1; i >= 0; i--) {
                temp >>= 1;

                intermediateTruncations.zero();
                thrust::transform(divisorBits.begin(), divisorBits.end(), temp.begin(), intermediateTruncations.begin(), secureml_truncation_functor<T>((T)-1, i));

                *a.getShare(0) += intermediateTruncations;
            }

            break;
    }
}
*/

template<typename T, typename I, typename I2>
void reconstruct(TPC<T, I> &in, DeviceData<T, I2> &out) {

    comm_profiler.start();
    // 1 - send shareA to next party
    in.getShare(0)->transmit(TPC<T>::otherParty(partyNum));

    // 2 - receive shareA from previous party into DeviceBuffer 
    DeviceData<T> rxShare(in.size());
    rxShare.receive(TPC<T>::otherParty(partyNum));

    in.getShare(0)->join();
    rxShare.join();
    comm_profiler.accumulate("comm-time");

    // 3 - result is our shareB + received shareA
    out.zero();
    out += *in.getShare(0);
    out += rxShare;

    func_profiler.add_comm_round();
}

template<typename T>
void matmul(const TPC<T> &a, const TPC<T> &b, TPC<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c, T truncation) {

    localMatMul(a, b, c, M, N, K, transpose_a, transpose_b, transpose_c);

    // truncate
    dividePublic(c, (T)1 << truncation);
}

template<typename T, typename U, typename I, typename I2, typename I3, typename I4>
void selectShare(const TPC<T, I> &x, const TPC<T, I2> &y, const TPC<U, I3> &b, TPC<T, I4> &z) {

    assert(x.size() == y.size() && x.size() == b.size() && x.size() == z.size() && "TPC selectShare input size mismatch");

    //TO_BE_DONE
    TPC<T> c(x.size());
    TPC<U> cbits(b.size());

    // b XOR c, then open -> e
    cbits ^= b;

    DeviceData<U> e(cbits.size());
    reconstruct(cbits, e);

    // d = 1-c if e=1 else d = c       ->        d = (e)(1-c) + (1-e)(c)
    TPC<T> d(e.size());
    convex_comb(d, c, e);

    // z = ((y - x) * d) + x
    TPC<T> result(x.size());
    result += y;
    result -= x;
    result *= d;
    result += x;
    
    z.zero();
    z += result;
}

template<typename T, typename I, typename I2>
void sqrt(TPC<T, I> &in, TPC<T, I2> &out) {
    /*
     * Approximations:
     *   > sqrt(x) = 0.424 + 0.584(x)
     *     sqrt(x) = 0.316 + 0.885(x) - 0.202(x^2)
     */
    taylorSeries(in, out, 0.424, 0.584, 0.0, sqrt_lambda());
}

template<typename T, typename I, typename I2>
void inverse(TPC<T, I> &in, TPC<T, I2> &out) {
    /*
     * Approximations:
     *     1/x = 2.838 - 1.935(x)
     *   > 1/x = 4.245 - 5.857(x) + 2.630(x^2)
     */
    taylorSeries(in, out, 4.245, -5.857, 2.630, inv_lambda());
}

template<typename T, typename I, typename I2>
void sigmoid(TPC<T, I> &in, TPC<T, I2> &out) {
    /*
     * Approximation:
     *   > sigmoid(x) = 0.494286 + 0.275589(x) + -0.038751(x^2)
     */
    taylorSeries(in, out, 0.494286, 0.275589, -0.038751, sigmoid_lambda());
}



//-------------added----------------------------
template<typename T, typename I, typename I2>
void softmax(const TPC<T, I> &x,  TPC<T, I2> &y, int batch_size, int nClasses)
{
    //y = y + (x - (y * x).reduce_sum(axis=-1, keepdims=True)) * y / softmax_iter_num
    uint64_t softmax_iter_num = SOFTMAX_ITER_NUM;
    assert(x.size()==batch_size*nClasses);
    //TPC<T> y(batch_size*nClasses);
    TPC<T, I> inner_value(batch_size); 
    TPC<T, I> inner_value_ext(batch_size*nClasses); 

    // DeviceData<T> ones(pow.size());
    // ones.fill(1);
    // ones <<= pow;

    y.fill((T) (1.0/nClasses * (1<<FLOAT_PRECISION)));
    //matmul(x, y, inner_value, false, true, false, true);
    
    TPC<T, I> w(batch_size*nClasses);
    for (int i=0; i<softmax_iter_num; i++)
    {
        
        //auto w = x*y;
        w.zero();
        w += x;
        w *= y;
        
        gpu::reduceSum(
                w.getShare(0),
                inner_value.getShare(0), 
                false, batch_size, nClasses
            );
        dividePublic_free(inner_value, (T)(1<<FLOAT_PRECISION));
        gpu::vectorExpand(inner_value.getShare(0), inner_value_ext.getShare(0), nClasses);

        
        //auto z = (x - inner_value_ext) * y;
        //z /= softmax_iter_num;
        //y +=z;
        w.zero();
        w += x;
        w -=inner_value_ext;
        w *=y;
        dividePublic_free(w,  (T) (1<<FLOAT_PRECISION));
        //w *= (1.0/softmax_iter_num);
        dividePublic_free(w, (T)  softmax_iter_num);
        y += w;
    }
}




template<typename T, typename I, typename I2>
void softmax_reveal(const TPC<T, I> &input,  TPC<T, I2> &deltas, int batch_size, int nClasses)
{

    // Share<T> x(deltas.size());
    // x += *layers[layers.size() - 1]->getActivation();
    TPC<T, I> x(input.size());
    x.zero();
    x += input;

    //printShareFinite(*layers[layers.size() - 1]->getActivation(), "activations", 10);

    //printShareTensor(x, "logits", 1, 1, 1, 10);

    DeviceData<T, I> revealedX(x.size());
    reconstruct(x, revealedX);

    thrust::device_vector<double> floatActivations(revealedX.size());
    thrust::transform(revealedX.begin(), revealedX.end(),
            floatActivations.begin(), to_double_functor<T>());

    /*
    printf("input to softmax:\n");
    for (int i = 0; i < floatActivations.size(); i++) {
        double act = floatActivations[i];
        printf("%f ", act);
    }
    printf("\n");
    */

    // int nClasses = labels.size() / MINI_BATCH_SIZE;
    thrust::device_vector<double> sums(floatActivations.size() / nClasses, 0);
    for (int i = 0; i < floatActivations.size(); i++) {
        floatActivations[i] = exp(floatActivations[i]); 
        //floatActivations[i] *= floatActivations[i]; 

        sums[i / nClasses] += floatActivations[i];
    }

    for (int i = 0; i < floatActivations.size(); i++) {
        floatActivations[i] /= sums[i / nClasses];
    }

    /*
    printf("after softmax:\n");
    for (int i = 0; i < floatActivations.size(); i++) {
        double act = floatActivations[i];
        printf("%f ", act);
    }
    printf("\n");
    */

    DeviceData<T, I> softmax_values(floatActivations.size());
    thrust::transform(floatActivations.begin(), floatActivations.end(), softmax_values.begin(),
        to_fixed_functor<T>());

    deltas.zero();
    deltas += softmax_values;
}







template<typename T, typename I, typename I2>
void softmax_asm(const TPC<T, I> &input,  TPC<T, I2> &deltas, int batch_size, int nClasses)
{
    TPC<T, I> x(input.size());
    x.zero();
    x += input;
    // x += *layers[layers.size() - 1]->getActivation();

    size_t n = 3;
    dividePublic(x, (T)1 << n);

    x += 1 << FLOAT_PRECISION;
    
    for (int i = 0; i < n - 1; i++) {
        x *= x;
        dividePublic(x, (T)1 << FLOAT_PRECISION);
    } 

    //int nClasses = labels.size() / MINI_BATCH_SIZE;

    TPC<T, I> sums(batch_size);

        gpu::reduceSum(
            x.getShare(0),
            sums.getShare(0), 
            false, batch_size, nClasses
        );
    

    TPC<T, I> inversedSums(batch_size);

#if 0
    DeviceData<T> revealedSums(sums.size());
    reconstruct(sums, revealedSums);

    DeviceData<T> invertedRevealedSums(sums.size());

    thrust::transform(revealedSums.begin(), revealedSums.end(), invertedRevealedSums.begin(),
        inv_fixed_point_functor<T>());

    inversedSums += invertedRevealedSums;
#else
    inverse(sums, inversedSums);
#endif
    
    gpu::vectorExpand(inversedSums.getShare(0), deltas.getShare(0), nClasses);
    
    deltas *= x;
    dividePublic(deltas, (T)1 << FLOAT_PRECISION);

}




//template<typename T, typename I, typename I2>
// template<typename T>
// void softmax_adhoc(const TPC<T> &x,  TPC<T> &deltas, int batch_size, int numClasses)
// {

//      //Share<T> maxVals(MINI_BATCH_SIZE);
//     TPC<T> maxVals(batch_size);
//     //int numClasses = deltas.size() / MINI_BATCH_SIZE;
//     int paddedSize = pow(ceil(log2(numClasses)), 2);

//     //Share<T> pools(MINI_BATCH_SIZE * paddedSize);
//     TPC<T> pools(batch_size * paddedSize);
//     //for(int i = 0; i < Share<T>::numShares(); i++) {

//         // TODO fix 4PC
//     T pad_value = (T)(-10 * (1 << FLOAT_PRECISION));

//     DeviceData<T>* xs = x.getShare(0);
//     DeviceData<T>* ps = pools.getShare(0);

//     //gpu::sin2kpix();
//     //gpu::averagepool_im2row();
//     //gpu::stride_pad(xs, ps, numClasses, paddedSize - numClasses, pad_value);
//     //}

//     //Share<uint8_t> 
//     TPC<uint8_t> expandedPrime(pools.size());
//     maxpool(pools, maxVals, expandedPrime, paddedSize);

//     //printShareFinite(maxVals, "max val", 1);

//     //Share<T> 
//     TPC<T> expandedMaxVals(deltas.size());
//     //for(int i = 0; i < Share<T>::numShares(); i++) {
//         gpu::vectorExpand(maxVals.getShare(0), expandedMaxVals.getShare(0), numClasses);
//     //}

//     //printShareFinite(expandedMaxVals, "expanded max val", 10);

//     //Share<T>
//     TPC<T> diff(deltas.size());
//     //diff += *layers[layers.size() - 1]->getActivation();
//     diff += x;
//     diff -= expandedMaxVals;
//     //printShareFinite(diff, "diff", 10);

//     diff += (1 << (FLOAT_PRECISION + 1));
//     //printShareFinite(diff, "diff + 2", 10);

//     //Share<T> 
//     TPC<T> zeros(deltas.size());
//     Share<uint8_t> b(deltas.size());
//     ReLU(diff, zeros, b);
//     zeros.zero();
//     zeros += (T)(0.001 * (1 << FLOAT_PRECISION));

//     dividePublic(diff, (T)2);

//     //Share<T> 
//     TPC<T> exponentialApprox(deltas.size());
//     selectShare(zeros, diff, b, exponentialApprox);

//     //Share<T> 
//     TPC<T> sums(batch_size);
//     //for(int share = 0; share < Share<T>::numShares(); share++) {
//         gpu::reduceSum(
//             exponentialApprox.getShare(0),
//             sums.getShare(0),
//             false, batch_size, numClasses
//         );
//     //}

//     DeviceData<T> revealedSums(sums.size());
//     reconstruct(sums, revealedSums);

//     thrust::transform(revealedSums.begin(), revealedSums.end(), revealedSums.begin(), inv_fixed_point_functor<T>());

//     //Share<T> 
//     TPC<T> inversedSums(batch_size);
//     inversedSums += revealedSums;
//     //inverse(sums, inversedSums);

//     //for(int share = 0; share < Share<T>::numShares(); share++) {
//         gpu::vectorExpand(inversedSums.getShare(0), deltas.getShare(0), numClasses);
//     //}

//     deltas *= exponentialApprox;
//     dividePublic(deltas, (T)1 << FLOAT_PRECISION);


// }


template<typename T>
void softmax_adhoc(const TPC<T> &x,  TPC<T> &deltas, int batch_size, int numClasses)
{

     //Share<T> maxVals(MINI_BATCH_SIZE);
    TPC<T> maxVals(batch_size);
    //int numClasses = deltas.size() / MINI_BATCH_SIZE;
    int paddedSize = pow(ceil(log2(numClasses)), 2);

    //Share<T> pools(MINI_BATCH_SIZE * paddedSize);
    TPC<T> pools(batch_size * paddedSize);
    //for(int i = 0; i < Share<T>::numShares(); i++) {

        // TODO fix 4PC
    T pad_value = (T)(-10 * (1 << FLOAT_PRECISION));


    // gpu::stride_pad(x.getShare(0), pools.getShare(0), numClasses, paddedSize - numClasses, pad_value);


    //Share<uint8_t> 
    TPC<uint8_t> expandedPrime(pools.size());

    std::cout<<"----------l1059-------------"<<std::endl;
    maxpool(pools, maxVals, expandedPrime, paddedSize);

    //printShareFinite(maxVals, "max val", 1);

    //Share<T> 
    TPC<T> expandedMaxVals(deltas.size());
    //for(int i = 0; i < Share<T>::numShares(); i++) {
    std::cout<<"----------l1067-------------"<<std::endl;
        gpu::vectorExpand(maxVals.getShare(0), expandedMaxVals.getShare(0), numClasses);
    //}

    //printShareFinite(expandedMaxVals, "expanded max val", 10);

    //Share<T>
    TPC<T> diff(deltas.size());
    //diff += *layers[layers.size() - 1]->getActivation();
    diff += x;
    diff -= expandedMaxVals;
    //printShareFinite(diff, "diff", 10);

    diff += (1 << (FLOAT_PRECISION + 1));
    //printShareFinite(diff, "diff + 2", 10);

    //Share<T> 
    TPC<T> zeros(deltas.size());
    //Share
    TPC<uint8_t> b(deltas.size());
    std::cout<<"----------l1087-------------"<<std::endl;
    ReLU(diff, zeros, b);
    zeros.zero();
    zeros += (T)(0.001 * (1 << FLOAT_PRECISION));
std::cout<<"----------l1091-------------"<<std::endl;
    dividePublic(diff, (T)2);

    //Share<T> 
    TPC<T> exponentialApprox(deltas.size());
    std::cout<<"zeros.size()=="<<zeros.size()<<std::endl;
    std::cout<<"diff.size()=="<<diff.size()<<std::endl;
    std::cout<<"b.size()=="<<b.size()<<std::endl;
    std::cout<<"exponentialApprox.size()=="<<exponentialApprox.size()<<std::endl;
    selectShare(zeros, diff, b, exponentialApprox);

    //Share<T> 
    TPC<T> sums(batch_size);
    //for(int share = 0; share < Share<T>::numShares(); share++) {
        gpu::reduceSum(
            exponentialApprox.getShare(0),
            sums.getShare(0),
            false, batch_size, numClasses
        );
    //}

    DeviceData<T> revealedSums(sums.size());
    reconstruct(sums, revealedSums);

    thrust::transform(revealedSums.begin(), revealedSums.end(), revealedSums.begin(), inv_fixed_point_functor<T>());

    //Share<T> 
    TPC<T> inversedSums(batch_size);
    inversedSums += revealedSums;
    //inverse(sums, inversedSums);

    //for(int share = 0; share < Share<T>::numShares(); share++) {
        gpu::vectorExpand(inversedSums.getShare(0), deltas.getShare(0), numClasses);
    //}

    deltas *= exponentialApprox;
    dividePublic(deltas, (T)1 << FLOAT_PRECISION);


}



template<typename T>
void sin2kpix(const TPC<T> &x, TPC<T> &y, size_t m, size_t rows, size_t cols) // y = sin 2 k pi x/2^m
{   // (sin 2pix/2^m, sin 2*2*pix/2^m,  sin 2*3*pix/2^m, ...... ) 
    TPC<T> x_tilde(rows), s_tilde(rows*cols), c_tilde(rows*cols), delta_x(rows);
    PrecomputeObject.getSinTriples<T, TPC<T> >(x_tilde, s_tilde, m, rows, cols);
    PrecomputeObject.getCosTriples<T, TPC<T> >(x_tilde, c_tilde, m, rows, cols);
    delta_x.zero();
    delta_x += x;
    delta_x -= x_tilde;


    DeviceData<T> reconstructed_delta_x(rows), s_delta(rows*cols), c_delta(rows*cols);

    reconstruct(delta_x, reconstructed_delta_x);



    gpu::sin2kpix(reconstructed_delta_x, s_delta, m+FLOAT_PRECISION, rows, cols);
    gpu::cos2kpix(reconstructed_delta_x, c_delta, m+FLOAT_PRECISION, rows, cols);
    DeviceData<T> sc(rows*cols), cs(rows*cols);
    sc.zero();
    cs.zero();
    sc += s_delta;
    sc *= *c_tilde.getShare(0);
    cs += c_delta;
    cs *= *s_tilde.getShare(0);
    y.zero();
    y += sc;
    y += cs;
    dividePublic_free(y,  (T) (1<<FLOAT_PRECISION));
    cudaThreadSynchronize();
}



template<typename T>
void sigmoid_fourier(const TPC<T> &x,  TPC<T> &y)
{

    size_t batch_size = x.size();
    size_t cols = 5;
    size_t m=5;
    TPC<T> s(batch_size*cols);
    sin2kpix(x, s, m, batch_size, cols);



    // double coeff[]={79.01170243, -4.37338149, 21.67435635, -5.88409071, 10.44369469};

    // std::vector<T> coeffT;
    // coeffT.resize(cols);
    // for (int i=0; i<cols; i++)
    // {
    //     coeffT[i] = (T) (coeff[i] * (1<< FLOAT_PRECISION));
    // }
    DeviceData<T> k={5693393177946504192,0, 1561801970901685504, 0,  752547512228083584};
    k >>= (56-FLOAT_PRECISION);
    DeviceData<T> km={0, 315135347979408384, 0, 423993419663523136,  0};
    km >>= (56-FLOAT_PRECISION);
    k -= km;


    //k.set(coeffT);
    y.zero();
    gpu::gemm(batch_size, 1, cols, s.getShare(0), true, &k, false, y.getShare(0), false);
    dividePublic_free(y, (T)(1<<FLOAT_PRECISION));
    dividePublic_free(y, (T) 128);
    y += (T) (0.5 * (1<< FLOAT_PRECISION));
    
    
}




//---------------------------------------------
template<typename T>
void localFprop(const TPC<T> &A, const TPC<T> &B, TPC<T> &C,
        int batchSize, int imageHeight, int imageWidth, int Din,
        int Dout, int filterHeight, int filterWidth,
        int paddingHeight, int paddingWidth,
        int stride, int dilation) {

    TPC<T> x(A.size()), y(B.size()), z(C.size());
    PrecomputeObject.getConvBeaverTriple_fprop<T, TPC<T> >(x, y, z, 
        batchSize, imageHeight, imageWidth, Din,
        Dout, filterHeight, filterWidth,
        paddingHeight, paddingWidth,
        stride, dilation);
    DeviceData<T> e(x.size()), f(y.size()), temp(z.size());

    x += A; y += B;
    reconstruct(x, e); reconstruct(y, f);
    x -= A; y -= B;

    C.zero();
    C += z;

    gpu::conv_fprop(&e, &f, &temp, 
        batchSize, imageHeight, imageWidth, Din,
        Dout, filterHeight, filterWidth,
        paddingHeight, paddingWidth,
        stride, dilation);
    C += temp;
    temp.zero();

    gpu::conv_fprop(&e, y.getShare(0), &temp, 
        batchSize, imageHeight, imageWidth, Din,
        Dout, filterHeight, filterWidth,
        paddingHeight, paddingWidth,
        stride, dilation);
    *C.getShare(0) -= temp;
    temp.zero();

    gpu::conv_fprop(x.getShare(0), &f, &temp, 
        batchSize, imageHeight, imageWidth, Din,
        Dout, filterHeight, filterWidth,
        paddingHeight, paddingWidth,
        stride, dilation);
    *C.getShare(0) -= temp;

    cudaThreadSynchronize();
}

template<typename T>
void localDgrad(const TPC<T> &A, const TPC<T> &B, TPC<T> &C,
        int batchSize, int outputHeight, int outputWidth, int Dout,
        int filterHeight, int filterWidth, int Din,
        int paddingHeight, int paddingWidth, int stride, int dilation,
        int imageHeight, int imageWidth) {

    TPC<T> x(A.size()), y(B.size()), z(C.size());
    PrecomputeObject.getConvBeaverTriple_dgrad<T, TPC<T> >(x, y, z, 
        batchSize, outputHeight, outputWidth, Dout,
        filterHeight, filterWidth, Din,
        paddingHeight, paddingWidth, stride, dilation,
        imageHeight, imageWidth);
    DeviceData<T> e(x.size()), f(y.size()), temp(z.size());

    x += A; y += B;
    reconstruct(x, e); reconstruct(y, f);
    x -= A; y -= B;

    C.zero();
    C += z;

    gpu::conv_dgrad(&e, &f, &temp, 
        batchSize, outputHeight, outputWidth, Dout,
        filterHeight, filterWidth, Din,
        paddingHeight, paddingWidth, stride, dilation,
        imageHeight, imageWidth);
    C += temp;
    temp.zero();

    gpu::conv_dgrad(&e, y.getShare(0), &temp, 
        batchSize, outputHeight, outputWidth, Dout,
        filterHeight, filterWidth, Din,
        paddingHeight, paddingWidth, stride, dilation,
        imageHeight, imageWidth);
    *C.getShare(0) -= temp;
    temp.zero();

    gpu::conv_dgrad(x.getShare(0), &f, &temp, 
        batchSize, outputHeight, outputWidth, Dout,
        filterHeight, filterWidth, Din,
        paddingHeight, paddingWidth, stride, dilation,
        imageHeight, imageWidth);
    *C.getShare(0) -= temp;

    cudaThreadSynchronize();
}

template<typename T>
void localWgrad(const TPC<T> &A, const TPC<T> &B, TPC<T> &C,
        int batchSize, int outputHeight, int outputWidth, int Dout,
        int imageHeight, int imageWidth, int Din,
        int filterHeight, int filterWidth,
        int paddingHeight, int paddingWidth, int stride, int dilation) {

    TPC<T> x(A.size()), y(B.size()), z(C.size());
    PrecomputeObject.getConvBeaverTriple_wgrad<T, TPC<T> >(x, y, z, 
        batchSize, outputHeight, outputWidth, Dout,
        imageHeight, imageWidth, Din,
        filterHeight, filterWidth,
        paddingHeight, paddingWidth, stride, dilation);
    DeviceData<T> e(x.size()), f(y.size()), temp(z.size());

    x += A; y += B;
    reconstruct(x, e); reconstruct(y, f);
    x -= A; y -= B;

    C.zero();
    C += z;

    gpu::conv_wgrad(&e, &f, &temp, 
        batchSize, outputHeight, outputWidth, Dout,
        imageHeight, imageWidth, Din,
        filterHeight, filterWidth,
        paddingHeight, paddingWidth, stride, dilation);
    C += temp;
    temp.zero();

    gpu::conv_wgrad(&e, y.getShare(0), &temp, 
        batchSize, outputHeight, outputWidth, Dout,
        imageHeight, imageWidth, Din,
        filterHeight, filterWidth,
        paddingHeight, paddingWidth, stride, dilation);
    *C.getShare(0) -= temp;
    temp.zero();

    gpu::conv_wgrad(x.getShare(0), &f, &temp, 
        batchSize, outputHeight, outputWidth, Dout,
        imageHeight, imageWidth, Din,
        filterHeight, filterWidth,
        paddingHeight, paddingWidth, stride, dilation);
    *C.getShare(0) -= temp;

    cudaThreadSynchronize();
}

template<typename T>
void convolution(const TPC<T> &A, const TPC<T> &B, TPC<T> &C,
        cutlass::conv::Operator op,
        int batchSize, int imageHeight, int imageWidth, int filterSize,
        int Din, int Dout, int stride, int padding, int truncation) {

    int outputHeight = (imageHeight + 2 * padding - filterSize) / stride + 1; 
    int outputWidth = (imageWidth + 2 * padding - filterSize) / stride + 1; 
    C.zero();
    // DeviceData<T> localResult(C.size());

    switch (op) {
        case cutlass::conv::Operator::kFprop:
            localFprop(A, B, C,
                    batchSize, imageHeight, imageWidth, Din,
                    Dout, filterSize, filterSize,
                    padding, padding,
                    stride, (T)1);
            break;
        case cutlass::conv::Operator::kDgrad:
            localDgrad(A, B, C,
                    batchSize, outputHeight, outputWidth, Dout,
                    filterSize, filterSize, Din,
                    padding, padding, stride, (T)1,
                    imageHeight, imageWidth);
            break;
        case cutlass::conv::Operator::kWgrad:
            localWgrad(A, B, C,
                    batchSize, outputHeight, outputWidth, Dout,
                    imageHeight, imageWidth, Din,
                    filterSize, filterSize,
                    padding, padding, stride, (T)1);
            break;
    }

    // *C.getShare(0) += localResult;
    dividePublic(C, (T)1 << truncation);
}

// TODO change into 2 arguments with subtraction, pointer NULL indicates compare w/ 0
template<typename T, typename U, typename I, typename I2>
void dReLU(const TPC<T, I> &input, TPC<U, I2> &result) {

    //TO_BE_DONE
    int bitWidth = sizeof(T) * 8;

    TPC<T> r(input.size());
    TPC<U> rbits(input.size() * bitWidth);
    rbits.fill(1);

    DeviceData<T> a(input.size());
    r += input;
    reconstruct(r, a);
    a += 1;

    DeviceData<U> abits(rbits.size());
    gpu::bitexpand(&a, &abits);

    TPC<U> msb(input.size());

    // setCarryOutMSB overwrites abits/rbits, so make sure if we're party C that we don't accidentally use the modified values (hacky)
    gpu::setCarryOutMSB(*(rbits.getShare(0)), abits, *(msb.getShare(0)), bitWidth, partyNum == TPC<U>::PARTY_A);

    TPC<U> g(rbits.size());
    g.zero();
    g += rbits;
    g &= abits;

    TPC<U> p(rbits.size());
    p.zero();
    p += rbits;
    p ^= abits;

    TPC<U> preResult(result.size());
    carryOut(p, g, bitWidth, preResult);

    preResult ^= msb;

    result.fill(1);
    result -= preResult;
}
    
template<typename T, typename U, typename I, typename I2, typename I3>
void ReLU(const TPC<T, I> &input, TPC<T, I2> &result, TPC<U, I3> &dresult) {

    //TO_BE_DONE

    //func_profiler.start();
    dReLU(input, dresult);
    //func_profiler.accumulate("relu-drelu");

    TPC<T> zeros(input.size());

    //func_profiler.start();
    selectShare(zeros, input, dresult, result);
    //func_profiler.accumulate("relu-selectshare");
}

template<typename T, typename U, typename I, typename I2, typename I3>
void maxpool(TPC<T, I> &input, TPC<T, I2> &result, TPC<U, I3> &dresult, int k) {

    //TO_BE_DONE

    // d(Maxpool) setup
    dresult.fill(1);

    // split input into even, odd
    using SRIterator = typename StridedRange<I>::iterator;

    int stride = 2;
    int offset = 1;

    func_profiler.start();
    StridedRange<I> even0Range(input.getShare(0)->begin(), input.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> even0(even0Range.begin(), even0Range.end());
    TPC<T, SRIterator> even(&even0);

    StridedRange<I> odd0Range(input.getShare(0)->begin() + offset, input.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> odd0(odd0Range.begin(), odd0Range.end());
    TPC<T, SRIterator> odd(&odd0);
    func_profiler.accumulate("range creation");

    //printf("func-maxpool-post-rangecreate\n");
    //printMemUsage();

    while(k > 2) {

        // -- MP --

        // diff = even - odd
        func_profiler.start();
        TPC<T> diff(even.size());
        diff.zero();
        diff += even;
        diff -= odd;
        func_profiler.accumulate("maxpool-diff");

        //printf("func-maxpool-post-diff-k=%d\n", k);
        //printMemUsage();

        // DRELU diff -> b
        func_profiler.start();
        TPC<U> b(even.size());
        dReLU(diff, b);
        func_profiler.accumulate("maxpool-drelu");

        //printf("func-maxpool-post-drelu-k=%d\n", k);
        //printMemUsage();
     
        selectShare(odd, even, b, even);

        // unzip even -> into even, odd
        stride *= 2;

        //printf("func-maxpool-pre-rangeupdate-k=%d\n", k);
        //printMemUsage();

        func_profiler.start();
        even0Range.set(input.getShare(0)->begin(), input.getShare(0)->end(), stride);
        even0.set(even0Range.begin(), even0Range.end());
        even.set(&even0);

        odd0Range.set(input.getShare(0)->begin() + stride/2, input.getShare(0)->end(), stride);
        odd0.set(odd0Range.begin(), odd0Range.end());
        odd.set(&odd0);
        func_profiler.accumulate("maxpool-unzip");
     
        // -- dMP --

        //printf("func-maxpool-pre-expand-k=%d\n", k);
        //printMemUsage();

        // expandCompare b -> expandedB
        func_profiler.start();
        TPC<U> negated(b.size());
        negated.fill(1);
        negated -= b;
        TPC<U> expandedB(input.size());

        gpu::expandCompare(*b.getShare(0), *negated.getShare(0), *expandedB.getShare(0));

        func_profiler.accumulate("maxpool-expandCompare");

        //printf("func-maxpool-post-expand-k=%d\n", k);
        //printMemUsage();
     
        // dresult &= expandedB
        func_profiler.start();
        dresult &= expandedB;
        func_profiler.accumulate("maxpool-dcalc");

        k /= 2;
    }

    // Fencepost - don't unzip the final results after the last comparison and finish
    // calculating derivative.
 
    // -- MP --
 
    // diff = even - odd
    func_profiler.start();
    TPC<T> diff(even.size());
    diff.zero();
    diff += even;
    diff -= odd;
    func_profiler.accumulate("maxpool-z-diff");

    // DRELU diff -> b
    func_profiler.start();
    TPC<U> b(even.size());
    dReLU(diff, b);
    func_profiler.accumulate("maxpool-z-drelu");
 
    // b * even + 1-b * odd
    selectShare(odd, even, b, even);

    func_profiler.start();
    //even *= b;
    //odd *= negated;
    //even += odd;

    result.zero();
    result += even;
    func_profiler.accumulate("maxpool-z-calc");

    // -- dMP --

    // expandCompare b -> expandedB
    func_profiler.start();
    TPC<U> negated(b.size());
    negated.fill(1);
    negated -= b;
    TPC<U> expandedB(input.size());
    gpu::expandCompare(*b.getShare(0), *negated.getShare(0), *expandedB.getShare(0));
    func_profiler.accumulate("maxpool-z-expandCompare");
 
    // dresult &= expandedB
    func_profiler.start();
    dresult &= expandedB;
    func_profiler.accumulate("maxpool-z-dcalc");
}

/*
template<typename T, typename I, typename I2, typename I3>
void localConvolution(TPC<T, I> &im, TPC<T, I2> &filters, DeviceData<T, I3> &out,
        size_t imageWidth, size_t imageHeight, size_t filterSize, size_t Din, size_t Dout,
        size_t stride, size_t padding) {

    size_t widthKernels = ((imageWidth - filterSize + (2*padding))/stride)+1;
    size_t heightKernels = ((imageHeight - filterSize + (2*padding))/stride)+1;
    int outSize = widthKernels * heightKernels * Dout;

    TPC<T> imRows((size_t)0);
    for(int share = 0; share < TPC<T>::numShares(); share++) {
        gpu::im2row(
            im.getShare(share),
            imRows.getShare(share),
            imageWidth, imageHeight, filterSize, Din, stride, padding
        );
    }

    //DeviceData<T> result(outSize);
    TPC<T> result(outSize);
    localMatMul(imRows, filters, result,
            widthKernels * heightKernels, Dout, Din * filterSize * filterSize,
            false, true);

    gpu::transpose(result.getShare(0), &out, widthKernels * heightKernels, Dout);
}
*/

template<typename T>
void localMatMul(const TPC<T> &a, const TPC<T> &b, TPC<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c) {
    
    TPC<T> x(a.size()), y(b.size()), z(c.size());

    int a_rows = transpose_a ? K : M; int a_cols = transpose_a ? M : K;
    int b_rows = transpose_b ? N : K; int b_cols = transpose_b ? K : N;
    PrecomputeObject.getMatrixBeaverTriple<T, TPC<T> >(x, y, z, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b);

    DeviceData<T> e(x.size()), f(y.size()), temp(z.size());

    x += a; y += b;
    reconstruct(x, e);
    reconstruct(y, f);
    x -= a; y -= b;

    c.zero();
    c += z;

    gpu::gemm(M, N, K, &e, transpose_a, &f, transpose_b, &temp, transpose_c);
    c += temp;
    temp.zero();

    gpu::gemm(M, N, K, &e, transpose_a, y.getShare(0), transpose_b, &temp, transpose_c);
    c -= temp;
    temp.zero();

    gpu::gemm(M, N, K, x.getShare(0), transpose_a, &f, transpose_b, &temp, transpose_c);
    c -= temp;
}

template<typename T, typename I, typename I2>
void carryOut(TPC<T, I> &p, TPC<T, I> &g, int k, TPC<T, I2> &out) {

    // get zip iterators on both p and g
    //  -> pEven, pOdd, gEven, gOdd
 
    int stride = 2;
    int offset = 1;

    using SRIterator = typename StridedRange<I>::iterator;

    StridedRange<I> pEven0Range(p.getShare(0)->begin(), p.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> pEven0(pEven0Range.begin(), pEven0Range.end());
    TPC<T, SRIterator> pEven(&pEven0);

    StridedRange<I> pOdd0Range(p.getShare(0)->begin() + offset, p.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> pOdd0(pOdd0Range.begin(), pOdd0Range.end());
    TPC<T, SRIterator> pOdd(&pOdd0);

    StridedRange<I> gEven0Range(g.getShare(0)->begin(), g.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> gEven0(gEven0Range.begin(), gEven0Range.end());
    TPC<T, SRIterator> gEven(&gEven0);

    StridedRange<I> gOdd0Range(g.getShare(0)->begin() + offset, g.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> gOdd0(gOdd0Range.begin(), gOdd0Range.end());
    TPC<T, SRIterator> gOdd(&gOdd0);

    while(k > 1) {

        // gTemp = pOdd & gEven
        //  store result in gEven
        gEven &= pOdd;

        // pEven & pOdd
        //  store result in pEven
        pEven &= pOdd;

        // gOdd ^ gTemp
        //  store result in gOdd
        gOdd ^= gEven;
     
        // regenerate zip iterators to p and g
     
        //  gOdd -> gEven, gOdd
        gEven0Range.set(g.getShare(0)->begin() + offset, g.getShare(0)->end(), stride*2);
        gEven0.set(gEven0Range.begin(), gEven0Range.end());
        gEven.set(&gEven0);

        offset += stride;

        gOdd0Range.set(g.getShare(0)->begin() + offset, g.getShare(0)->end(), stride*2);
        gOdd0.set(gOdd0Range.begin(), gOdd0Range.end());
        gOdd.set(&gOdd0);

        //  pEven -> pEven, pOdd
        stride *= 2;

        pEven0Range.set(p.getShare(0)->begin(), p.getShare(0)->end(), stride);
        pEven0.set(pEven0Range.begin(), pEven0Range.end());
        pEven.set(&pEven0);

        pOdd0Range.set(p.getShare(0)->begin() + stride/2, p.getShare(0)->end(), stride);
        pOdd0.set(pOdd0Range.begin(), pOdd0Range.end());
        pOdd.set(&pOdd0);
     
        k /= 2;
    }

    // copy output to destination
    // out.zip(gEven, gOdd);
    StridedRange<I> outputEven0Range(out.getShare(0)->begin(), out.getShare(0)->end(), 2);
    thrust::copy(gEven.getShare(0)->begin(), gEven.getShare(0)->end(), outputEven0Range.begin());

    StridedRange<I> outputOdd0Range(out.getShare(0)->begin() + 1, out.getShare(0)->end(), 2);
    thrust::copy(gOdd.getShare(0)->begin(), gOdd.getShare(0)->end(), outputOdd0Range.begin());
}

template<typename T, typename I, typename I2>
void getPowers(TPC<T, I> &in, DeviceData<T, I2> &pow) {

    TPC<T> powers(pow.size()); // accumulates largest power yet tested that is less than the input val
    TPC<T> currentPowerBit(in.size()); // current power
    TPC<T> diff(in.size());
    TPC<uint8_t> comparisons(in.size());

    for (int bit = 0; bit < sizeof(T) * 8; bit++) {
        currentPowerBit.fill(bit);

        diff.zero();
        diff += in;
        diff -= (((T)1) << bit);

        comparisons.zero();
        dReLU(diff, comparisons); // 0 -> current power is larger than input val, 1 -> input val is larger than current power

        // 0 -> keep val, 1 -> update to current known largest power less than input
        selectShare(powers, currentPowerBit, comparisons, powers);
    }

    reconstruct(powers, pow);
}

template<typename T, typename I, typename I2, typename Functor>
void taylorSeries(TPC<T, I> &in, TPC<T, I2> &out,
        double a0, double a1, double a2,
        Functor fn) {

    out.zero();
    TPC<T> scratch(out.size());

    DeviceData<T> pow(out.size());
    getPowers(in, pow);
    pow += 1;

    DeviceData<T> ones(pow.size());
    ones.fill(1);
    ones <<= pow;

    if (a2 != 0.0) {
        DeviceData<T> a2Coeff(out.size());
        thrust::transform(
            pow.begin(), pow.end(), a2Coeff.begin(),
            tofixed_variable_precision_functor<T>(a2));

        scratch.zero();
        scratch += in;
        scratch *= in;
        dividePublic(scratch, ones);

        scratch *= a2Coeff;
        dividePublic(scratch, ones);
        out += scratch;
    }

    if (a1 != 0.0) {

        DeviceData<T> a1Coeff(out.size());
        thrust::transform(
            pow.begin(), pow.end(), a1Coeff.begin(),
            tofixed_variable_precision_functor<T>(a1));

        scratch.zero();
        scratch += in;
        scratch *= a1Coeff;
        dividePublic(scratch, ones);

        out += scratch;
    }

    DeviceData<T> a0Coeff(out.size());
    thrust::transform(
        pow.begin(), pow.end(), a0Coeff.begin(),
        tofixed_variable_precision_functor<T>(a0));
    out += a0Coeff;

    DeviceData<T> powCoeff(out.size());
    thrust::transform(
        pow.begin(), pow.end(), powCoeff.begin(),
        calc_fn<T, Functor>(fn));
    out *= powCoeff;

    dividePublic(out, ones);

    // turn values back to base (e.g. 20 bit) precision

    pow -= FLOAT_PRECISION;

    DeviceData<T> positivePow(pow.size());
    thrust::transform(
        pow.begin(), pow.end(), positivePow.begin(),
        filter_positive_powers<T>());

    ones.fill(1);
    ones <<= positivePow;

    dividePublic(out, ones);

    DeviceData<T> negativePow(pow.size());
    thrust::transform(
        pow.begin(), pow.end(), negativePow.begin(),
        filter_negative_powers<T>());

    for (int share = 0; share < TPC<T>::numShares(); share++) {
        thrust::transform(
            out.getShare(share)->begin(), out.getShare(share)->end(), negativePow.begin(), out.getShare(share)->begin(),
            lshift_functor<T>()); 
    }
}

template<typename T, typename U, typename I, typename I2>
void convex_comb(TPC<T, I> &a, TPC<T, I> &c, DeviceData<U, I2> &b) {

    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(b.begin(), c.getShare(0)->begin(), a.getShare(0)->begin())),
        thrust::make_zip_iterator(thrust::make_tuple(b.end(), c.getShare(0)->end(), a.getShare(0)->end())),
        tpc_convex_comb_functor(partyNum)
    );
}


