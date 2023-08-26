//
//#include "unitTests.h"
//
//template<typename T>
//struct DeviceDataTest : public testing::Test {
//    using ParamType = T;
//};
//
//TYPED_TEST_CASE(DeviceDataTest, uint64_t);
//
//TYPED_TEST(DeviceDataTest, DeviceData) {
//
//    using T = typename TestFixture::ParamType;
//
//    DeviceData<T> d1 = {1, 2, 3};
//    DeviceData<T> d2 = {1, 1, 1};
//
//    d1 += d2;
//
//    std::vector<double> expected = {2, 3, 4};
//    assertDeviceData(d1, expected, false);
//}
//
//
//
//TYPED_TEST(DeviceDataTest, DeviceDataSin) {
//
//    using T = typename TestFixture::ParamType;
//
//    DeviceData<T> d1 = {65536,  65536, 131072, 524288, 799539,  65536};
//    DeviceData<T> s(12);
//    size_t m=5;
//    size_t rows = 6;
//    size_t cols = 2;
//    gpu::sin2kpix(d1, s, m+FLOAT_PRECISION, rows, cols);
//
//    for (auto iter=s.begin();iter!=s.end();++iter)
//    {
//        std::cout<< ((int64_t) *iter) * 1.0 / (1<<FLOAT_PRECISION)<<std::endl;
//    }
//        std::vector<double> expected = { 1.95090322e-01,  3.82683432e-01,
// 1.95090322e-01,  3.82683432e-01,
// 3.82683432e-01,  7.07106781e-01,
// 1.00000000e+00,  1.22464680e-16,
// 6.78800746e-01, -9.96917334e-01,
// 1.95090322e-01,  3.82683432e-01};
//    assertDeviceData(s, expected, false);
//}
//
//
//
//
//
//template<typename T>
//using VIterator = thrust::detail::normal_iterator<thrust::device_ptr<T> >;
//
//template<typename T>
//using TIterator = thrust::transform_iterator<thrust::negate<T>, VIterator<T> >;
//
//TYPED_TEST(DeviceDataTest, DeviceDataView) {
//
//    using T = typename TestFixture::ParamType;
//
//    DeviceData<T> d1 = {1, 2, 3};
//    DeviceData<T, TIterator<T> > negated(
//        thrust::make_transform_iterator(d1.begin(), thrust::negate<T>()),
//        thrust::make_transform_iterator(d1.end(), thrust::negate<T>())
//    );
//
//    d1 += negated;
//
//    std::vector<double> expected = {0, 0, 0};
//    assertDeviceData(d1, expected, false);
//}
//
