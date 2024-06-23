#include "cute/algorithm/axpby.hpp"
#include "cute/algorithm/copy.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/config.hpp"
#include "cute/int_tuple.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/pointer.hpp"
#include "cute/util/debug.hpp"
#include <cstdlib>
#include <cute/tensor.hpp>
#include <cutlass/util/helper_cuda.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <
    typename ProblemShape, typename CTATiler, typename TA, typename AStride,
    typename ASmemLayout, typename AThreadLayout, typename TB, typename BStride,
    typename BSmemLayout, typename BThreadLayout, typename TC, typename CStride,
    typename CSmemLayout, typename CThreadLayout, typename Alpha, typename Beta>
__global__ static __launch_bounds__(decltype(cute::size(
    CThreadLayout{}))::value) void gemm_device(ProblemShape shape_MNK,
                                               CTATiler tiler, const TA *A,
                                               AStride dA,
                                               ASmemLayout sA_layout,
                                               AThreadLayout tA, const TB *B,
                                               BStride dB,
                                               BSmemLayout sB_layout,
                                               BThreadLayout tB, TC *C,
                                               CStride dC, CSmemLayout,
                                               CThreadLayout tC, Alpha alpha,
                                               Beta beta) {
  using namespace cute;

  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == _3{});
  CUTE_STATIC_ASSERT_V(rank(tiler) == _3{});

  static_assert(is_static<AThreadLayout>::value);
  static_assert(is_static<BThreadLayout>::value);
  static_assert(is_static<CThreadLayout>::value);

  CUTE_STATIC_ASSERT_V(size(tA) == size(tB));
  CUTE_STATIC_ASSERT_V(size(tC) == size(tB));

  // tiler: M, N, K
  // tA: M, K
  // tB: N, K
  // tC: M, N
  CUTE_STATIC_ASSERT_V(size<0>(tiler) % size<0>(tA) == _0{});
  CUTE_STATIC_ASSERT_V(size<2>(tiler) % size<1>(tA) == _0{});
  CUTE_STATIC_ASSERT_V(size<1>(tiler) % size<0>(tB) == _0{});
  CUTE_STATIC_ASSERT_V(size<2>(tiler) % size<1>(tB) == _0{});
  CUTE_STATIC_ASSERT_V(size<0>(tiler) % size<0>(tC) == _0{});
  CUTE_STATIC_ASSERT_V(size<1>(tiler) % size<1>(tC) == _0{});

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<0>(tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

  // M, N, K
  auto mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // M, K
  auto mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // N, K
  auto mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // M, N

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  // TODO: what is tiler?
  // a tiler can be considered as the shape of each tile
  auto gA = local_tile(mA, tiler, cta_coord, Step<_1, X, _1>{});
  auto gB = local_tile(mB, tiler, cta_coord, Step<X, _1, _1>{});
  auto gC = local_tile(mC, tiler, cta_coord, Step<_1, _1, X>{});

  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  auto sA = make_tensor(make_smem_ptr(smemA), sA_layout);
  auto sB = make_tensor(make_smem_ptr(smemB), sB_layout);

  auto tAgA = local_partition(gA, tA, threadIdx.x);
  auto tAsA = local_partition(sA, tA, threadIdx.x);

  auto tBgB = local_partition(gB, tB, threadIdx.x);
  auto tBsB = local_partition(sB, tB, threadIdx.x);

  CUTE_STATIC_ASSERT_V(size<0>(tAgA) == size<0>(tAsA));
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));
  CUTE_STATIC_ASSERT_V(size<0>(tBgB) == size<0>(tBsB));
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));

  auto tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
  auto tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{});
  auto tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});

  auto tCrC = make_tensor_like(tCgC);

  CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCgC));
  CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCsA));
  CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<1>(tCgC));
  CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<0>(tCsB));
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCsB));

  clear(tCrC);

  if (thread0()) {
    cute::print(size<2>(tAgA));
    cute::print("\n");
  }
  for (int k_tile = 0; k_tile < size<2>(tAgA); ++k_tile) {
    if (thread0()) {
      cute::print(k_tile);
      cute::print("\n");
    }
    copy(tAgA(_, _, k_tile), tAsA);
    copy(tBgB(_, _, k_tile), tBsB);

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    gemm(tCsA, tCsB, tCrC);

    __syncthreads();
  }

  axpby(alpha, tCrC, beta, tCgC);
}

template <typename TA, typename TB, typename TC, typename Alpha, typename Beta>
void gemm_nt(int m, int n, int k, Alpha alpha, const TA *A, int ldA,
             const TB *B, int ldB, Beta beta, TC *C, int ldC,
             cudaStream_t stream = nullptr) {
  using namespace cute;

  auto M = m;
  auto N = n;
  auto K = k;
  auto prob_shape = make_shape(M, N, K);

  auto dA = make_stride(_1{}, ldA);
  auto dB = make_stride(_1{}, ldB);
  auto dC = make_stride(_1{}, ldC);

  auto bM = _128{};
  auto bN = _128{};
  auto bK = _8{};
  auto cta_tiler = make_shape(bM, bN, bK);

  auto sA = make_layout(make_shape(bM, bK));
  auto sB = make_layout(make_shape(bN, bK));
  auto sC = make_layout(make_shape(bM, bN));

  auto tA = make_layout(make_shape(_32{}, _8{}));
  auto tB = make_layout(make_shape(_32{}, _8{}));
  auto tC = make_layout(make_shape(_16{}, _16{}));

  dim3 block_size(size(tC));
  dim3 grid_size(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

  gemm_device<<<grid_size, block_size, 0, stream>>>(prob_shape, cta_tiler, A,
                                                    dA, sA, tA, B, dB, sB, tB,
                                                    C, dC, sC, tC, alpha, beta);
}

template <typename TA, typename TB, typename TC, typename Alpha, typename Beta>
void gemm(int m, int n, int k, Alpha alpha, const TA *A, int ldA, const TB *B,
          int ldB, Beta beta, TC *C, int ldC, cudaStream_t stream = nullptr) {
  return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
}

int main() {
  int m = 5120;
  int n = 5120;
  int k = 4096;

  using TA = float;
  using TB = float;
  using TC = float;
  using TI = float;

  TI alpha = 1.0;
  TI beta = 1.0;

  cute::device_init(0);

  thrust::host_vector<TA> h_A(m * k);
  thrust::host_vector<TB> h_B(n * k);
  thrust::host_vector<TC> h_C(m * n);

  for (int i = 0; i < m * k; ++i) {
    h_A[i] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
  }

  for (int i = 0; i < n * k; ++i) {
    h_B[i] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
  }

  for (int i = 0; i < m * n; ++i) {
    h_C[i] = static_cast<TC>(-1);
  }

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  // gemm_nt
  int ldA = m;
  int ldB = k;
  int ldC = m;

  d_C = h_C;
  gemm(m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, beta,
       d_C.data().get(), ldC);
  CUTE_CHECK_LAST();
}