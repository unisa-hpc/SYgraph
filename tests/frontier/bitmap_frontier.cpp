#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

constexpr size_t NUM_ELEMS = 128;

int main() {
  sycl::queue q {sycl::gpu_selector_v};
  
  sygraph::frontier::Frontier<size_t> f {q, NUM_ELEMS};
  auto n = f.getNumActiveElements();
  using type = typename sygraph::frontier::Frontier<size_t>::bitmap_type;
  assert(n == 0);
  assert(sizeof(type) * 8 == f.getBitmapRange());
  assert(f.getNumElems() == NUM_ELEMS);

  q.submit([&](sycl::handler& cgh) {
    auto bitmap = f.getDeviceBitmap();
    cgh.parallel_for(sycl::range<1>{f.getNumElems()}, [=](sycl::id<1> idx) {
      bitmap.setOn(idx);
    });
  }).wait();
  
  n = f.getNumActiveElements();
  assert(n == NUM_ELEMS);

  f.remove(0);
  n = f.getNumActiveElements();
  assert(n == NUM_ELEMS - 1);
}