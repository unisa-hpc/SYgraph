#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

constexpr size_t NUM_ELEMS = 128;

int main() {
  sycl::queue q {sycl::gpu_selector_v};
  
  sygraph::frontier::Frontier<size_t> f {q, NUM_ELEMS};
  auto n = f.get_num_active_elements();
  using type = typename sygraph::frontier::Frontier<size_t>::bitmap_type;
  assert(n == 0);
  assert(sizeof(type) * 8 == f.get_bitmap_range());
  assert(f.get_num_elems() == NUM_ELEMS);

  q.submit([&](sycl::handler& cgh) {
    auto bitmap = f.get_device_frontier();
    cgh.parallel_for(sycl::range<1>{f.get_num_elems()}, [=](sycl::id<1> idx) {
      bitmap.set_on(idx);
    });
  }).wait();
  
  n = f.get_num_active_elements();
  assert(n == NUM_ELEMS);

  f.remove(0);
  n = f.get_num_active_elements();
  assert(n == NUM_ELEMS - 1);
}