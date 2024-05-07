#include <string>
#include <iostream>
#include <random>


struct args_t {
  bool print_output = false;
  bool validate = false;
  std::string path;
  bool random_source = true;
  uint source;

  args_t(int argc, char** argv) {
    if (argc < 2) {
      std::cerr << "Usage: " << argv[0] << " <path-to-graph>" << std::endl;
      exit(1);
    } else {
      path = argv[1];
    }
    for (int i = 2; i < argc; i++) {
      if (std::string(argv[i]) == "-p") {
        print_output = true;
      } else if (std::string(argv[i]) == "-v") {
        validate = true;
      } else if (std::string(argv[i]) == "-s") {
        random_source = false;
        try {
          source = std::stoi(argv[i + 1]);
        } catch (std::exception& e) {
          std::cerr << "Error: -s flag requires an integer argument" << std::endl;
          exit(1);
        }
      }
    }
  }
};

template<typename T>
void PRINT_FRONTIER(T& f, std::string prefix = "") {
  using type_t = typename T::type_t;
  auto size = f.get_bitmap_size() * f.get_bitmap_range();
  std::cout << prefix;
  for (int i = size - 1; i >= 0; --i) {
    std::cout << (f.check(static_cast<type_t>(i)) ? "1" : "0");
  }
  std::cout << " [" << f.get_device_frontier().get_data()[0] << "]" << std::endl; 
  std::cout << std::endl;
}

uint get_random_source(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, size - 1);
  return dis(gen);
}