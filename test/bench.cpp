#include <filesystem>

#include "fmt/color.h"
#include "fmt/format.h"
#include "fmt/ostream.h"

#include "bench.hpp"

void run_bench(const std::string& name) {
  namespace fs = std::filesystem;
  fs::create_directory("bench_out");
  std::vector<std::string> arg_str = {
      name,                                                  //
      "--benchmark_format=json",                             //
      fmt::format("--benchmark_out=bench_out/{}.json", name) //
  };

  std::vector<char*> argv;
  argv.reserve(arg_str.size());

  for (auto& arg : arg_str) {
    argv.push_back(arg.data());
    fmt::print("{}\n", arg);
  }
  int argc = static_cast<int>(argv.size());

  fmt::print("----------------------------------------"
             "----------------------------------------\n");
  benchmark::Initialize(&argc, argv.data());
  if (benchmark::ReportUnrecognizedArguments(argc, argv.data())) {
    std::exit(1);
  }
  std::ofstream json_file(fmt::format("bench_out/{}.json", name));
  benchmark::JSONReporter json;
  json.SetOutputStream(&json_file);
  json.SetErrorStream(&std::cerr);

  benchmark::ConsoleReporter console;
  console.SetOutputStream(&std::cout);
  console.SetErrorStream(&std::cerr);

  benchmark::RunSpecifiedBenchmarks(&console, &json);
}
