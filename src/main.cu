#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

struct Options {
  std::string input_dir = "data/input";
  std::string output_dir = "output/filtered";
  int window = 7;
  int limit = 100;
  int generate = 100;
  int length = 2048;
};

struct Result {
  std::string input_file;
  std::string output_file;
  int samples = 0;
  float gpu_ms = 0.0f;
};

__global__ void MovingAverageFilter(const float *input, float *output, int n,
                                    int radius) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  int start = i - radius;
  int end = i + radius;
  if (start < 0) {
    start = 0;
  }
  if (end >= n) {
    end = n - 1;
  }

  float sum = 0.0f;
  for (int j = start; j <= end; ++j) {
    sum += input[j];
  }
  output[i] = sum / static_cast<float>(end - start + 1);
}

void CheckCuda(cudaError_t status, const char *message) {
  if (status != cudaSuccess) {
    std::cerr << message << ": " << cudaGetErrorString(status) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

void EnsureDirectory(const std::string &path) {
  if (path.empty()) {
    return;
  }

  for (size_t i = 1; i <= path.size(); ++i) {
    if (i != path.size() && path[i] != '/') {
      continue;
    }
    std::string part = path.substr(0, i);
    if (part.empty()) {
      continue;
    }
    if (mkdir(part.c_str(), 0755) != 0 && errno != EEXIST) {
      std::cerr << "Failed to create directory " << part << ": "
                << std::strerror(errno) << "\n";
      std::exit(EXIT_FAILURE);
    }
  }
}

std::string JoinPath(const std::string &left, const std::string &right) {
  if (left.empty()) {
    return right;
  }
  return left.back() == '/' ? left + right : left + "/" + right;
}

std::string BaseName(const std::string &path) {
  size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

bool EndsWithCsv(const std::string &name) {
  return name.size() >= 4 && name.substr(name.size() - 4) == ".csv";
}

std::vector<std::string> ListCsvFiles(const std::string &dir_path) {
  std::vector<std::string> files;
  DIR *dir = opendir(dir_path.c_str());
  if (dir == nullptr) {
    return files;
  }

  while (dirent *entry = readdir(dir)) {
    std::string name = entry->d_name;
    if (EndsWithCsv(name)) {
      files.push_back(JoinPath(dir_path, name));
    }
  }
  closedir(dir);
  std::sort(files.begin(), files.end());
  return files;
}

std::vector<float> ReadCsv(const std::string &path) {
  std::ifstream file(path);
  if (!file) {
    std::cerr << "Failed to read " << path << "\n";
    std::exit(EXIT_FAILURE);
  }

  std::vector<float> values;
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream row(line);
    std::string cell;
    while (std::getline(row, cell, ',')) {
      if (!cell.empty()) {
        values.push_back(std::stof(cell));
      }
    }
  }
  return values;
}

void WriteCsv(const std::string &path, const std::vector<float> &values) {
  std::ofstream file(path);
  if (!file) {
    std::cerr << "Failed to write " << path << "\n";
    std::exit(EXIT_FAILURE);
  }

  file << std::fixed << std::setprecision(6);
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      file << ",";
    }
    file << values[i];
  }
  file << "\n";
}

void GenerateSignals(const std::string &input_dir, int count, int length) {
  EnsureDirectory(input_dir);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> noise(-0.25f, 0.25f);

  for (int file_index = 0; file_index < count; ++file_index) {
    std::ostringstream name;
    name << "signal_" << std::setw(4) << std::setfill('0') << file_index
         << ".csv";

    std::vector<float> values(length);
    for (int i = 0; i < length; ++i) {
      float t = static_cast<float>(i) / 32.0f;
      values[i] = std::sin(t) + 0.5f * std::sin(t * 0.25f) + noise(rng);
    }
    WriteCsv(JoinPath(input_dir, name.str()), values);
  }
}

Result ProcessFile(const std::string &input_path, const std::string &output_dir,
                   int window) {
  std::vector<float> input = ReadCsv(input_path);
  if (input.empty()) {
    std::cerr << "Input file is empty: " << input_path << "\n";
    std::exit(EXIT_FAILURE);
  }

  int n = static_cast<int>(input.size());
  int radius = std::max(0, window / 2);
  std::vector<float> output(n);
  float *d_input = nullptr;
  float *d_output = nullptr;
  size_t bytes = sizeof(float) * input.size();

  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&d_input), bytes),
            "cudaMalloc input failed");
  CheckCuda(cudaMalloc(reinterpret_cast<void **>(&d_output), bytes),
            "cudaMalloc output failed");
  CheckCuda(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy host to device failed");

  cudaEvent_t start;
  cudaEvent_t stop;
  CheckCuda(cudaEventCreate(&start), "cudaEventCreate start failed");
  CheckCuda(cudaEventCreate(&stop), "cudaEventCreate stop failed");

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  CheckCuda(cudaEventRecord(start), "cudaEventRecord start failed");
  MovingAverageFilter<<<blocks, threads>>>(d_input, d_output, n, radius);
  CheckCuda(cudaGetLastError(), "MovingAverageFilter launch failed");
  CheckCuda(cudaEventRecord(stop), "cudaEventRecord stop failed");
  CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop failed");

  float gpu_ms = 0.0f;
  CheckCuda(cudaEventElapsedTime(&gpu_ms, start, stop),
            "cudaEventElapsedTime failed");
  CheckCuda(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy device to host failed");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_input);
  cudaFree(d_output);

  std::string output_path = JoinPath(output_dir, BaseName(input_path));
  WriteCsv(output_path, output);

  return {input_path, output_path, n, gpu_ms};
}

Options ParseArgs(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto require_value = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };

    if (arg == "--input") {
      options.input_dir = require_value("--input");
    } else if (arg == "--output") {
      options.output_dir = require_value("--output");
    } else if (arg == "--window") {
      options.window = std::stoi(require_value("--window"));
    } else if (arg == "--limit") {
      options.limit = std::stoi(require_value("--limit"));
    } else if (arg == "--generate") {
      options.generate = std::stoi(require_value("--generate"));
    } else if (arg == "--length") {
      options.length = std::stoi(require_value("--length"));
    } else if (arg == "--help") {
      std::cout << "Usage: ./batch_signal_filter --input data/input "
                   "--output output/filtered --window 7 --limit 100 "
                   "--generate 100 --length 2048\n";
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      std::exit(EXIT_FAILURE);
    }
  }

  if (options.window < 1 || options.limit < 1 || options.length < 1 ||
      options.generate < 0) {
    std::cerr << "window, limit, and length must be positive; generate must be "
                 "zero or greater\n";
    std::exit(EXIT_FAILURE);
  }
  return options;
}

void WriteSummary(const std::vector<Result> &results, int window) {
  EnsureDirectory("output");
  std::ofstream summary("output/summary.csv");
  if (!summary) {
    std::cerr << "Failed to write output/summary.csv\n";
    std::exit(EXIT_FAILURE);
  }

  summary << "input_file,output_file,samples,window,gpu_ms\n";
  for (const Result &result : results) {
    summary << result.input_file << "," << result.output_file << ","
            << result.samples << "," << window << "," << std::fixed
            << std::setprecision(4) << result.gpu_ms << "\n";
  }
}

int main(int argc, char **argv) {
  Options options = ParseArgs(argc, argv);
  EnsureDirectory(options.input_dir);
  EnsureDirectory(options.output_dir);

  std::vector<std::string> files = ListCsvFiles(options.input_dir);
  if (files.empty() && options.generate > 0) {
    std::cout << "Generating " << options.generate << " input signals with "
              << options.length << " samples each.\n";
    GenerateSignals(options.input_dir, options.generate, options.length);
    files = ListCsvFiles(options.input_dir);
  }

  if (files.empty()) {
    std::cerr << "No input CSV files found in " << options.input_dir << "\n";
    return EXIT_FAILURE;
  }

  int processed = 0;
  std::vector<Result> results;
  for (const std::string &file : files) {
    if (processed >= options.limit) {
      break;
    }
    Result result = ProcessFile(file, options.output_dir, options.window);
    std::cout << "processed=" << BaseName(result.input_file)
              << " samples=" << result.samples << " gpu_ms=" << std::fixed
              << std::setprecision(4) << result.gpu_ms << "\n";
    results.push_back(result);
    ++processed;
  }

  WriteSummary(results, options.window);
  std::cout << "Processed " << results.size()
            << " signal files using a CUDA moving-average filter.\n";
  std::cout << "Summary: output/summary.csv\n";
  std::cout << "Filtered signals: " << options.output_dir << "\n";
  return EXIT_SUCCESS;
}
