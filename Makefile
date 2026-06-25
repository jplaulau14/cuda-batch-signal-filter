NVCC ?= nvcc
CXXFLAGS ?= -std=c++17 -O2

.PHONY: clean build run

build: src/main.cu
	$(NVCC) $(CXXFLAGS) src/main.cu -o batch_signal_filter

run: build
	./batch_signal_filter --input data/input --output output/filtered --window 7 --limit 100 --generate 100 --length 2048

clean:
	rm -f batch_signal_filter
	rm -f output/run.log output/summary.csv output/filtered/*.csv
