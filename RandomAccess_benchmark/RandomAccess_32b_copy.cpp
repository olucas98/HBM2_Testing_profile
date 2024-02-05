// Owen Lucas

//Mirror the testing done for the RandomAcess benchmark
//Try to do the fancy technique from the OpenCL github


#include <omp.h>			// Included for timing
#include <string>			
#include <iostream>
#include <bits/stdc++.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <random>

#include <sycl/ext/intel/fpga_extensions.hpp>
#include "pipe_utils.hpp"
#include "unrolled_loop.hpp"
#include <CL/sycl.hpp>
#include <oneapi/dpl/random>
using namespace cl::sycl;


#include "dpc_common.hpp"

// M is number of items in table
#define M 0x1000000ULL
//L is the number of random data points
#define L 0x80000000
//BANKS is how many banks to use for the test (not including the one for the input)
#define BANKS 32

#define MIXER 0xDEADBEEFBEEFDEAD

#define NSEC_IN_SEC 1000000000.0
// 64 - log2(M*BANKS) = 64 - 29
#define ADDR_SHIFT 35

#define CONCURRENT_GEN 32
#define CONCURRENT_GEN_LOG 5
#define BLOCK_SIZE_LOG 3
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG)
#define SHIFT_GAP 5
#define data_chunk 0x1000000
#define m (M << 5)
#define BUFFER_SIZE 1
#define POLY 7

struct InputItem {
	uint32_t addr; //what bit
	uint8_t bank; //what bank
};

struct BankAdd {
	uint32_t addr; //what bit
	bool done = false;
};

class test_kernel;

template<int ID>
class bank_kernel;

using DEVICE_DATA_TYPE_UNSIGNED = uint64_t;
using HOST_DATA_TYPE = uint64_t;
using HOST_DATA_TYPE_SIGNED = int64_t;
using DEVICE_DATA_TYPE = int64_t;
 

int main (int argc, char *argv[]) {
    
    
    // Select either:
    //  - the FPGA emulator device (CPU emulation of the FPGA)
    //  - the FPGA device (a real FPGA)
    #if defined(FPGA_EMULATOR)
        auto selector = sycl::ext::intel::fpga_emulator_selector_v;
    #else
        auto selector = sycl::ext::intel::fpga_selector_v;
    #endif
    
    try{
		
		//printf("\nInput data uses %f MB per bank\n", (sizeof(InputItem) * L)*0.000001);
		
		printf("\nEach bank gets %f  MB for the Bloom Filter\n", M * sizeof(uint64_t) * 0.000001);
        

		cl::sycl::property_list propList{cl::sycl::property::queue::enable_profiling()};
        sycl::queue q(selector, dpc_common::exception_handler, propList);
        //printf("Queue has been made\n");

        // Print out the device information.
        std::cout << "Running on device: "
                  << q.get_device().get_info<info::device::name>() << "\n";
				  
		uint64_t* out_matrix[BANKS];
		uint64_t* host_copy =(uint64_t*)aligned_alloc(64, sizeof(uint64_t) * M * BANKS);
		uint64_t* start_copy[BANKS]; //see how many of the item have changed
		
		for (int j = 0; j < BANKS; j++){
			out_matrix[j] = (uint64_t*)aligned_alloc(64, sizeof(uint64_t) * M);
			//host_copy[j] = (uint64_t*)aligned_alloc(64, sizeof(uint64_t) * M);
			start_copy[j] = (uint64_t*)aligned_alloc(64, sizeof(uint64_t) * M);
		}
		
		for (int j = 0; j < BANKS; j++){
			for(int k = 0; k < M; k++){
				host_copy[j*M + k] = out_matrix[j][k];
				start_copy[j][k] = out_matrix[j][k];
			}
		}
		
		HOST_DATA_TYPE* random_inits;
        posix_memalign(reinterpret_cast<void**>(&random_inits), 4096, sizeof(HOST_DATA_TYPE)* 32);
        HOST_DATA_TYPE chunk = m * 4 / std::min((unsigned long long)32, m * 4);
        HOST_DATA_TYPE ran = 1;
        random_inits[0] = ran;
        for (HOST_DATA_TYPE r=0; r < 32; r++) {
            for (HOST_DATA_TYPE run = 0; run < chunk; run++) {
                HOST_DATA_TYPE_SIGNED v = 0;
                if (((HOST_DATA_TYPE_SIGNED) ran) < 0) {
                    v = POLY;
                }
                ran = (ran << 1) ^v;
            }
            random_inits[r + 1] = ran;
        }
		
        
        sycl::range<1> data_range{(size_t)M};
		event bank_events[BANKS];
		
		{
			std::vector<sycl::buffer<uint64_t, 1> *> bank_bufs;
            fpga_tools::UnrolledLoop<BANKS>([&](auto i) {
                __declspec( align(64) ) sycl::buffer<uint64_t,1> *bank_buffer = new sycl::buffer<uint64_t, 1>(out_matrix[i], data_range);
                bank_bufs.push_back(bank_buffer);
            });


			fpga_tools::UnrolledLoop<BANKS>([&](auto kernel_number) {
				bank_events[kernel_number] = q.submit([&](handler &h) {
					accessor data(bank_bufs[kernel_number][0], h, read_write, ext::oneapi::accessor_property_list{sycl::ext::oneapi::no_alias, ext::intel::buffer_location<kernel_number>});
					h.single_task<bank_kernel<kernel_number>>([=]() {
						DEVICE_DATA_TYPE_UNSIGNED ran_initials[CONCURRENT_GEN/BLOCK_SIZE][BLOCK_SIZE];
						// Load RNG initial values from global memory
						for (int r = 0; r < CONCURRENT_GEN/BLOCK_SIZE; r++) {
							DEVICE_DATA_TYPE_UNSIGNED tmp[BLOCK_SIZE];
							// __attribute__((opencl_unroll_hint(BLOCK_SIZE)))
							#pragma unroll
							for (int b = 0; b < BLOCK_SIZE; b++) {
								tmp[b] = random_inits[r* BLOCK_SIZE + b];
							}
							//__attribute__((opencl_unroll_hint(BLOCK_SIZE)))
							#pragma unroll
							for (int b = 0; b < BLOCK_SIZE; b++) {
								ran_initials[r][b] = tmp[b];
							}
						}
						DEVICE_DATA_TYPE_UNSIGNED ran[CONCURRENT_GEN];
						DEVICE_DATA_TYPE_UNSIGNED number_count[CONCURRENT_GEN];
						//__attribute__((opencl_unroll_hint(CONCURRENT_GEN)))
						#pragma unroll
						for (int r = 0; r < CONCURRENT_GEN; r++) {
							number_count[r] = 0;
							ran[r] = ran_initials[r >> BLOCK_SIZE_LOG][ r & (BLOCK_SIZE - 1)];
						}

						// Initialize shift register
						// this is the data shift register that contains the random numbers
						DEVICE_DATA_TYPE_UNSIGNED random_number_shift[(CONCURRENT_GEN + 1) * SHIFT_GAP + 1];
						// these two shift registers contain a valid bit and a complete bit
						// the valid bit is set, if the current random number is valid and in the range of the current kernel
						// the complete bit is only set if all random number generators before the current one have completed execution
						bool random_number_valid[(CONCURRENT_GEN + 1) * SHIFT_GAP + 1];
						bool random_number_done_shift[(CONCURRENT_GEN + 1) * SHIFT_GAP + 1];

						//__attribute__((opencl_unroll_hint((CONCURRENT_GEN + 1) * SHIFT_GAP + 1)))
						#pragma unroll
						for (int r = 0; r < (CONCURRENT_GEN + 1) * SHIFT_GAP + 1; r++) {
							random_number_shift[r] = 0;
							random_number_done_shift[r] = false;
							random_number_valid[r] = false;
						}

						// calculate the start of the address range this kernel is responsible for
					
						DEVICE_DATA_TYPE_UNSIGNED const address_start = kernel_number * data_chunk;

						DEVICE_DATA_TYPE_UNSIGNED const mupdate = 4 * m;
						bool done = false;
						
						[[intel::ivdep]]
						while (!done) {
							DEVICE_DATA_TYPE_UNSIGNED local_address_buffer[BUFFER_SIZE];
							DEVICE_DATA_TYPE_UNSIGNED loaded_data_buffer[BUFFER_SIZE];
							
							#pragma unroll
							for (uint i = 0; i < 2 * BUFFER_SIZE; i++) {
								if (i < BUFFER_SIZE) {
									#pragma unroll
									for (int r=0; r < CONCURRENT_GEN; r++) {
										DEVICE_DATA_TYPE_UNSIGNED total_updates = (mupdate >> CONCURRENT_GEN_LOG) + ((r < (mupdate & (CONCURRENT_GEN - 1)) ? 1 : 0));
										number_count[r] = !random_number_valid[(r + 1) * SHIFT_GAP] ? number_count[r] + 1 : number_count[r];
										bool is_inrange = false;
										if (!random_number_valid[(r + 1) * SHIFT_GAP] && number_count[r] <= total_updates) {
											DEVICE_DATA_TYPE_UNSIGNED v = ((DEVICE_DATA_TYPE) ran[r] < 0) ? POLY : 0UL;
											ran[r] = (ran[r] << 1) ^ v;
											DEVICE_DATA_TYPE_UNSIGNED address = (ran[r] >> 3) & (m - 1);
											#ifndef SINGLE_KERNEL
											DEVICE_DATA_TYPE_UNSIGNED local_address = address - address_start;
											#else
											DEVICE_DATA_TYPE_UNSIGNED local_address = address;
											#endif
											is_inrange = (local_address < data_chunk);
											random_number_shift[(r + 1) * SHIFT_GAP] = ran[r];
										}
										random_number_valid[(r + 1) * SHIFT_GAP] = (random_number_valid[(r + 1) * SHIFT_GAP] || is_inrange);
										random_number_done_shift[(r + 1) * SHIFT_GAP] = (number_count[r] >= total_updates && (random_number_done_shift[(r + 1) * SHIFT_GAP] || r == CONCURRENT_GEN - 1));
									}
									DEVICE_DATA_TYPE_UNSIGNED random_number = random_number_shift[0];
									bool valid = random_number_valid[0];
									done = random_number_done_shift[0];
									DEVICE_DATA_TYPE_UNSIGNED address = (random_number >> 3) & (m - 1);
									DEVICE_DATA_TYPE_UNSIGNED local_address = address - address_start;
									
									local_address_buffer[i] = local_address;
									
									if (valid) {
										loaded_data_buffer[i] = data[local_address] ^ random_number;
									}
									#pragma unroll
									for (int r = 0; r < (CONCURRENT_GEN + 1) * SHIFT_GAP; r++) {
										random_number_shift[r] = random_number_shift[r + 1];
										random_number_done_shift[r] = random_number_done_shift[r + 1];
										random_number_valid[r] = random_number_valid[r + 1];
									}
									// Set the last value in the shift register to invalid so the RNGs can update it
									random_number_valid[(CONCURRENT_GEN + 1) * SHIFT_GAP] = false;
									random_number_done_shift[(CONCURRENT_GEN + 1) * SHIFT_GAP] = false;
								}
								else{
									 DEVICE_DATA_TYPE_UNSIGNED local_address = local_address_buffer[i - BUFFER_SIZE];
									if (local_address < data_chunk) {
										data[local_address] = loaded_data_buffer[i - BUFFER_SIZE];
									}
								}
							}
						}
						
					});
				});
			});

			q.wait();
			for(int j =0; j < BANKS; j++){
				delete bank_bufs[j];
			}
		}
		
		
		printf("\nAll kernels finished\n");
		
		//check results on host
		
		HOST_DATA_TYPE temp = 1;
        for (HOST_DATA_TYPE i=0; i < 4L * m; i++) {
            HOST_DATA_TYPE_SIGNED v = 0;
            if (((HOST_DATA_TYPE_SIGNED)temp) < 0) {
                v = POLY;
            }
            temp = (temp << 1) ^ v;
            host_copy[(temp >> 3) & (m - 1)] ^= temp;
        }
		
		
		uint64_t correct = 0;
		uint64_t same = 0;
		
		for (int j = 0; j < BANKS; j++){
			uint64_t bank_correct = 0;
			uint64_t bank_same = 0;
			for(int k = 0; k < M; k++){
				if (host_copy[j * M + k] == out_matrix[j][k]){
					correct++;
					bank_correct++;
				}
				if (start_copy[j][k] == out_matrix[j][k]){
					same++;
					bank_same++;
				}
			}
			printf("\nBank %d: %f %% correct, %f unchanged", j, (bank_correct / (double)M) * 100, (bank_same / (double)M) * 100);
		}
		
		event start_event = bank_events[0];
		event end_event = bank_events[0];
		
		
		for(uint32_t j = 1; j < BANKS; j++){
			if ( bank_events[j].get_profiling_info<sycl::info::event_profiling::command_start>() < start_event.get_profiling_info<sycl::info::event_profiling::command_start>()){
				start_event = bank_events[j];
			}
			if ( bank_events[j].get_profiling_info<sycl::info::event_profiling::command_end>() > end_event.get_profiling_info<sycl::info::event_profiling::command_end>()){
				end_event = bank_events[j];
			}
		}
		
		uint64_t start = start_event.get_profiling_info<sycl::info::event_profiling::command_start>();
		uint64_t end = end_event.get_profiling_info<sycl::info::event_profiling::command_end>();
		
		
		double duration = static_cast<double>(end - start) / NSEC_IN_SEC;
		std::cout << "\nKernel execution time: " << duration << " sec" << std::endl;
		std::cout << "\nMillions of updates per second: " << ((L/1000000.0) / duration) << " MUPS" << std::endl;
		std::cout << "\nMillions of updates per bank per second: " << (((L)/1000000.0) / duration) / BANKS << " MUPS" << std::endl;
		std::cout << "\nPercent agreement between host and device: " << (correct / ((double)M * BANKS)) * 100 << "%, error rate : " << ((M * BANKS) - correct / ((double)M * BANKS)) * 100 << "%" <<std::endl;
		std::cout << "\nPercent of items that were unchanged: " << (same / ((double)M * BANKS)) * 100 << "%" << std::endl;

        printf("\n");
		
		for (int j = 0; j < BANKS; j++){
			free(out_matrix[j]);
			//free(host_copy[j]);
			free(start_copy[j]);
		}
		free(host_copy);

        return 0;
    }
    catch (sycl::exception const& e) {
		// Catches exceptions in the host code
		std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

		// Most likely the runtime couldn't find FPGA hardware!
		if (e.code().value() == CL_DEVICE_NOT_FOUND) {
		  std::cerr << "If you are targeting an FPGA, please ensure that your "
					   "system has a correctly configured FPGA board.\n";
		  std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
		  std::cerr << "If you are targeting the FPGA emulator, compile with "
					   "-DFPGA_EMULATOR.\n";
		}
		std::terminate();
	}
}


