#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cinttypes>
#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <cuda_runtime.h>

#include "primes.h"

using namespace std::chrono;

__global__ void 
__launch_bounds__(BLOCK_SIZE)
unmark_multiples_naive(uint32_t primeCount, uint32_t *primes, uint64_t halfSize, uint32_t sizeSqrt, sieve_t *sieve)
{
    // one warp is comprised of 32 consecutive threads that mostly act in lockstep (similar to a CPU vector unit)
    const uint32_t warpIdx = threadIdx.x >> LOG2_WARP_SIZE;
    // in the context of a warp, threads are called lanes, i.e. threadIdx.x == warpIdx * 32 + laneIdx
    const uint32_t laneIdx = threadIdx.x & LANE_MASK;
    const uint32_t warpsPerBlock = blockDim.x >> LOG2_WARP_SIZE;
    const uint32_t warpsPerGrid = gridDim.x * warpsPerBlock;
    
    // parallelize among warps s.t. primes from global memory can be broadcast among lanes
    for (uint32_t primeIndex = warpsPerBlock * blockIdx.x + warpIdx; primeIndex < primeCount; primeIndex += warpsPerGrid) 
    {
        const uint32_t prime = primes[primeIndex];
        const uint64_t primeSquared = uint64_t(prime) * prime;

        // Unmark multiples starting at just beyond the square root of the sieve size or the square of the prime, 
        //   whichever is larger.
        uint64_t firstUnmarked = primeSquared > sizeSqrt ? primeSquared : ((sizeSqrt / prime + 1) * prime);
        // We're marking off odd multiples only, so make sure we start with one of those!
        if (!(firstUnmarked & 1))
            firstUnmarked += prime;

        // parallelize among lanes
        const uint64_t stride = uint64_t(prime) << 5;
        for (uint64_t index = (firstUnmarked >> 1) + uint64_t(laneIdx) * prime; index < halfSize; index += stride) 
            // Clear the bit in the word that corresponds to the last part of the index 
            atomicAnd(&sieve[WORD_INDEX(index)], ~(sieve_t(1) << BIT_INDEX(index)));
    }
}

__global__ void
__launch_bounds__(BLOCK_SIZE)
unmark_multiples_tiled(uint32_t primeCount, uint32_t *primes, uint64_t halfSize, uint32_t sizeSqrt, uint64_t blockSize, sieve_t *sieve, uint32_t shBlockWords)
{
    const uint32_t warpIdx = threadIdx.x >> LOG2_WARP_SIZE;
    const uint32_t laneIdx = threadIdx.x & LANE_MASK;
    const uint32_t warpsPerBlock = blockDim.x >> LOG2_WARP_SIZE;

    // Calculate the start and end of the block we need to work on, at buffer word boundaries. 
    //   Note that the first variable is a number in sieve space...
    const uint64_t blockStart = (uint64_t(blockIdx.x) * blockSize + sizeSqrt) & SIEVE_WORD_MASK;
    const uint64_t nextBlockStart = ((blockStart + blockSize) & SIEVE_WORD_MASK);
    //   ...and the second is an index in the sieve buffer (representing odd numbers only)
    const uint64_t endIndex = (blockIdx.x == gridDim.x - 1) ? halfSize
                                                            : min(halfSize, nextBlockStart >> 1);

    // "local" shared memory (shared between threads of the same block)
    // allows for faster atomics!
    extern __shared__ sieve_t localSieve[];

    const uint64_t stride = shBlockWords << (WORD_SHIFT + 1);
    const uint64_t finish = (endIndex << 1) - 1;

    #ifdef DEBUG_GPU
    if (threadIdx.x == 0)
        printf("  - block %d: blockStart = %" PRIu64 " (index %" PRIu64 "), finish = %" PRIu64 " (endIndex %" PRIu64 ").\n", blockIdx.x, blockStart, (blockStart >> 1), finish, endIndex);
    #endif

    for (uint64_t start = blockStart; start < finish; start += stride)
    {
        const uint64_t tileEndIndex = min(endIndex, (start + stride) >> 1); 
        const uint32_t effShBlockWords = min(shBlockWords,
                                             uint32_t(WORD_INDEX(tileEndIndex - (start >> 1) + BITS_PER_WORD - 1)));
        for (uint32_t localIdx = threadIdx.x; localIdx < effShBlockWords; localIdx += blockDim.x)
            localSieve[localIdx] = MAX_WORD_VALUE;
        // synchronize block to avoid race conditions on shared memory
        __syncthreads();

        // parallelize among warps
        for (uint32_t primeIndex = warpIdx; primeIndex < primeCount; primeIndex += warpsPerBlock)
        {
            const uint32_t prime = primes[primeIndex];
            const uint64_t primeSquared = uint64_t(prime) * prime;

            // Unmark multiples starting at just beyond the start of our block or the square of the prime, 
            //   whichever is larger.
            uint64_t firstUnmarked = (primeSquared >= start) ? primeSquared
                                                             : ((start / prime + 1) * prime);
            // We're marking off odd multiples only, so make sure we start with one of those!
            if (!(firstUnmarked & 1))
                firstUnmarked += prime;

            // parallelize among lanes of warp
            for (uint64_t index = (firstUnmarked >> 1) + uint64_t(laneIdx) * prime; index < tileEndIndex; index += (prime << LOG2_WARP_SIZE)) 
                // Clear the bit in the word that corresponds to the last part of the index 
                atomicAnd_block(&localSieve[WORD_INDEX(index - (start >> 1))], ~(sieve_t(1) << BIT_INDEX(index)));
        }

        __syncthreads();
        for (uint32_t localIdx = threadIdx.x; localIdx < effShBlockWords; localIdx += blockDim.x)
            sieve[WORD_INDEX(start >> 1) + localIdx] = localSieve[localIdx];
        __syncthreads();
    }
}

class Sieve 
{
    uint64_t sieve_size;
    uint64_t half_size;
    uint32_t size_sqrt;
    uint64_t buffer_word_size;
    uint64_t buffer_byte_size;
    sieve_t *device_sieve_buffer;
    sieve_t *host_sieve_buffer;
    Parallelization type;

    void unmark_multiples(Parallelization type, uint32_t primeCount, uint32_t *primeList) const
    {
        // Copy the first (square root of sieve size) buffer bytes to the device
        checkCUDA(cudaMemcpy(device_sieve_buffer, host_sieve_buffer, (size_sqrt >> 4) + 1, cudaMemcpyHostToDevice));
        // Allocate device buffer for the list of primes and copy the prime list to it
        uint32_t *devicePrimeList;
        checkCUDA(cudaMalloc(&devicePrimeList, primeCount * sizeof(uint32_t)));
        checkCUDA(cudaMemcpy(devicePrimeList, primeList, primeCount << 2, cudaMemcpyHostToDevice));

        // Unmark multiples on the GPU using the selected method
        switch(type)
        {
            case Parallelization::naive:
            {
                // The number of threads we use is the maximum or the number of primes to process, whichever is lower
                const uint32_t threadsPerBlock = BLOCK_SIZE;
                const uint32_t warpsPerBlock = threadsPerBlock >> LOG2_WARP_SIZE;
                const uint32_t blockCount = (primeCount + warpsPerBlock - 1) / warpsPerBlock;

                #ifdef DEBUG
                printf("- starting multiple unmarking with %u threads.\n", threadsPerBlock * blockCount);
                #endif

                unmark_multiples_naive<<<blockCount, threadsPerBlock>>>(primeCount, devicePrimeList, half_size, size_sqrt, device_sieve_buffer);
                // check for launch failure (no sync)
                checkCUDA(cudaGetLastError());
            }
            break;

            case Parallelization::tiled:
            {
                // Our workspace is the part of the sieve beyond the square root of its size...
                const uint64_t sieveSpace = sieve_size - size_sqrt;
                // ...which we halve and then divide by the word bit count to establish the number of words...
                uint64_t wordCount = sieveSpace >> (WORD_SHIFT + 1);
                // ...and increase that if the division left a remainder.
                if (sieveSpace & SIEVE_BITS_MASK)
                    wordCount++;
                
                // The number of blocks is the maximum thread count or the number of words, whichever is lower
                // const uint32_t blockCount = (uint32_t)min(uint64_t(BLOCK_SIZE), wordCount);
                int numBlocksPerSm = 0;
                int dev;
                cudaDeviceProp deviceProp;
                checkCUDA(cudaGetDevice(&dev));
                checkCUDA(cudaGetDeviceProperties(&deviceProp, dev));
                // Determine how many blocks fit on one SM assuming no shared memory
                checkCUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm,
                                                                         unmark_multiples_tiled,
                                                                         BLOCK_SIZE, 0));
                // Determine the amount of shared memory per block that can be used without changing
                // the number of blocks per SM 
                const uint32_t shBlockBytes = min(deviceProp.sharedMemPerMultiprocessor / numBlocksPerSm, deviceProp.sharedMemPerBlock);
                const uint32_t shBlockWords = shBlockBytes / BYTES_PER_WORD;
                const uint32_t minWordsPerBlock = BLOCK_SIZE >> WORD_SHIFT; // == warpsPerBlock
                const uint32_t blockCount = (uint32_t)min(uint64_t(deviceProp.multiProcessorCount) * numBlocksPerSm,
                                                          (wordCount + minWordsPerBlock - 1) / minWordsPerBlock);

                const uint64_t blockSize = ((sieveSpace + blockCount - 1) / blockCount) & SIEVE_WORD_MASK;

                #ifdef DEBUG
                printf("- starting block multiple unmarking with blockCount %u and blockSize %" PRIu64 ".\n", blockCount, blockSize);
                #endif

                unmark_multiples_tiled<<<blockCount, BLOCK_SIZE, shBlockWords * BYTES_PER_WORD>>>(primeCount, devicePrimeList, half_size, size_sqrt, blockSize, device_sieve_buffer, shBlockWords);
                checkCUDA(cudaGetLastError());
            }
            break;

            default:
                // This is some method variation we don't know, so we warn and do nothing
                fprintf(stderr, "WARNING: Parallelization type %d unknown, multiple unmarking skipped!\n\n", to_underlying(type));
            break;
        }
        
        // Release the device prime list buffer
        checkCUDA(cudaFree(devicePrimeList));

        // Copy the sieve buffer from the device to the host. This function implies a wait for all GPU threads to finish.
        checkCUDA(cudaMemcpy(host_sieve_buffer, device_sieve_buffer, buffer_byte_size, cudaMemcpyDeviceToHost));
        
        #ifdef DEBUG
        printf("- device to host copy of sieve buffer complete.\n");
        #endif
    }

    public:

    Sieve(unsigned long size, Parallelization par = Parallelization::tiled) :
        sieve_size(size),
        half_size(size >> 1),
        size_sqrt((uint32_t)sqrt(size)),
        buffer_word_size((half_size >> WORD_SHIFT) + 1),
        buffer_byte_size(buffer_word_size * BYTES_PER_WORD),
        type(par)
    {
        if (uint64_t(size_sqrt) * size_sqrt < size)
            ++size_sqrt;
        #ifdef DEBUG
        printf("- constructing sieve with buffer_word_size %zu and buffer_byte_size %zu.\n", buffer_word_size, buffer_byte_size);
        #endif

        // Allocate the device sieve buffer
        checkCUDA(cudaMalloc(&device_sieve_buffer, buffer_byte_size));

        // The number of blocks is the maximum number of threads or the number of words in the buffer, whichever is lower
        const uint32_t blockCount = (uint32_t)min(uint64_t(BLOCK_SIZE), buffer_word_size);
        
        uint64_t blockSize = buffer_word_size / blockCount;
        // Increase block size if the calculating division left a remainder
        if (buffer_word_size % blockCount)
            blockSize++;

        #ifdef DEBUG
        printf("- initializing device buffer with blockCount %u and blockSize %zu.\n", blockCount, blockSize);
        #endif

        // unnecessary for tiled version
        // if (type == Parallelization::naive)
            checkCUDA(cudaMemset(device_sieve_buffer, 255, buffer_byte_size));

        // Allocate host sieve buffer (odd numbers only) and initialize the bytes up to the square root of the sieve 
        //   size to all 1s.
        host_sieve_buffer = (sieve_t *)malloc(buffer_byte_size);
        memset(host_sieve_buffer, 255, (size_sqrt >> 4) + 1);

        // Make sure the initialization of the device sieve buffer has completed
        checkCUDA(cudaDeviceSynchronize());

        #ifdef DEBUG
        printf("- post buffer initialization device sync complete.\n");
        #endif
    }

    ~Sieve() 
    {
        checkCUDA(cudaFree(device_sieve_buffer));
        free(host_sieve_buffer);
    }

    sieve_t *run() const
    {
        // Calculate the size of the array we need to reserve for the primes we find up to and including the square root of
        //   the sieve size. x / (ln(x) - 1) is a good approximation, but often lower than the actual number, which would
        //   cause out-of-bound indexing. This is why we use x / (ln(x) - 1.2) to "responsibly over-allocate".
        const uint32_t primeListSize = uint32_t(double(size_sqrt) / (log(size_sqrt) - 1.2));

        uint32_t primeList[primeListSize];
        uint32_t primeCount = 0;

        // What follows is the basic Sieve of Eratosthenes algorithm, except we clear multiples up to and including the
        //   square root of the sieve size instead of to the sieve limit. We also keep track of the primes we find, so the
        //   GPU can unmark them later.
        const uint32_t lastMultipleIndex = size_sqrt >> 1;

        for (uint32_t factor = 3; factor <= size_sqrt; factor += 2)
        {
            uint64_t index = factor >> 1;

            if (host_sieve_buffer[WORD_INDEX(index)] & (sieve_t(1) << BIT_INDEX(index))) 
            {
                primeList[primeCount++] = factor;

                for (index = (factor * factor) >> 1; index <= lastMultipleIndex; index += factor)
                    host_sieve_buffer[WORD_INDEX(index)] &= ~(sieve_t(1) << BIT_INDEX(index));
            }
        }

        // Use the GPU to unmark the rest of the primes multiples
        unmark_multiples(type, primeCount, primeList);

        // Required to be truly compliant with Primes project rules
        return host_sieve_buffer;
    }

    uint64_t count_primes() const
    {
        uint64_t primeCount = 0;
        const uint64_t lastWord = WORD_INDEX(half_size);
        sieve_t word;

        // For all buffer words except the last one, just count the set bits in the word until there are none left.
        //   We only hold bits for odd numbers in the sieve buffer. However, due to a small "mathematical coincidence"
        //   bit 0 of word 0 effectively represents the only even prime 2. This means the "count set bits" approach 
        //   in itself yields the correct result.
        for (uint64_t index = 0; index < lastWord; index++)
        {
            word = host_sieve_buffer[index];
            while (word) 
            {
                if (word & 1)
                    primeCount++;

                word >>= 1;
            }
        }

        // For the last word, only count bits up to the (halved) sieve limit
        word = host_sieve_buffer[lastWord];
        const uint32_t endBit = BIT_INDEX(half_size);
        for (int32_t index = 0; word && index < endBit; index++) 
        {
            if (word & 1)
                primeCount++;
            
            word >>= 1;
        }

        return primeCount;
    }
};

const std::map<uint64_t, const int> resultsDictionary =
{
    {             10UL, 4         }, // Historical data for validating our results - the number of primes
    {            100UL, 25        }, //   to be found under some limit, such as 168 primes under 1000
    {          1'000UL, 168       },
    {         10'000UL, 1229      },
    {        100'000UL, 9592      },
    {      1'000'000UL, 78498     },
    {     10'000'000UL, 664579    },
    {    100'000'000UL, 5761455   },
    {  1'000'000'000UL, 50847534  },
    { 10'000'000'000UL, 455052511 },
};

const std::map<Parallelization, const char *> parallelizationDictionary = 
{
    { Parallelization::naive, "naive" },
    { Parallelization::tiled, "tiled" }
};

// Assumes any numerical first argument is the desired sieve size. Defaults to DEFAULT_SIEVE_SIZE.
uint64_t determineSieveSize(int argc, char *argv[])
{
    if (argc < 2)
        return DEFAULT_SIEVE_SIZE;

    const uint64_t sieveSize = strtoul(argv[1], nullptr, 0);

    if (sieveSize == 0) 
        return DEFAULT_SIEVE_SIZE;

    if (resultsDictionary.find(sieveSize) == resultsDictionary.end())
        fprintf(stderr, "WARNING: Results cannot be validated for selected sieve size of %zu!\n\n", sieveSize);
    
    return sieveSize;
}

void printResults(Parallelization type, uint64_t sieveSize, uint64_t primeCount, double duration, uint64_t passes)
{
    const auto expectedCount = resultsDictionary.find(sieveSize);
    const auto countValidated = expectedCount != resultsDictionary.end() && expectedCount->second == primeCount;
    const auto parallelizationEntry = parallelizationDictionary.find(type);
    const char *parallelizationLabel = parallelizationEntry != parallelizationDictionary.end() ? parallelizationEntry->second : "unknown";

    fprintf(stderr, "Passes: %zu, Time: %lf, Avg: %lf, Word size: %d, Max GPU threads per block: %d, Type: %s, Limit: %zu, Count: %zu, Validated: %d\n", 
            passes,
            duration,
            duration / passes,
            BITS_PER_WORD,
            BLOCK_SIZE,
            parallelizationLabel,
            sieveSize,
            primeCount,
            countValidated);

    printf("pauleonix_faithful_cuda_%s;%zu;%f;1;algorithm=base,faithful=yes,bits=1\n\n", parallelizationLabel, passes, duration);
}

int main(int argc, char *argv[])
{
    const uint64_t sieveSize = determineSieveSize(argc, argv);

    Parallelization types[] = { Parallelization::naive,
                                Parallelization::tiled };

    for (auto &type : types)
    {
        uint64_t passes = 0;

        std::unique_ptr<Sieve> sieve(nullptr);

        const auto startTime = steady_clock::now();
        duration<double, std::micro> runTime;

        #if !( defined(DEBUG) || defined(DEBUG_GPU) || defined(SINGLE) )
        do
        {
        #endif

            sieve.reset(nullptr);

            sieve = std::make_unique<Sieve>(sieveSize, type);
            sieve->run();

            passes++;

            runTime = steady_clock::now() - startTime;

        #if !( defined(DEBUG) || defined(DEBUG_GPU) || defined(SINGLE) )
        }
        while (duration_cast<seconds>(runTime).count() < 5);
        #endif
        
        #if defined(DEBUG) || defined(DEBUG_GPU)
        printf("\n");
        #endif

        const size_t primeCount = sieve->count_primes();
        
        printResults(type, sieveSize, primeCount, duration_cast<microseconds>(runTime).count() / 1000000.0, passes); 
    }
}
