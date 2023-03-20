#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <map>
#include <cuda_runtime.h>

#include "primes.h"

using namespace std::chrono;

__global__ void unmark_multiples_blocks(uint32_t primeCount, uint32_t *primes, uint64_t halfSize, uint32_t sizeSqrt, uint32_t maxBlockIndex, uint64_t blockSize, sieve_t *sieve)
{
    // Calculate the start and end of the block we need to work on, at buffer word boundaries. 
    //   Note that the first variable is a number in sieve space...
    uint64_t blockStart = uint64_t(blockIdx.x) * blockSize + sizeSqrt;
    //   ...and the second is an index in the sieve buffer (representing odd numbers only)
    const uint64_t lastIndex = (blockIdx.x == maxBlockIndex) ? halfSize : (((blockStart + blockSize) & SIEVE_WORD_MASK) >> 1) - 1;

    // If this is not the first block, we actually start at the beginning of the first block word
    if (blockIdx.x != 0)
        blockStart &= SIEVE_WORD_MASK;

    for (uint32_t primeIndex = 0; primeIndex < primeCount; primeIndex++)
    {
        const uint32_t prime = primes[primeIndex];
        const uint64_t primeSquared = uint64_t(prime) * prime;

        // Unmark multiples starting at just beyond the start of our block or the square of the prime, 
        //   whichever is larger.
        uint64_t firstUnmarked = primeSquared >= blockStart ? primeSquared : ((blockStart / prime + 1) * prime);
        // We're marking off odd multiples only, so make sure we start with one of those!
        if (!(firstUnmarked & 1))
            firstUnmarked += prime;

        if (prime <= ROLLING_LIMIT)
        {
            uint64_t index = firstUnmarked >> 1;
            if (index > lastIndex)
                continue;

            uint64_t wordIndex = WORD_INDEX(index);
            uint32_t bitIndex = BIT_INDEX(index);                
            sieve_t bitMask = 0;

            do
            {
                // Check if our bit index has moved past the current word's bits. If so...
                if (bitIndex > MAX_BIT_INDEX) 
                {
                    // ...clear the current word's bits that are set in the mask, and move on the next word.
                    sieve[wordIndex++] &= ~bitMask;
                    // "Shift bitmask one word to the right" through calculation. It has to be done that way
                    //   in part because our word length may be the maximum the GPU supports (64 bits). 
                    bitIndex %= BITS_PER_WORD;
                    bitMask = sieve_t(1) << bitIndex;
                }
                else
                    // Just add the current bit index to the current word's mask
                    bitMask |= sieve_t(1) << bitIndex;

                // Add prime to overall sieve index and current word's bit index
                index += prime;
                bitIndex += prime;
            }
            while (index <= lastIndex);

            // Let's not forget to apply the last bitmask
            sieve[wordIndex] &= ~bitMask;
        }
        else
        {
            for (uint64_t index = firstUnmarked >> 1; index <= lastIndex; index += prime) 
                sieve[WORD_INDEX(index)] &= ~(sieve_t(1) << BIT_INDEX(index));   // Clear the bit in the word that corresponds to the last part of the index 
        }
    }
    
}

class Sieve 
{
    const uint64_t sieve_size;
    const uint64_t half_size;
    const uint32_t size_sqrt;
    const uint64_t buffer_word_size;
    const uint64_t buffer_byte_size;
    sieve_t *device_sieve_buffer = nullptr;

    void unmark_multiples(uint32_t primeCount, uint32_t *primeList) 
    {
        // Our workspace is the part of the sieve beyond the square root of its size...
        const uint64_t sieveSpace = sieve_size - size_sqrt;
        // ...which we halve and then divide by the word bit count to establish the number of words...
        uint64_t wordCount = sieveSpace >> (WORD_SHIFT + 1);
        // ...and increase that if the division left a remainder.
        if (sieveSpace & SIEVE_BITS_MASK)
            wordCount++;
        
        // The number of blocks is the maximum thread count or the number of words, whichever is lower
        const uint32_t blockCount = (uint32_t)min(uint64_t(MAX_THREADS), wordCount);
        
        uint64_t blockSize = sieveSpace / blockCount;
        // Increase block size if the calculating division left a remainder
        if (sieveSpace % blockCount)
            blockSize++;

        unmark_multiples_blocks<<<blockCount, 1>>>(primeCount, primeList, half_size, size_sqrt, blockCount - 1, blockSize, device_sieve_buffer);
    }

    public:

    Sieve(unsigned long size) :
        sieve_size(size),
        half_size(size >> 1),
        size_sqrt((uint32_t)sqrt(size) + 1),
        buffer_word_size((half_size >> WORD_SHIFT) + 1),
        buffer_byte_size(buffer_word_size * BYTES_PER_WORD)
    {
        // Allocate the device sieve buffer
        cudaError_t result = cudaMallocManaged(&device_sieve_buffer, buffer_byte_size);
        if (cudaSuccess != result)
        {
            fprintf(stderr, "Unable to alllocate %ld managed memory bytes, error=%08x, %s", buffer_byte_size, result, cudaGetErrorString(result));
            exit(0);
        }
        cudaMemset(device_sieve_buffer, ~0, buffer_byte_size);

    }

    ~Sieve() 
    {
        cudaFree(device_sieve_buffer);
    }

    sieve_t *run()
    {
        // Calculate the size of the array we need to reserve for the primes we find up to and including the square root of
        //   the sieve size. x / (ln(x) - 1) is a good approximation, but often lower than the actual number, which would
        //   cause out-of-bound indexing. This is why we use x / (ln(x) - 1.2) to "responsibly over-allocate".
        const uint32_t primeListSize = uint32_t(double(size_sqrt) / (log(size_sqrt) - 1.2));

        uint32_t * primeList;
        cudaError_t result = cudaMallocManaged(&primeList, primeListSize * sizeof(uint32_t));
        if (cudaSuccess != result)
        {
            fprintf(stderr, "Unable to alllocate %ld managed memory bytes for prime list, error=%08x, %s", buffer_byte_size, result, cudaGetErrorString(result));
            exit(0);
        }

        uint32_t primeCount = 0;

        // What follows is the basic Sieve of Eratosthenes algorithm, except we clear multiples up to and including the
        //   square root of the sieve size instead of to the sieve limit. We also keep track of the primes we find, so the
        //   GPU can unmark them later.
        const uint32_t lastMultipleIndex = size_sqrt >> 1;

        for (uint32_t factor = 3; factor <= size_sqrt; factor += 2)
        {
            uint64_t index = factor >> 1;

            if (device_sieve_buffer[WORD_INDEX(index)] & (sieve_t(1) << BIT_INDEX(index))) 
            {
                primeList[primeCount++] = factor;

                for (index = (factor * factor) >> 1; index <= lastMultipleIndex; index += factor)
                    device_sieve_buffer[WORD_INDEX(index)] &= ~(sieve_t(1) << BIT_INDEX(index));
            }
        }

        // Use the GPU to unmark the rest of the primes multiples
        unmark_multiples(primeCount, primeList);

        cudaFree(primeList);

        // Required to be truly compliant with Primes project rules
        return device_sieve_buffer;
    }

    uint64_t count_primes() 
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
            word = device_sieve_buffer[index];
            while (word) 
            {
                if (word & 1)
                    primeCount++;

                word >>= 1;
            }
        }

        // For the last word, only count bits up to the (halved) sieve limit
        word = device_sieve_buffer[lastWord];
        const uint32_t lastBit = BIT_INDEX(half_size);
        for (uint32_t index = 0; word && index <= lastBit; index++) 
        {
            if (word & 1)
                primeCount++;
            
            word >>= 1;
        }

        return primeCount;
    }
};

// resultsDictionary
//
// A table listing how many primes should be found up to a specific limit

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

void printResults(uint64_t sieveSize, uint64_t primeCount, double duration, uint64_t passes)
{
    const auto expectedCount         = resultsDictionary.find(sieveSize);
    const auto countValidated        = expectedCount != resultsDictionary.end() && expectedCount->second == primeCount;

    fprintf(stderr, "Passes: %zu, Time: %lf, Avg: %lf, Word size: %d, Max GPU threads: %d, Limit: %zu, Count: %zu, Validated: %d\n", 
            passes,
            duration,
            duration / passes,
            BITS_PER_WORD,
            MAX_THREADS,
            sieveSize,
            primeCount,
            countValidated);

    printf("rbergen_faithful_cuda_shared;%zu;%f;1;algorithm=base,faithful=yes,bits=1\n\n", passes, duration);
}

int main(int argc, char *argv[])
{
    const uint64_t sieveSize = determineSieveSize(argc, argv);
    uint64_t passes = 0;
    Sieve *sieve = nullptr;

    const auto startTime = steady_clock::now();
    duration<double, std::micro> runTime;

    do
    {
        delete sieve;

        sieve = new Sieve(sieveSize);
        sieve->run();

        passes++;

        runTime = steady_clock::now() - startTime;
    }
    while (duration_cast<seconds>(runTime).count() < 5);
    
    const size_t primeCount = sieve->count_primes();
    delete sieve;
    printResults(sieveSize, primeCount, duration_cast<microseconds>(runTime).count() / 1000000.0, passes); 
}
