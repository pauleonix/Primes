#define DEFAULT_SIEVE_SIZE 1'000'000

// can be modified to optimize GPU perf
// should always be a multiple of 32 (warp size) for these kernels to work properly
#define BLOCK_SIZE 256

// Bits per sieve buffer word. Can be 32 or 64.
#define BITS_PER_WORD 32

// If defined, the code will show debug output and run all unmarking methods once.
//#define DEBUG

//=======================================================================================================
//
// No user-modifiable defines below. :)
//

// While this could in theory change for a future GPU generation, at present it is true for all Nvidia GPUs
#define WARP_SIZE 32
#define LOG2_WARP_SIZE 5
#define LANE_MASK ((WARP_SIZE) - 1)

#include <cstdint>
#include <type_traits>

#if BITS_PER_WORD == 32
    typedef unsigned int sieve_t;
    #define MAX_WORD_VALUE UINT32_MAX
    #define WORD_SHIFT 5
#elif BITS_PER_WORD == 64
    typedef unsigned long long sieve_t;
    #define MAX_WORD_VALUE UINT64_MAX
    #define WORD_SHIFT 6
#else
    #error "Unsupported value for BITS_PER_WORD defined; 32 and 64 are supported."
#endif

#define BYTES_PER_WORD ((BITS_PER_WORD) >> 3)
#define MAX_BIT_INDEX ((BITS_PER_WORD) - 1)
#define WORD_INDEX(index) ((index) >> (WORD_SHIFT))
#define BIT_INDEX(index) ((index) & (MAX_BIT_INDEX))
// This is actually BITS_PER_WORD * 2 - 1, but this is a "cheap" way to get there
#define SIEVE_BITS_MASK ((BITS_PER_WORD) + (MAX_BIT_INDEX))
#define SIEVE_WORD_MASK ~uint64_t(SIEVE_BITS_MASK)

enum class Parallelization : char
{
    naive,
    tiled
};

// We have to define this ourselves, as we're not doing C++23 (yet)
template<class TEnum>
constexpr auto to_underlying(TEnum enumValue)  
{
   return static_cast<typename std::underlying_type<TEnum>::type>(enumValue);
}

// CUDA runtime error checking:
#define errorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s in %s(%d): %s\n", cudaGetErrorName(code), file, line, cudaGetErrorString(code));
        if (abort) exit(code);
    }
}
