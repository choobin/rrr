#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <ctime>

#include <algorithm>

#include "rrr.h"

// Test bvector::rrr access, rank and select operations using randomly
// generated bit vectors with varied densities, block lengths and
// sample rates.

struct Fixture {
    uint64_t *bvector;
    uint64_t *access;
    uint64_t *rank0;
    uint64_t *rank1;
    uint64_t *select0;
    uint64_t *select1;
    uint64_t *order;

    uint64_t nbits;
    double density;
    uint64_t block_length;
    uint64_t sample_length;

    uint64_t one_bits;
    uint64_t zero_bits;

    bvector::rrr *rrr;

    Fixture(uint64_t n, double d, uint64_t b, uint64_t s) :
        bvector(new uint64_t[(n + 63) / 64]()),
        access(new uint64_t[n]),
        rank0(new uint64_t[n]),
        rank1(new uint64_t[n]),
        select0(new uint64_t[n]),
        select1(new uint64_t[n]),
        order(new uint64_t[n]),
        nbits(n),
        density(d),
        block_length(b),
        sample_length(s),
        one_bits(0),
        zero_bits(0),
        rrr(nullptr) {

        srand48(time(NULL));

        for (uint64_t i = 0; i < nbits; i++) {
            rank0[i] = zero_bits;
            rank1[i] = one_bits;

            if (drand48() <= density) {
                uint64_t block = i / 64;
                uint64_t index = i % 64;

                bvector[block] |=  (1ULL << index);

                access[i] = 1;

                select1[one_bits] = i;
                one_bits++;
            }
            else {
                access[i] = 0;

                select0[zero_bits] = i;
                zero_bits++;
            }
        }

        rrr = new bvector::rrr(bvector, nbits, block_length, sample_length);

        test();
    }

    Fixture(const Fixture&) = delete;

    Fixture(const Fixture&&) = delete;

    Fixture& operator=(const Fixture&) = delete;

    ~Fixture() {
        delete[] bvector;
        delete[] access;
        delete[] rank0;
        delete[] rank1;
        delete[] select0;
        delete[] select1;
        delete[] order;
        delete rrr;
    }

    void compute_order(uint64_t n) {
        for (uint64_t i = 0; i < n; i++)
            order[i] = i;

        std::random_shuffle(order, order + n);
    }

#define TEST(fn, n)                                     \
    void test_##fn() {                                  \
        compute_order(n);                               \
        for (uint64_t i = 0; i < n; i++)                \
            assert(rrr->fn(order[i]) == fn[order[i]]);  \
    }
    TEST(access, nbits)
    TEST(rank0, nbits)
    TEST(rank1, nbits)
    TEST(select0, zero_bits)
    TEST(select1, one_bits)
#undef TEST

    void test_rank0_select0() {
        compute_order(zero_bits);
        for (uint64_t i = 0; i < zero_bits; i++)
            assert(rrr->rank0(rrr->select0(order[i])) == rank0[select0[order[i]]]);
    }

    void test_rank1_select1() {
        compute_order(one_bits);
        for (uint64_t i = 0; i < one_bits; i++)
            assert(rrr->rank1(rrr->select1(order[i])) == rank1[select1[order[i]]]);
    }

    void test() {
        test_access();
        test_rank0();
        test_rank1();
        test_select0();
        test_select1();
        test_rank0_select0();
        test_rank1_select1();
    }
};

int main()
{
    uint64_t nbits = 1000000;
    uint64_t block_length[] = { 15, 48, 63 };
    double density[] = { 0.05, 0.1, 0.2, 0.4, 0.8 };
    uint64_t sample_length[] = { 32, 64, 128 };

    for (uint64_t b : block_length) {
        for (double d : density) {
            for (uint64_t s : sample_length) {
                printf("testing: block_length: %2ld, density: %.2f, sample_length: %3ld ",
                       b, d, s);

                Fixture f(nbits, d, b, s);

                printf("\033[1m[PASSED]\033[0m\n");
            }
        }
    }

    return EXIT_SUCCESS;
}
