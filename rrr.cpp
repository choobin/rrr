/*
Copyright (c) 2013, Christopher Hoobin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rrr.h"

namespace bvector {

static uint64_t clog2(uint64_t x) {
    if (x == 0) return 0;
    return (uint64_t)ceil(log(x)/log(2));
}

static uint64_t encode(uint64_t value, FILE *fp) {
    uint64_t nbytes = 0;
    uint8_t nibble;

    while (value >= 0x80) {
        nibble = (value & 0x7F) | 0x80;

        if (fp) fwrite(&nibble, sizeof nibble, 1, fp);

        nbytes++;

        value >>= 7;
    }

    nibble = value & 0x7F;

    if (fp) fwrite(&nibble, sizeof nibble, 1, fp);

    nbytes++;

    return nbytes;
}

static uint64_t decode(uint64_t *value, FILE *fp) {
    assert(value);
    assert(fp);

    uint64_t nbytes = 0;
    uint64_t shift = 0;
    uint8_t nibble;

    *value = 0;

    while (1) {
        fread(&nibble, sizeof nibble, 1, fp);
        nbytes++;

        *value |= ((nibble & 0x7F) << shift);

        shift += 7;

        if (nibble < 0x80)
            break;
    }

    return nbytes;
}

// A simple bit vector class that supports
// fixed and variable width access/insertions.

class bvector {
private:
    uint64_t *data;
    uint64_t nbits;
    uint64_t width;
    uint64_t position;
    bool cleanup;

public:
    // Create an empty bvector of size n bits.
    bvector(uint64_t n) :
        data(new uint64_t[(n + 63) / 64]()),
        nbits(n),
        width(0),
        position(0),
        cleanup(true) {
    }

    // Create an empty fixed width bvector of n w bits items.
    bvector(uint64_t n, uint64_t w) :
        data(new uint64_t[((n * w) + 63) / 64]()),
        nbits(n * w),
        width(w),
        position(0),
        cleanup(true) {
    }

    // Create a fixed width bvector of n w bits items from array d.
    bvector(uint64_t *d, uint64_t n, uint64_t w, bool c = true) :
        data(d),
        nbits(n),
        width(w),
        position(n),
        cleanup(c) {
    }

    bvector(FILE *fp) {
        assert(fp);

        decode(&nbits, fp);
        decode(&width, fp);
        decode(&position, fp);

        uint64_t nblocks = (nbits + 63) / 64;

        data = new uint64_t[nblocks]();

        fread(&data, sizeof *data, nblocks, fp);

        cleanup = true;
    }

    ~bvector() {
        if (cleanup)
            delete[] data;
    }

    bvector(const bvector&) = delete;

    bvector(const bvector&&) = delete;

    bvector& operator=(const bvector&) = delete;

    // Return n bits from index i.
    uint64_t get(uint64_t i, uint64_t n) const {
      //assert(i + n <= nbits);
        assert(n > 0);
        assert(n <= 64);

        uint64_t block = i / 64;

        uint64_t index = i % 64;

        uint64_t mask = (1ULL << n) - 1;

        uint64_t value = (data[block] >> index) & mask;

        uint64_t bits = 64 - index;

        uint64_t nblocks = (nbits + 63) / 64;

        // rrr blocks contain a fixed number of bits. If the last
        // block is smaller than the required block size we set the
        // remaining bits to zero. Hence the commented assertion at
        // the beginning of the function and the check below to see
        // if we step out of bounds when fetching the overflow bits
        // from the next block.

        if (bits < n && (block + 1) < nblocks) {
            uint64_t overflow = n - bits;

            uint64_t overflow_mask = (1ULL << overflow) - 1;

            value |= (data[block + 1] & overflow_mask) << bits;
        }

        return value;
    }

    // Insert value in n bits at the current position.
    void set(uint64_t value, uint64_t n) {
        assert(position + n <= nbits);
        assert(clog2(value) <= n);
        assert(n > 0);
        assert(n <= 64);

        uint64_t block = position / 64;

        uint64_t index = position % 64;

        uint64_t bits = 64 - index;

        data[block] |= (value << index);

        if (bits < n)
            data[block + 1] |= (value >> bits);

        position += n;
    }

    // Returns width bits from index i*width.
    // Assumes a fixed width bvector, i.e., width > 0.
    uint64_t get(uint64_t i) const {
        assert(width > 0);

        return get(i * width, width);
    }

    // Insert value in width bits at the current position.
    // Assumes a fixed width bvector, i.e., width > 0.
    void set(uint64_t value) {
        assert(width > 0);

        set(value, width);
    }

    void save(FILE *fp) const {
        assert(fp);

        encode(nbits, fp);
        encode(width, fp);
        encode(position, fp);

        size_t nblocks = (nbits + 63) / 64;

        fwrite(&data, sizeof *data, nblocks, fp);
    }

    uint64_t size() const {
        return
            encode(nbits, NULL) +
            encode(width, NULL) +
            encode(position, NULL) +
            (sizeof *data * ((nbits + 63) / 64));
    }
};

class rrr::impl {
public:
    impl(uint64_t *d, uint64_t n, uint64_t b, uint64_t s);

    impl(FILE *fp);

    ~impl();

    uint64_t access(uint64_t i) const;

    uint64_t rank0(uint64_t i) const;

    uint64_t rank1(uint64_t i) const;

    uint64_t select0(uint64_t i) const;

    uint64_t select1(uint64_t i) const;

    void save(FILE *fp) const;

    uint64_t size() const;

private:
    impl(const impl&) = delete;
    impl(const impl&&) = delete;
    impl& operator=(const impl&) = delete;

    uint64_t **binomial_coefficient;

    uint16_t *offset_bits;
    uint16_t *offset_position;
    uint16_t *class_offset;
    uint16_t *combinations;

    bvector *classes;
    bvector *offsets;

    bvector *rank_samples;
    bvector *offset_samples;

    uint64_t block_length;
    uint64_t sample_length;
    uint64_t nblocks;
    uint64_t nsamples;
    uint64_t none_bits;
    uint64_t nbits;

    void compute_tables();

    uint64_t compute_offset(uint64_t klass, uint64_t offset) const;

    uint64_t compute_block(uint64_t klass, uint64_t i) const;

    void build(uint64_t *data);

    uint64_t popcount(uint64_t i) const;
};

// Represent n bits from d in blocks of size b sampling every s blocks.
rrr::impl::impl(uint64_t *d, uint64_t n, uint64_t b, uint64_t s) :
    block_length(b),
    sample_length(s),
    nblocks((n + block_length - 1) / block_length),
    nbits(n) {
    compute_tables();
    build(d);
}

rrr::impl::impl(FILE *fp) {
    assert(fp);

    decode(&block_length, fp);
    decode(&sample_length, fp);
    decode(&nbits, fp);
    decode(&none_bits, fp);

    nblocks = (nbits + block_length - 1) / block_length;
    nsamples = (nblocks + sample_length - 1) / sample_length;

    classes = new bvector(fp);
    offsets = new bvector(fp);

    rank_samples = new bvector(fp);
    offset_samples = new bvector(fp);

    compute_tables();
}

rrr::impl::~impl() {
    for (uint64_t i = 0; i <= block_length; i++)
        delete[] binomial_coefficient[i];

    delete[] binomial_coefficient;

    delete[] offset_bits;

    if (block_length <= 15) {
        delete[] class_offset;
        delete[] offset_position;
        delete[] combinations;
    }

    delete classes;
    delete offsets;

    delete rank_samples;
    delete offset_samples;
}

void rrr::impl::compute_tables() {
    assert(block_length <= 63);

    binomial_coefficient = new uint64_t*[block_length + 1];

    for (uint64_t i = 0; i <= block_length; i++) {
        binomial_coefficient[i] = new uint64_t[i + 1];
        binomial_coefficient[i][0] = 1;
        binomial_coefficient[i][i] = 1;

        for (uint64_t j = 1; j < i; j++) {
            binomial_coefficient[i][j] =
                binomial_coefficient[i - 1][j - 1] +
                binomial_coefficient[i - 1][j];
        }
    }

    offset_bits = new uint16_t[block_length + 1];

    offset_bits[0] = 0;
    offset_bits[block_length] = 0;

    for (uint64_t i = 1; i < block_length; i++)
        offset_bits[i] = clog2(binomial_coefficient[block_length][i]);

    if (block_length <= 15) {
        class_offset = new uint16_t[block_length + 1];
        offset_position = new uint16_t[1ULL << block_length];
        combinations = new uint16_t[1ULL << block_length];

        uint64_t i = 0;

        for (uint64_t klass = 0; klass <= block_length; klass++) {

            uint64_t n = binomial_coefficient[block_length][klass];

            uint64_t x = (1ULL << klass) - 1;

            class_offset[klass] = i;

            for (uint64_t j = 0; j < n; j++) {
                offset_position[x] = i - class_offset[klass];

                combinations[i] = x;

                uint64_t k = x | (x - 1); // bithacks #NextBitPermutation

                x = (k + 1) | (((~k & -~k) - 1) >> (__builtin_ctz(x) + 1));

                i++;
            }
        }

        assert(i == 1ULL << block_length);
    }
}

// For information about encoding/decoding block offsets on the fly
// please refer to section 3 of G. Navarro and E. Providel. Fast, small,
// simple rank/select on bitmaps. In Experimental Algorithms, volume 7276
// of Lecture Notes in Computer Science, pages 295–306. 2012.

inline uint64_t rrr::impl::compute_offset(uint64_t klass, uint64_t i) const {
    if (block_length <= 15)
        return offset_position[i];

    uint64_t r = 0;
    uint64_t n = block_length - 1;
    uint64_t k = klass;

    while (k > 0) {
        if (i & (1ULL << n)) {
            if (k <= n)
                r += binomial_coefficient[n][k];

            k--;
        }
        n--;
    }

    return r;
}

// Note that there are a number of improvements we can make to rank,
// select and access calls by specializing this function. That is, we
// can stop decoding once we reach the required bit. This is mentioned
// in the references. I doubt that it would be much of an improvement,
// nevertheless, I found implementing it this way makes it significantly
// easier to understand -- from a pedagogical point of view -- which
// was the main goal of this project.

inline uint64_t rrr::impl::compute_block(uint64_t klass, uint64_t offset) const {
    if (block_length <= 15)
        return combinations[class_offset[klass] + offset];

    uint64_t r = 0;
    uint64_t n = block_length - 1;
    uint64_t k = klass;

    while (k > 0 && n > 0) {
        if (k <= n && offset >= binomial_coefficient[n][k]) {
            offset -= binomial_coefficient[n][k];

            r |= (1ULL << n);

            k--;
        }
        n--;
    }

    if (k > 0)
        r |= (1ULL << k) - 1;

    return r;
}

// A brief description of RRR.

// For the nitty gritty please refer to the papers in the README file.

// We divide a bit vector into blocks of length u.

// Each block represents a pair (c_i, o_i).

// c_i represents the class of the block, i.e., the number of 1's in
// the block.

// o_i is the offset in its class (of all possible combinations of class c_i).

// If u is <= 15 we compute a table storing every possible combination
// of u bits, sorted by class and offset within each class.
// (see rrr::impl::compute_tables).

// If u > 15 we encode/decode offsets and blocks on the fly

// We store the concatenation of all c's using ceil(log(u + 1)) bits
// for each c. For example; if u is 15 each c is coded in 4 bits. If u
// is 63 each c is coded in 6 bits.

// We store the concatenation of all o's using ceil(log(u \choose c_i))
// bits for each o. Note that the o's are variable width.

// To provide O(1) rank and O(log n) select we store two partial sum
// structures sampling the cumulative rank and offset position at
// sample_length block intervals.

// That is, each rank sample stores the rank up to the current block and is
// stored in ceil(log(number of 1 bits)) (see rrr::impl::class_samples).

// Each offset sample stores the current offset in the offsets array
// and is stored in ceil(log(number of bits in the offsets array))
// (see rrr::impl::offset_samples).

// Compression requires at least two passes of the input bit
// vector. An initial pass to compute the size of c, o and the
// sampling arrays, and a second pass to encode the bit vector.

// access and rank calls involve finding the closest rank and offset
// sample then continuing decoding from that point until we reach the
// required index.

// select calls initially perform a binary search over the rank
// samples then continue decoding in a similar manner to access/rank.

// Note that if the offsets are decoded on the fly we add an extra O(u)
// time to both operations.

void rrr::impl::build(uint64_t *data) {
    assert(data);

    // First pass.
    // Compute size for classes, offsets and both partial sum structures.

    uint64_t class_bits = clog2(block_length + 1);

    uint64_t rank_sum = 0;

    uint64_t offset_sum = 0;

    bvector input(data, nbits, block_length, false);

    for (uint64_t i = 0; i < nblocks; i++) {
        uint64_t value = input.get(i);

        uint64_t klass = popcount(value);

        rank_sum += klass;

        // We can skip klass=0 and klass=block_length offsets as there
        // is only one combination of their bits.
        if (klass != 0 && klass != block_length)
            offset_sum += offset_bits[klass];
    }

    // Second pass.
    // Now we compute classes, offsets and both partial sum structures.

    none_bits = rank_sum;

    classes = new bvector(nblocks, class_bits);

    offsets = new bvector(offset_sum);

    nsamples = (nblocks + sample_length - 1) / sample_length;

    uint64_t rsample_bits = clog2(rank_sum);

    uint64_t osample_bits = clog2(offset_sum);

    rank_samples = new bvector(nsamples, rsample_bits);

    offset_samples = new bvector(nsamples, osample_bits);

    rank_sum = 0;

    offset_sum = 0;

    for (uint64_t i = 0; i < nblocks; i++) {
        if (i % sample_length == 0) {
            rank_samples->set(rank_sum);

            offset_samples->set(offset_sum);
        }

        uint64_t value = input.get(i);

        uint64_t klass = popcount(value);

        classes->set(klass);

        rank_sum += klass;

        if (klass != 0 && klass != block_length) {
            uint64_t offset = compute_offset(klass, value);

            offsets->set(offset, offset_bits[klass]);

            offset_sum += offset_bits[klass];
        }
    }
}

// If the CPU supports POPCNT or SSE3's PSHUFB instruction we should
// call __builtin_popcountll. Otherwise, use Knuth's sideways addition
// rule (TAOCPv4 Fascicle 1: Bitwise tricks & techniques; Binary
// Decision Diagrams).

// Note that POPCNT is not actually considered part of SSE4.2,
// however, it was introduced at the same time. In fact, POPCNT and
// LZCNT have their own dedicated CPUID bits to indicate support,
// hence checking for __POPCNT__ and not __SSE4_2__.

inline uint64_t rrr::impl::popcount(uint64_t i) const {
#if defined __POPCNT__ || defined __SSE3__
    return __builtin_popcountll(i);
#else
    register uint64_t x = i - ((i & 0xAAAAAAAAAAAAAAAA) >> 1);
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F;
    return x * 0x0101010101010101 >> 56;
#endif
}

// B[i]

uint64_t rrr::impl::access(uint64_t i) const {
    assert(i < nbits);

    // A slower alternative: return rank1(i) - rank1(i - 1);

    uint64_t block = i / block_length;

    uint64_t index = i % block_length;

    uint64_t sample = i / block_length / sample_length;

    uint64_t j = sample * sample_length;

    uint64_t mask = 1ULL << index;

    uint64_t offset = offset_samples->get(sample);

    uint64_t klass = 0;

    while (j < block) {
        klass = classes->get(j);

        offset += offset_bits[klass];

        j++;
    }

    klass = classes->get(j);

    if (klass == 0)
        return 0;
    else if (klass == block_length)
        return 1;

    offset = offsets->get(offset, offset_bits[klass]);

    block = compute_block(klass, offset);

    return (block & mask) ? 1 : 0;
}

// rank0(i) = |{j ∈ [0, i) : B[j] = 0}|

uint64_t rrr::impl::rank0(uint64_t i) const {
    assert(i < nbits);

    return i - rank1(i);
}

// rank1(i) = |{j ∈ [0, i) : B[j] = 1}|

uint64_t rrr::impl::rank1(uint64_t i) const {
    assert(i < nbits);

    uint64_t block = i / block_length;

    uint64_t index = i % block_length;

    uint64_t sample = i / block_length / sample_length;

    uint64_t j = sample * sample_length;

    uint64_t mask = (1ULL << index) - 1;

    uint64_t rank = rank_samples->get(sample);

    uint64_t offset = offset_samples->get(sample);

    uint64_t klass = 0;

    while (j < block) {
        klass = classes->get(j);

        rank += klass;

        offset += offset_bits[klass];

        j++;
    }

    klass = classes->get(j);

    if (klass == 0)
        return rank;
    else if (klass == block_length)
        return rank + index;

    offset = offsets->get(offset, offset_bits[klass]);

    block = compute_block(klass, offset);

    return rank + popcount(block & mask);
}

// select0(i) = max{j ∈ [0, n) | rank0(j) = i}

uint64_t rrr::impl::select0(uint64_t i) const {
    assert(i < nbits);

    uint64_t zero_bits = nbits - none_bits;

    if (i > zero_bits) return ~0ULL;

    uint64_t sp = 0;
    uint64_t ep = nsamples - 1;

    while (sp < ep) {
        uint64_t mid = sp + ((ep - sp) >> 1);
        uint64_t index = mid * sample_length * block_length;

        if (index - rank_samples->get(mid) < i) {
            if (sp == mid) break;
            sp = mid;
        }
        else {
            if (mid == 0) break;
            ep = mid - 1;
        }
    }

    uint64_t j = sp * sample_length;

    uint64_t rank = j * block_length - rank_samples->get(sp);

    uint64_t offset = offset_samples->get(sp);

    uint64_t klass = 0;

    while (j < nblocks) {
        klass = classes->get(j);

        if (rank + block_length - klass > i) break;

        rank += block_length - klass;

        offset += offset_bits[klass];

        j++;
    }

    uint64_t position = j * block_length;

    uint64_t block = 0;

    assert(klass < block_length); // Need at least one 0 bit.

    if (klass > 0) {
        offset = offsets->get(offset, offset_bits[klass]);

        block = compute_block(klass, offset);
    }

    j = 0;
    while (j < block_length) {
        if (!(block & 1)) rank++;
        if (rank > i) break;
        block >>= 1;
        position++;
        j++;
    }

    return position;
}

// select1(i) = max{j ∈ [0, n) | rank1(j) = i}

uint64_t rrr::impl::select1(uint64_t i) const {
    assert(i < nbits);

    if (i > none_bits) return ~0ULL;

    uint64_t sp = 0;
    uint64_t ep = nsamples - 1;

    while (sp < ep) {
        uint64_t mid = sp + ((ep - sp) >> 1);

        if (rank_samples->get(mid) < i) {
            if (sp == mid) break;
            sp = mid;
        }
        else {
            if (mid == 0) break;
            ep = mid - 1;
        }
    }

    uint64_t j = sp * sample_length;

    uint64_t rank = rank_samples->get(sp);

    uint64_t offset = offset_samples->get(sp);

    uint64_t klass = 0;

    while (j < nblocks) {
        klass = classes->get(j);

        if (rank + klass > i) break;

        rank += klass;

        offset += offset_bits[klass];

        j++;
    }

    uint64_t position = j * block_length;

    uint64_t block;

    assert(klass > 0); // Need at least one 1 bit.

    if (klass == block_length) {
        block = (1ULL << block_length) - 1;
    }
    else {
        offset = offsets->get(offset, offset_bits[klass]);

        block = compute_block(klass, offset);
    }

    j = 0;
    while (j < block_length) {
        if (block & 1) rank++;
        if (rank > i) break;
        block >>= 1;
        position++;
        j++;
    }

    return position;
}

void rrr::impl::save(FILE *fp) const {
    assert(fp);

    encode(block_length, fp);
    encode(sample_length, fp);
    encode(nbits, fp);
    encode(none_bits, fp);

    classes->save(fp);
    offsets->save(fp);

    rank_samples->save(fp);
    offset_samples->save(fp);
}

uint64_t rrr::impl::size() const {
    return
        encode(block_length, NULL) +
        encode(sample_length, NULL) +
        encode(nbits, NULL) +
        encode(none_bits, NULL) +
        classes->size() +
        offsets->size() +
        rank_samples->size() +
        offset_samples->size();
}

// Represent n bits from d in blocks of size b sampling every s blocks.
rrr::rrr(uint64_t *d, uint64_t n, uint64_t b, uint64_t s) :
    pimpl(new impl(d, n, b, s)) {
}

rrr::rrr(FILE *fp) :
    pimpl(new impl(fp)) {
}

rrr::~rrr() {
    delete pimpl;
}

uint64_t rrr::access(uint64_t i) const {
    return pimpl->access(i);
}

uint64_t rrr::rank0(uint64_t i) const {
    return pimpl->rank0(i);
}

uint64_t rrr::rank1(uint64_t i) const {
    return pimpl->rank1(i);
}

uint64_t rrr::select0(uint64_t i) const {
    return pimpl->select0(i);
}

uint64_t rrr::select1(uint64_t i) const {
    return pimpl->select1(i);
}

void rrr::save(FILE *fp) const {
    pimpl->save(fp);
}

uint64_t rrr::size() const {
    return pimpl->size();
}

} // end namespace bvector
