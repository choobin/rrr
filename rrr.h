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

#ifndef BVECTOR_RRR_H
#define BVECTOR_RRR_H

#include <cstdint>
#include <cstdio>

namespace bvector{

class rrr {
public:
    rrr(uint64_t *data, uint64_t nbits, uint64_t block_length = 15, uint64_t sample_length = 32);

    rrr(FILE *fp);

    ~rrr();

    uint64_t access(uint64_t i) const;  // B[i]

    uint64_t rank0(uint64_t i) const;   // rank0(i) = |{j ∈ [0, i) : B[j] = 0}|

    uint64_t rank1(uint64_t i) const;   // rank1(i) = |{j ∈ [0, i) : B[j] = 1}|

    uint64_t select0(uint64_t i) const; // select0(i) = max{j ∈ [0, n) | rank0(j) = i}

    uint64_t select1(uint64_t i) const; // select1(i) = max{j ∈ [0, n) | rank1(j) = i}

    void save(FILE *fp) const;

    uint64_t size() const;

private:
    rrr(const rrr&) = delete;
    rrr(const rrr&&) = delete;
    rrr& operator=(const rrr&) = delete;

    class impl;
    impl *pimpl;
};

} // end namespace bvector

#endif // BVECTOR_RRR_H
