/*
 * x86/find_min_cost_path.h - AVX2 implementation of deflate_consider_match
 *
 * Copyright 2026 Google LLC
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef LIB_X86_FIND_MIN_COST_PATH_H
#define LIB_X86_FIND_MIN_COST_PATH_H

#include "cpu_features.h"

#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)

#define ATTRIBUTES	_target_attribute("avx2")

/*
 * Return the minimum of 'cost_to_end[len] + length_costs[len] mod 2^32' over
 * 'start_len <= len <= max_len'.  This requires start_len <= max_len.
 *
 * This is used to find the best length to choose for a match that has multiple
 * lengths available.  For details, see find_min_cost_path_template.h.
 *
 * This implementation processes 8 lengths at a time using AVX2.
 */
static forceinline ATTRIBUTES u32
deflate_consider_match_avx2(const u32 *cost_to_end, const u32 *length_costs,
			    const size_t start_len, const size_t max_len)
{
	size_t len = start_len;
	/* Aligned to avoid a cacheline split */
	static const u32 _aligned_attribute(64) tail_masks[15] = {
		0, 0, 0, 0, 0, 0, 0, 0, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U, ~0U,
	};
	__m256i v_best, v_cost;
	__m128i best128;

	/*
	 * Compute the packed (cost, len) values for the first 8 lengths.  If
	 * fewer than 8 lengths are being considered, this loads entries past
	 * max_len; that's safe because cost_to_end and length_costs have 7
	 * padding entries at the end, and the extra lanes are masked out below.
	 */
	v_best = _mm256_add_epi32(
		_mm256_loadu_si256((const __m256i *)&cost_to_end[len]),
		_mm256_loadu_si256((const __m256i *)&length_costs[len]));
	if (likely(len + 7 >= max_len)) {
		/*
		 * <= 8 lengths: set the lanes past max_len to all 1s so that
		 * they won't be chosen as the minimum.
		 */
		v_best = _mm256_or_si256(
			v_best,
			_mm256_loadu_si256((const __m256i *)&tail_masks[len + 7 - max_len]));
	} else {
		/*
		 * > 8 lengths: process full vectors of 8, then use an
		 * overlapping vector ending at max_len for the last 1..8.
		 */
		len += 8;
		while (len + 7 < max_len) {
			v_cost = _mm256_add_epi32(
				_mm256_loadu_si256((const __m256i *)&cost_to_end[len]),
				_mm256_loadu_si256((const __m256i *)&length_costs[len]));
			v_best = _mm256_min_epu32(v_cost, v_best);
			len += 8;
		}
		len = max_len - 7;
		v_cost = _mm256_add_epi32(
			_mm256_loadu_si256((const __m256i *)&cost_to_end[len]),
			_mm256_loadu_si256((const __m256i *)&length_costs[len]));
		v_best = _mm256_min_epu32(v_cost, v_best);
	}
	/* Reduce the 8 lanes of v_best to a single minimum. */
	best128 = _mm_min_epu32(_mm256_castsi256_si128(v_best),
				_mm256_extracti128_si256(v_best, 1));
	best128 = _mm_min_epu32(best128, _mm_shuffle_epi32(best128, 0x31));
	best128 = _mm_min_epu32(best128, _mm_shuffle_epi32(best128, 0x02));
	return _mm_cvtsi128_si32(best128);
}
#define deflate_consider_match_simd	deflate_consider_match_avx2
#define have_consider_match_simd()	HAVE_AVX2(get_x86_cpu_features())

#endif /* defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER) */

#endif /* LIB_X86_FIND_MIN_COST_PATH_H */
