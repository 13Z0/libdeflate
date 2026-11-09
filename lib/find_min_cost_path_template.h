/*
 * find_min_cost_path_template.h - template for deflate_find_min_cost_path()
 *
 * Copyright 2016 Eric Biggers
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

/*
 * This file is a template parameterized by FUNCNAME, ATTRIBUTES, and
 * deflate_consider_match.
 *
 * It implements the core part of deflate_find_min_cost_path(): finding the
 * minimum-cost path over the graph of match/literal choices for a block.
 *
 * For each position 'i' from 'block_length - 1' to 0, it decides which literal
 * or match to choose at position 'i' to reach end-of-block in the cheapest way.
 * It saves the choice into optimum_items[i] and the corresponding cost to
 * end-of-block into cost_to_end[i].
 *
 * The main bottleneck is considering matches; there can be many match lengths
 * at each position that need to be considered.  To minimize the overhead of
 * this step and enable fast SIMD implementations, we pack costs and lengths
 * together as follows.  First, we preprocess the bare length costs
 * 'c->p.n.costs.length' to create an array 'length_costs', containing:
 *
 *	len | (c->p.n.costs.length[len] << 9)
 *
 * That is, its low 9 (OPTIMUM_OFFSET_SHIFT in the code) bits contain the
 * length, and its high 23 bits contain the length's cost.
 *
 * Then, in 'cost_to_end' we store a shifted cost, using the high 23 bits only:
 *
 *	cost << 9
 *
 * To consider multiple lengths (with the same offset), we just sum the
 * corresponding 'length_costs' and 'cost_to_end' entries, giving:
 *
 *	len | ((cost_to_end + c->p.n.costs.length[len]) << 9)
 *
 * ... and then take the minimum of those to get:
 *
 *	best_len | (best_cost_to_end << 9)
 *
 * This vectorizes well and results in a 'u32' that has both the best cost and
 * the length that gave that cost, without needing to track them separately.
 *
 * However, this scheme gives only 23 bits for the cost, which isn't enough for
 * the entire block.  (In theory, a block's cost can reach 'MAX_BLOCK_LENGTH *
 * MAX_LITLEN_CODEWORD_LEN * BIT_COST', needing just over 26 bits.)
 *
 * But 23 bits is more than sufficient for the *differences* in cost over a
 * window of length DEFLATE_MAX_MATCH_LEN.  (The limit there is
 * MAX_MATCH_SPAN_COST, which takes 16 bits.)  Therefore, we also maintain a
 * value 'base_cost'.  'base_cost' starts at 0.  Each time 'best_cost_to_end'
 * reaches REBASE_THRESHOLD (2^22, which is half of 2^23, leaving plenty of
 * room), we increment 'base_cost' by 'best_cost_to_end - REBASE_MARGIN' where
 * REBASE_MARGIN is 2^16 (chosen as the next power of 2 greater than
 * MAX_MATCH_SPAN_COST + MAX_LENGTH_COST), set 'best_cost_to_end' to
 * REBASE_MARGIN, and update the values in 'length_costs' to be:
 *
 *	len | (((c->p.n.costs.length[len] - base_cost) << 9) mod 2^32)
 *
 * This way, each time we sum length_costs[len] with cost_to_end[i + len] when
 * considering a match, we're really computing:
 *
 *	len | ((cost_to_end + c->p.n.costs.length[len] - base_cost) << 9)
 *
 * Taking the minimum of those values gives the minimum of 'cost_to_end +
 * c->p.n.costs.length[len] - base_cost' mod 2^23, with ties broken by taking
 * the smaller length.  And since 'cost_to_end + c->p.n.costs.length[len] -
 * base_cost' never reaches 2^23 (since REBASE_THRESHOLD left plenty of room),
 * nor can it underflow (since REBASE_MARGIN left enough for
 * MAX_MATCH_SPAN_COST), the 'mod 2^23' makes no difference in the answer;
 * that's why it's safe to truncate the costs in cost_to_end[] to 23 bits.
 *
 * Note: the sentinel entries in cost_to_end[] past block_length also have their
 * cost set to REBASE_MARGIN, ensuring their cost is so high they are never
 * chosen.  (They can be reached only from the last DEFLATE_MAX_MATCH_LEN - 1
 * positions of the block.  No rebase can have occurred yet at those positions,
 * and choosing only literals from them costs less than MAX_MATCH_SPAN_COST,
 * which is less than REBASE_MARGIN.)
 */
static ATTRIBUTES void
FUNCNAME(struct libdeflate_compressor *c, const u32 block_length,
	 const struct lz_match *cache_ptr)
{
	u32 best_cost_to_end = 0;
	u32 base_cost = 0;
	u32 length_costs[DEFLATE_MAX_MATCH_LEN + 1 + 7];
	size_t i;

	/* Initialize length_costs[] from c->p.n.costs.length. */
	STATIC_ASSERT(DEFLATE_MAX_MATCH_LEN <= OPTIMUM_LEN_MASK);
#ifdef __clang__
	#pragma clang loop unroll(disable)
#endif
	for (i = DEFLATE_MIN_MATCH_LEN; i <= DEFLATE_MAX_MATCH_LEN; i++)
		length_costs[i] =
			(c->p.n.costs.length[i] << OPTIMUM_OFFSET_SHIFT) | i;

	/*
	 * Initialize padding entries to accommodate 256-bit SIMD loads.  The
	 * value doesn't matter, since the SIMD code masks out these lanes.
	 */
	for (; i < ARRAY_LEN(length_costs); i++)
		length_costs[i] = 0;

	/*
	 * For each position 'i' from 'block_length - 1' to 0, find the cheapest
	 * way to reach end-of-block from position 'i', and set cost_to_end[i]
	 * and optimum_items[i] accordingly.
	 */
	i = block_length;
	do {
		unsigned num_matches;
		u32 literal;
		u32 best_item;

		cache_ptr--;

		num_matches = cache_ptr->length;
		literal = cache_ptr->offset;

		/* It's always possible to choose a literal. */
		best_cost_to_end += c->p.n.costs.literal[literal];
		best_item = (literal << OPTIMUM_OFFSET_SHIFT) | 1;

		i--;

		/*
		 * Also consider matches, if there are any.  They are available
		 * at cache_ptr - num_matches, sorted in order of strictly
		 * increasing length and offset.
		 *
		 * We consider all possible match lengths that are available,
		 * including ones generated by truncating a longer match.
		 * However, we consider only the smallest offset for which each
		 * length is available, since it's very rare for an offset that
		 * isn't the smallest to be best.
		 */
		if (num_matches) {
			const struct lz_match *match = cache_ptr - num_matches;
			u32 len = DEFLATE_MIN_MATCH_LEN;

			do {
				u32 offset = match->offset;
				u8 offset_slot = c->p.n.offset_slot_full[offset];
				u32 match_len = match->length;
				u32 best, cost_to_end;

				/*
				 * Consider matches at 'offset' with lengths
				 * 'len' through 'match_len'.
				 */
				best = deflate_consider_match(
						&c->p.n.cost_to_end[i],
						length_costs, len, match_len);
				cost_to_end = c->p.n.costs.offset_slot[offset_slot] +
					      (best >> OPTIMUM_OFFSET_SHIFT);
				if (cost_to_end < best_cost_to_end) {
					best_cost_to_end = cost_to_end;
					best_item = (best & OPTIMUM_LEN_MASK) |
						    (offset << OPTIMUM_OFFSET_SHIFT);
				}
				len = match_len + 1;
			} while (++match != cache_ptr);
			cache_ptr -= num_matches;
		}
		/*
		 * Store the absolute cost shifted left by OPTIMUM_OFFSET_SHIFT.
		 * This truncates the cost to its low 23 bits, which is safe, as
		 * explained in this function's comment.
		 */
		c->p.n.cost_to_end[i] =
			(best_cost_to_end + base_cost) << OPTIMUM_OFFSET_SHIFT;
		c->p.n.optimum_items[i] = best_item;

		/*
		 * Increment base_cost if best_cost_to_end has gotten too close
		 * to 2^23.  See this function's comment.
		 */
		STATIC_ASSERT(REBASE_MARGIN > MAX_MATCH_SPAN_COST +
			      MAX_LENGTH_COST);
		STATIC_ASSERT(REBASE_MARGIN < REBASE_THRESHOLD);
		STATIC_ASSERT(REBASE_THRESHOLD + MAX_MATCH_SPAN_COST +
			      MAX_LENGTH_COST <
			      ((u32)1 << (32 - OPTIMUM_OFFSET_SHIFT)));
		if (unlikely(best_cost_to_end >= REBASE_THRESHOLD)) {
			u32 delta = best_cost_to_end - REBASE_MARGIN;
			u32 len;

			best_cost_to_end = REBASE_MARGIN;
			base_cost += delta;
			delta <<= OPTIMUM_OFFSET_SHIFT;
#ifdef __clang__
			#pragma clang loop unroll(disable)
#endif
			for (len = DEFLATE_MIN_MATCH_LEN;
			     len <= DEFLATE_MAX_MATCH_LEN; len++)
				length_costs[len] -= delta;
		}
	} while (i != 0);

	/*
	 * Store the unshifted, absolute cost in cost_to_end[0] as expected by
	 * the caller.
	 */
	c->p.n.cost_to_end[0] = best_cost_to_end + base_cost;
}

#undef FUNCNAME
#undef ATTRIBUTES
#undef deflate_consider_match
