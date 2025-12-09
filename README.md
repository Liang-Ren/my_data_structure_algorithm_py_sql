# my_data_structure_algorithm_py_sql

A personal practice notebook that combines:

- Core **data structures** and **algorithm** snippets in Python
- Common **analytics SQL** patterns for user/events/orders data

The goal of this repo is to provide a compact reference you can quickly review before interviews or when designing real systems.

---

## Contents

### Python: `myPy.py`

This file collects many small, focused implementations:

- **Security & infra utilities**
  - `my_soc` – simple SOC alert pipeline: ingest alerts, deduplicate by host/type, group spreading incidents, and enrich with threat intelligence
  - `my_requests` – a thin wrapper around `requests.Session` with retries, basic `get/post` helpers, and pagination (`get_all_pages`)

- **Rate limiting & caching**
  - `lru_cache` – thread-safe LRU cache using `OrderedDict`
  - `token_bucket` & `MultiTenant` – per-tenant + global token buckets for rate limiting (requests-per-second, burst)

- **Stack / Queue / Deque**
  - `min_stack` – stack that supports retrieving the current minimum in O(1)
  - `de_queue` – queue implemented with two stacks (amortized O(1) operations)

- **Array / Heap / Sorting utilities**
  - `left_rotation(a, p)` – array left rotation
  - `top_k`, `kth_largest`, `k_closest_to_origin` – top-K and K-closest problems via `heapq`
  - `bottom_k_bubble_sort` – example of bubble sort and getting the smallest K items

- **String & hashing problems**
  - `valid_parentheses`, `valid_palindrome`, `remove_spaces`, `find_urls`, `translate`
  - `group_anagrams`, `top_k_frequent`, `valid_anagram`

- **Subarrays & prefix sums**
  - `subarray_max` (maximum subarray / Kadane)
  - `max_profit` (best time to buy/sell stock)
  - `subarray_sum_count` (count subarrays with sum = target)
  - `subarray_sum_list` (list subarrays with sum = target)
  - `product_except_self`

- **Matrix / grid / stack patterns**
  - `is_valid_sudoku`
  - `daily_temperatures` (monotonic stack)
  - `largest_area` (largest rectangle in histogram)
  - `num_islands` (DFS on grid)

- **Sliding window & two pointers**
  - `longest_substring_without_repeat`
  - `shortest_substring_with_target` (minimum window substring)
  - `max_container` (container with most water)

- **Linked list & tree utilities**
  - `reverse_list`, `merge_two_lists`, `has_cycle`, `find_nth_from_end`, `reorder_list`
  - `TreeNode` / `Tree` (depth, BFS/DFS), `is_bst_dfs`, `common_ancestor_bst`

> Note: this file is meant as a **study sheet / playground**, not as a polished library. Some snippets are intentionally compact to highlight the core idea.

### SQL: `mySQL.sql`

This file focuses on **analytics SQL for product / growth / retention** scenarios using three logical tables:

- `users(user_id, signup_date, country, device)`
- `events(user_id, event_time, event_type)`
- `orders(order_id, user_id, order_time, amount, status)`

The comments also map each SQL query to an XQL-style pipeline (dataset → filter → stats → window ...), helping you think in both SQL and analytical DSLs.

Included patterns:

1. **DAU last 7 days** – distinct active users per day
2. **Rolling 7-day active users** – window function over daily DAU
3. **D1 retention by signup cohort**
4. **First purchase & days since signup**
5. **30-day RFM (recency, frequency, monetary)**
6. **Top-3 revenue users per country** (group top-K)
7. **Sessionization with 30-min timeout**
8. **Conversion funnel** (view → add_to_cart → purchase)
9. **Basic DML** – insert, truncate, delete with `USING`, update with join
10. **P90 purchase amount** via `percentile_cont`
11. **Churned users** – active 31–60 days ago but not in last 30 days
12. **Repurchase within 30 days of first purchase**
13. **Long ↔ Wide transforms** (pivot/unpivot style)
14. **Weekly GMV & ARPPU**
15. **Rolling 7-day spend per user**
16. **DAU WoW change** (compare to same weekday last week)
17. **Longest active-day streak** (gaps & islands)
18. **Monthly cohort revenue retention matrix** (k-months since signup)

---

## Notes

- This repo is primarily for **learning and reference**, not production use.
- Some implementations are deliberately concise; when using them in real systems, add error handling, tests, and type annotations.