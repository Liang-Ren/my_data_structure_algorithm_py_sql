1.  my_soc(moving_window)
        import sys, json 
        from datetime import datetime, timedelta
        from collections import defaultdict 
        class my_soc:
            def __init__(self):
                self.TI = {"aaa": {"threat": "ransom", "severity": "high"}, "bbb": {"threat": "worm", "severity": "medium"}}
                self.N = 0
                self.alerts = None
                self.deduped = []      
                self.incidents = [] 
            def ingest(self):
                self.N = int(sys.argv[1]) if len(sys.argv) > 1 else 2 
                self.alerts = [json.loads(l) for l in sys.stdin if l.strip()]
                self.alerts = sorted(self.alerts, key=lambda a: datetime.fromisoformat(a['ts']))
            def dedup(self):
                hosttype_time = {}
                for a in self.alerts:
                    (host, type) = (a['host'], a['type']); time = datetime.fromisoformat(a['ts'])
                    if (host, type) not in hosttype_time or time - hosttype_time[(host, type)] > timedelta(minutes=5):
                        self.deduped.append(a)   
                        hosttype_time[(host, type)] = time 
            def group(self):
                hash_hosttime = defaultdict(list)
                for d in self.deduped:          
                    hash = d['hash']; (host, time) = (d['host'], datetime.fromisoformat(d['ts']))
                    hash_hosttime[hash] = [(host, ts) for (host, ts) in hash_hosttime[hash] if time - ts < timedelta(minutes=10)]
                    hash_hosttime[hash].append((host, time))
                    hosts = sorted(set([host for host, _ in hash_hosttime[hash]]))
                    if len(hosts) > self.N:
                        self.incidents.append(
                            {'name': 'threat_spread',
                             'hash': hash,
                             'hosts': list(hosts),
                             'starttime': min([ts for _, ts in hash_hosttime[hash]]),
                             'endtime': time.isoformat()}
                        )
            def enrich(self, alert):
                alert['threat'] = self.TI[alert['hash']]['threat']
                alert['severity'] = self.TI[alert['hash']]['severity']
                return alert
2.  my_requests     
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        class my_requests:
            def __init__(self, token="YOUR_TOKEN"):
                self.headers = {"Authorization": f"Bearer {token}"}
                self.session = requests.Session()
                self.session.mount('https://', HTTPAdapter(Retry(total=3, backoff_factor=1)))
            def get(self, url, params={}):
                try:
                    resp = self.session.get(url, headers=self.headers, params=params, timeout=10)
                    resp.raise_for_status()
                    return resp.json() 
                except requests.RequestException as e:
                    print("GET failed:", e)
                    return None
            def post(self, url, payload={}):
                try:
                    resp = self.session.post(url, headers=self.headers, json=payload, timeout=10)
                    resp.raise_for_status()
                    return resp.json()
                except requests.RequestException as e:
                    print("POST failed:", e)
                    return None
            def get_all_pages(self, url):
                page = 1
                all_results = []
                while True:
                    try:
                        resp = self.session.get(url, headers=self.headers, params={"page": page, "size": 100}, timeout=10)
                        resp.raise_for_status()
                        data = resp.json()
                        if not data["results"]: break
                        else: all_results.extend(data["results"])
                        if not data["next_page"]: break
                        else: page += 1
                    except requests.RequestException as e:
                        print(f"Other error on page {page}:", e)
                        break
                return all_results
3.  class lru_cache:
        def __init__(self, capacity):
            from collections import OrderedDict
            import threading
            self.cache = OrderedDict()
            self.capacity = capacity
            self.lock = threading.Lock()
        def get(self, key):
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]
        def put(self, key, value):
            with self.lock:
                if key in self.cache:
                    self.cache.move_to_end(key)
                self.cache[key] = value
                if len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)
4.  class token_bucket:
        def __init__(self, rps=2, burst=10): 
            import time, threading
            self.rps = rps; self.burst = burst; self.tokens = burst
            self.ts = time.time(); self.lock = threading.Lock()
        def take(self, n=1):
            import time 
            with self.lock:
                    now = time.time()
                    self.tokens = min(self.burst, self.tokens + (now - self.ts) * self.rps)
                    self.ts = now
                    if self.tokens >= n:
                        self.tokens -= n
                        return True
                    return False
    class MultiTenant:
        def __init__(self, default_rps=10, default_burst=20, global_rps=1000, global_burst=2000):
            from collections import defaultdict
            self.tenants = defaultdict(lambda: token_bucket(default_rps, default_burst))
            self.global_bucket = token_bucket(global_rps, global_burst)
        def allow(self, tenant, cost=4):
            if self.global_bucket.take(cost) and self.tenants[tenant].take(cost): return True
            return False
5.  class min_stack:
        def __init__(self): 
            self.stk=[]; self.min_stk=[]
        def push(self, val):
            self.stk.append(val)
            if not self.min_stk or val <= self.min_stk[-1]: self.min_stk.append(val)
        def pop(self):
            x = self.stk.pop()
            if x == self.min_stk[-1]: self.min_stk.pop()
            return x
        def peek_min(self):
            if not self.min_stk: return None
            return self.min_stk[-1]
6.  class de_queue:
        def __init__(self): 
            self.in_stk, self.out_stk = [], []
        def _move(self): 
            if not self.out_stk:
                while self.in_stk:
                    self.out_stk.append(self.in_stk.pop())
        def push(self, val): 
            self.in_stk.append(val)
        def pop(self): 
            self._move()
            return self.out_stk.pop()
        def peek(self): 
            self._move()
            return self.out_stk[-1]
        def is_empty(self): 
            return not self.in_stk and not self.out_stk
7.  def left_rotation(a, p):
        p %= len(a)
        return a[p:] + a[:p]
8.  def top_k(nums, k):
        import heapq 
        return heapq.nlargest(k, nums)
8.  def kth_largest(nums, k):
        import heapq 
        return heapq.nlargest(k, nums)[-1]
8.  def k_closest_to_origin(points, k):
        import heapq
        return heapq.nsmallest(k, points, key=lambda p: p[0]**2 + p[1]**2)
9.  def bottom_k_bubble_sort(nums, k):
        n = len(nums)
        for i in range(n):
            swap = False
            for j in range(n-i-1):
                if nums[j] > nums[j + 1]: nums[j], nums[j + 1] = nums[j + 1], nums[j]; swap = True
            if not swap: break
        return nums[:k]
10. def valid_parentheses(s):
        map = {')':'(', ']':'[', '}':'{'}; stk = []
        for ch in s:
            if ch in '([{': stk.append(ch)
            elif not stk or stk[-1] != map.get(ch, '@'): return false
            else: stk.pop()
        return not stk
11. def valid_palindrome(s):
        filtered = [ch.lower() for ch in s if ch.isalnum()]
        return filtered == filtered[::-1]
12. def remove_spaces(s):
        import re 
        s = s.strip()
        return re.sub(r"\s+", " ", s)
12. def find_urls(s):
        import re
        return re.findall(r'https?://\S+', s)
13. def translate(s):
        return s.translate({ord('A'): 'a', ord('B'): 'b'})
14. def group_anagrams(strs):
        from collections import defaultdict
        m = defaultdict(list)
        for s in strs:
            m[tuple(sorted(s))].append(s)
        return list(m.values())
15. def top_k_frequent(nums, k):
        from collections import Counter
        return [x for x, _ in Counter(nums).most_common(k)]
16. def valid_anagram(s, t):
        from collections import Counter
        return Counter(s) == Counter(t)
17. def subarray_max(nums):
        best = cur = nums[0]
        for num in nums[1:]:
            cur += num
            cur = max(num, cur)
            best = max(best, cur)
        return best
18. def max_profit(nums):
        low, p = float('inf'), float('-inf')
        for num in nums:
            low = min(low, num)
            p = max(p, num - low)
        return p
18. def erase_overlap_intervals(intervals):
        intervals.sort(key=lambda x: x[1])
        end = float('-inf'); count = 0
        for s, e in intervals:
            if s < end: count += 1; else: end = e
        return count
18. def min_arrow_shots_intervals(self, points):
        points.sort(key=lambda x: x[1])
        end = float('-inf'); count = 0
        for s, e in points:
            if s > end: count += 1; end = e
        return count
19. def two_sum(nums, t):
        num_index = {}
        for index, num in enumerate(nums):
            diff = t - num
            if diff in num_index: return [num_index[diff], index]
            num_index[num] = index
        return []
20. def subarray_sum_count(nums, t):
        from collections import defaultdict 
        count = 0; subsum = 0
        subsum_count = defaultdict(int)
        subsum_count[0] = 1
        for num in nums:
            subsum += num; diff = subsum - t
            if diff in subsum_count: count += subsum_count[diff]
            subsum_count[subsum] += 1
        return count  
20. def subarray_sum_list(nums, t): 
        l = 0; sum = 0; res = []
        for r, v in enumerate(nums):
            sum += v
            while l <= r and sum > t: sum -= nums[l]; l += 1
            if sum == t: res.append(nums[l:r+1])
        return res 
21. def product_except_self(nums): 
        n = len(nums); a = [1] * n; p = 1
        for i in range(n):
            a[i] = p; p *= nums[i]
        p = 1
        for i in range(n - 1, -1, -1):
            a[i] *= p; p *= nums[i]
        return a
21. def is_valid_sudoku(board):
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]
        for i in range(9):
            for j in range(9):
                num = board[i][j]
                if num == '.': continue
                if num in rows[i] or num in cols[j] or num in boxes[(i//3)*3 + j//3]: return False
                rows[i].add(num)
                cols[j].add(num)
                boxes[(i//3)*3 + j//3].add(num)
        return True
22. def daily_temperatures(T):
        stk = []; a = [0] * len(T)
        for i, t in enumerate(T):
            while stk and T[stk[-1]] < t: 
                x = stk.pop(); a[x] = i - x
            stk.append(i)
        return a
22. def largest_area(H):
        H.append(0)
        stk = []; mx = 0
        for i, h in enumerate(H):
            while stk and H[stk[-1]] > h:
                x = stk.pop(); mx = max(mx, H[x] * (i if not stk else i - stk[-1] - 1))
            stk.append(i)
        return mx
23. def longest_substring_without_repeat(s):
        char_index = {}
        left = 0; max_length = 0
        for right, char in enumerate(s):
            if char in char_index and char_index[char] >= left: left = char_index[char] + 1
            char_index[char] = right
            max_length = max(max_length, right - left + 1)
        return max_length
24. def shortest_substring_with_target(s, t):
        from collections import Counter
        n, missing, l, x, y = Counter(t), len(t), 0, 0, 0
        for r, c in enumerate(s, 1):
            missing -= n[c] > 0; n[c] -= 1
            if missing == 0:
                while l < r and n[s[l]] < 0: n[s[l]] += 1; l += 1
                if y == 0 or r - l < y - x: y, x = r, l
                n[s[l]] += 1; l += 1; missing = 1
        return s[x:y]
24. def max_container(heights):
        l, r = 0, len(heights) - 1
        mx = 0
        while l < r:
            mx = max(mx, min(heights[l], heights[r]) * (r - l))
            if heights[l] < heights[r]: l += 1; else: r -= 1
        return mx
25. def reverse_list(head):
        prev = None; cur = head
        while cur:
            cur.next, prev, cur = prev, cur, cur.next
        return prev
26. def merge_two_lists(l1, l2):
        dummy = tail = Node(0)
        while l1 and l2:
            if l1.val < l2.val: tail.next, l1 = l1, l1.next
            else: tail.next, l2 = l2, l2.next
            tail = tail.next
        tail.next = l1 or l2
        return dummy.next
27. def has_cycle(head):
        slow = fast = head
        while fast and fast.next:
            slow = slow.next; fast = fast.next.next
            if slow == fast:
                return True
        return False
27. def find_nth_from_end(head, n):
        fast = slow = head
        for _ in range(n):
            if not fast: return None
            fast = fast.next
        while fast: fast = fast.next; slow = slow.next
        return slow
28. def reorder_list(head):
        slow = fast = head
        while fast and fast.next:
            slow = slow.next; fast = fast.next.next
        prev = None; cur = slow.next; slow.next = None
        while cur:
            cur.next, prev, cur = prev, cur, cur.next
        p1, p2 = head, prev
        while p2:
            n1, n2 = p1.next, p2.next
            p1.next = p2; p2.next = n1
            p1, p2 = n1, n2
29. def climb_stairs(n):
        a, b = 1, 1
        for _ in range(n):
            a, b = b, a + b
        return a
26. def climb_stairs_recursion(n):
        if n <= 1: return 1
        return climb_stairs(n - 1) + climb_stairs(n - 2)
30. class TreeNode: 
        __init__(val, left, right): self.val, self.left, self.right = val, left, right
    class Tree: 
        def max_depth(root):
            if not root: return 0
            return 1 + max(self.max_depth(root.left), self.max_depth(root.right))
        def tree_bfs(root):
            from collections import deque
            q = deque([root])
            while q:
                for _ in range(len(q)):
                    node = q.popleft(); print(node.val)
                    if node.left: q.append(node.left)
                    if node.right: q.append(node.right)
        def tree_dfs(root):
            if not root: return
            print(root.val); 
            tree_dfs(root.left); tree_dfs(root.right)
31. def is_bst_dfs(root, low=float('-inf'), high=float('inf')):
        if not root: return True
        if not (low < root.val < high): return False
        return is_bst_dfs(root.left, low, root.val) and is_bst_dfs(root.right, root.val, high)
32. def common_ancestor_bst(root, p, q):
        while root:
            if p.val < root.val and q.val < root.val: root = root.left
            elif p.val > root.val and q.val > root.val: root = root.right
            else: return root
33. def graph_dfs(g, s, visited=None):
        if not visited: visited = set()
        visited.add(s); print(s)
        for neighbor in g.get(s):
            if neighbor not in visited:
                graph_dfs(g, neighbor, visited)
34. def num_islands(grid):
        I, J = len(grid), len(grid[0])
        def dfs(i, j):
            if i < 0 or i >= I or j < 0 or j >= J or grid[i][j] == 0: return
            grid[i][j] = 0
            dfs(i+1, j); dfs(i-1, j); dfs(i, j+1); dfs(i, j-1)
        cnt = 0
        for i in range(I):
            for j in range(J):
                if grid[i][j] == 1: cnt += 1; dfs(i, j)
        return cnt
34. def max_island_area(grid):
        I, J = len(grid), len(grid[0])
        def dfs(i, j):
            if i < 0 or i >= I or j < 0 or j >= J or grid[i][j] == 0: return 0
            grid[i][j] = 0
            area = 1; area += dfs(i+1, j); area += dfs(i-1, j); area += dfs(i, j+1); area += dfs(i, j-1)
            return area
        max_area = 0
        for i in range(I):
            for j in range(J):
                if grid[i][j] == 1: max_area = max(max_area, dfs(i, j))
        return max_area
35. def graph_bfs(g, s):
        from collections import deque
        visited, q = {s}, deque([s])
        while q:
            vertex = q.popleft(); print(vertex)
            for neighbor in g.get(vertex):
                if neighbor not in visited:
                    q.append(neighbor); visited.add(neighbor)
36. def shortest_path(edges, start, target):
        from collections import defaultdict, deque 
        g = defaultdict(list)
        for u, v in edges:
            g[u].append(v); g[v].append(u)
        q = deque([[start]]); visited = {start}
        while q:
            path = q.popleft(); vertex = path[-1]
            if vertex == target: return path
            for neighbor in g[vertex]:
                if neighbor not in visited:
                    q.append(path + [neighbor]); visited.add(neighbor)
        return []
37. class GraphNode:
        def __init__(self, val, neighbors = None):
            self.val = val
            self.neighbors = neighbors if neighbors is not None else []
        @staticmethod
        def clone_dfs(n, visited=None):
            if not visited: visited = set()
            visited.add(n)
            copy = GraphNode(n.val)
            for neighbor in n.neighbors:
                if neighbor not in visited:
                    copy.neighbors.append(GraphNode.clone_dfs(neighbor, visited))
            return copy
37. def graph_clone(g): 
        from collections import defaultdict
        ng = defaultdict(list)
        for k in g:
            ng[k] = g[k]
        return ng
38. def binary_search(nums, target):
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target: return mid
            if nums[mid] < target: l = mid + 1
            else: r = mid - 1
        return -1
39. def find_min_rotated(nums):
        l, r = 0, len(nums) - 1
        while l < r:
            mid = (l + r) // 2
            if nums[mid] > nums[r]: l = mid + 1
            else: r = mid
        return nums[l]
40. def subarray_li_disc(nums):
        import bisect 
        lis = []
        for num in nums:
            p = bisect.bisect_left(lis, num)
            if p == len(lis): lis.append(num)
            else: lis[p] = num
        return lis
41. def coin_change_min(coins, amount):
        dp = [0] + [float('inf')] * amount
        for i in coins:
            for j in range(i, amount + 1):
                dp[j] = min(dp[j], dp[j - i] + 1)
        return dp[amount]
42. def word_break(s, words):
        amount = len(s)
        dp = [True] + [False] * (amount)
        for i in range(1, amount + 1):
            for j in range(i):
                if dp[j] and s[j:i] in words: dp[i] = True; break
        return dp[amount]
#1 typeguard
from typeguard import typechecked
@typechecked
def foo(x: int) -> str:
    return str(x)
foo("abc")
#2 pydantic 
from pydantic import BaseModel
from typing import Literal
MyIOCType = Literal["ip", "url", "domain", "hash"]
class MyIOC(BaseModel):
    value: str
    type: MyIOCType = "hash"
#3 json_logging
import json, logging                
from datetime import datetime       
class MyLoggingFormatter(logging.Formatter):
    def format(self, record):  
        return json.dumps({
            "ts": datetime.utcnow().isoformat(),
            "level": record.levelname,                
            "msg": record.getMessage(),               
            "logger": record.name                    
        })
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("info message")
    logging.warning("warning message")
    logging.error("error message")
#4 prometheus_counter
import time
from prometheus_client import Counter, start_http_server
API_CALLS = Counter("api_calls_total", "API calls total", ["api","status"])
if __name__ == '__main__': 
    start_http_server(port=8000)
    while True:
        time.sleep(3600)
#5 anyio_with_token_bucket_async
import anyio
bucket = token_bucket_async(rate=2, capacity=4)
async def cycle():
    async with anyio.create_task_group() as tg:
        for _ in range(10):
            tg.start_soon(bucket.take, 1)
if __name__ == '__main__': 
    anyio.run(cycle) 
#6 token_bucket_async
import anyio, time
class token_bucket_async:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.timestamp = time.time()
        self.lock = anyio.Lock()
    async def take(self, amount = 1):
        async with self.lock:
            now = time.time()
            self.tokens = min(self.capacity, self.tokens + (now - self.timestamp) * self.rate)
            self.timestamp = now
            while self.tokens < amount:
                need = (amount - self.tokens) / self.rate
                await time.sleep(need)
                now = time.time()
                self.tokens = min(self.capacity, self.tokens + (now - self.timestamp) * self.rate)
                self.timestamp = now
            self.tokens -= amount
#7 async_httpx
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt
from prometheus_counter import API_CALLS
@retry(wait=wait_exponential(1), stop=stop_after_attempt(3))
async def get(ioc):
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get("https://httpbin.org/delay/0.2", params={"v": ioc})
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        API_CALLS.labels(api="httpbin.org", status=str(e.status_code)).inc()
#8 parse_firewall_logs
import json
def parse_blocked_logs(log_file):
    blocked_entries = []
    with open(log_file, 'r') as f:
        for line in f:
            if 'action=block' in line:
                entry = {}; 
                for part in line.split():
                    key, value = part.split('='); 
                    entry[key] = value 
                blocked_entries.append(entry)
    return blocked_entries
if __name__ == "__main__":
    with open('firewall_log.txt', 'w') as f:
        f.write("src=10.1.1.2 dst=1.2.3.4 action=block\n")
    print(json.dumps(parse_blocked_logs('firewall_log.txt')))
#9 aggregation(jump_window)
import json 
from collections import defaultdict
from datetime import datetime, timedelta 
def window_start(ts, window):
    epoch = int(ts.timestamp())
    return epoch - (epoch % int(timedelta(minutes=window).total_seconds()))
def aggregation(file):
    buckets = defaultdict(list)
    events = [json.loads(e) for e in open(file, 'r') if e.strip()]
    for event in events:
        host = event['host']
        ts = datetime.fromisoformat(event['ts'])
        win_start = window_start(ts, 5)
        buckets[(host, win_start)].append(event) 
    alerts = []
    for (host, win_start) in buckets:
        severity = min(e['severity'] for e in buckets[(host, win_start)])
        alerts.append({
                        'host': host,
                        'starttime': win_start,
                        'severity': severity,
                        'eventcount': len(buckets[(host, win_start)])
                    })
    return alerts 
#10 user_failed_logins  
from collections import defaultdict, deque
from datetime import datetime, timedelta
def user_failed_logins(logs, threshold=5, minutes=1):
    suspicious = set()
    user_times = defaultdict(deque)
    logs = sorted(logs, key=lambda x: datetime.fromisoformat(x["time"].replace("Z", "+00:00")))
    for l in logs:
        if l.get("logType") != "login_failure": continue
        u = l.get("userName")
        t = datetime.fromisoformat(l.get("time").replace("Z", "+00:00")) 
        q = user_times[u]; q.append(t)
        while q and (t - q[0]) > timedelta(minutes=minutes): q.popleft() 
        if len(q) >= threshold and u not in suspicious: suspicious.add(u) 
    return sorted(suspicious) 
#11 ip_shared_among_users 
from collections import defaultdict, Counter, deque
from datetime import datetime, timedelta
def ip_shared_among_users(logs, threshold=2, minutes=1440):
    logs.sort(key=lambda x: datetime.fromisoformat(x.get('time').replace('Z', '+00:00')))
    ip_timeusers = defaultdict(deque) 
    ip_usercount = defaultdict(Counter) 
    suspicious = set()
    for l in logs:
        t = datetime.fromisoformat(l.get('time').replace('Z', '+00:00'))
        ip = l.get('ip')
        user = l.get('user')
        q = ip_timeusers[ip]; q.append((t, user))
        cnt = ip_usercount[ip]; cnt[user] += 1 
        while q and (t - q[0][0]) > timedelta(minutes=minutes):
            old_user = q.popleft()[1]
            cnt[old_user] -= 1
            if cnt[old_user] == 0: cnt.pop(old_user)
        if len(cnt) >= threshold and ip not in suspicious: suspicious.add(ip)
    return sorted(suspicious) 
#12 average_severity_per_endpoint 
from collections import defaultdict
def average_severity_per_endpoint(logs):
    endpoint_sevs = defaultdict(list)
    for l in logs:
        ep = l.get('endpoint')
        sev = int(l.get('severity'))
        endpoint_sevs[ep].append(sev) 
    return [{ep: sum(sevs) / len(sevs)} for ep, sevs in endpoint_sevs.items()]
#13 ip_used_frequently
from collections import defaultdict
def ip_used_frequently(logs):
    ip_count = defaultdict(int)
    for l in logs:
        ip = l.get('ip')
        ip_count[ip] += 1 
    avg = len(logs) / len(ip_count) 
    return [ip for ip in ip_count if ip_count[ip] > avg]
#14 merge_feeds
from collections import defaultdict 
def merge_feeds(f1, f2):
    feeds = f1 + f2 
    ioctype_feed = defaultdict(dict)
    for f in feeds:
        ioc = f.get('ioc')
        type = f.get('type') 
        ioctype_feed[(ioc, type)] = f 
    return list(ioctype_feed.values()) 
""" 9_DSA_TEMPLATES """
hashing > O(n)
    key_value = {}
    for index, val in enumerate(arr):
        x = ...
        if x in key_value: ... 
        key_value[val] = index 
sliding_window > O(n**2)
    for r, char in enumerate(s): 
        l = 0
        while l < r: 
            if condition: ...
            l += 1 
heap > O(n log k) / O(log n) / O(n)
    heapq.nlargest(k, nums, key=lambda x: x[0]**2 + x[1]**2) > O(n log k)
    heapq.nsmallest(k, nums, key=lambda x: x[0]**2 + x[1]**2) > O(n log k)
    heapq.heapify(nums) > O(n)
    heapq.heappush(h, x) > O(log n)
    heapq.heappop(h) > O(log n) 
node_list > O(n)
    slow = fast = head
    while fast and fast.next: slow = slow.next; fast = fast.next.next 
binary_search > O(log n)
    l, r = 0, len(arr) - 1
    while l <= r:
        mid = (l + r) // 2
        if arr[mid] == target: return mid
        if arr[mid] < target: l = mid + 1
        else: r = mid - 1
tree > O(n)
    dfs
        if not node: return
        dfs(node.left)
        dfs(node.right)
    bfs
        q = deque([root])
        while q:
            for _ in range(len(q)):
                node = q.popleft(); ... 
graph > O(n + m)
    dfs
        if not visited: visited = set()
        visited.add(node)
        for neighbor in node.neighbors:
            if neighbor not in visited:
                dfs(neighbor, visited)
    bfs
        visited, q = {start}, deque([start])
        while q:
            vertex = q.popleft(); ...
            for neighbor in vertex.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor); q.append(neighbor)
greedy_intervals > O(n log n) 
    intervals.sort(key=lambda x: x[1]) 
    end = float('-inf'); count = 0
    for s, e in intervals:
        if s < end: count += 1; else: end = e
mono_stack > O(n) 
    stk = [] 
    for i, val in enumerate(arr):
        while stk and arr[stk[-1]] < val: j = stk.pop(); ...
        stk.append(i) 