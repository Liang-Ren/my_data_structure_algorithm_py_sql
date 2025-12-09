/* SQL: io->join(CTE)->where->select->group(having)->window->pivot->order->columns->null
   XQL: io->dataset(join)->filter->alter->stats->window->pivot->sort->fields->null
		1. which tables contain the data you need? -> join
		2. do you need subqueries? -> CTE/subquery
			3. how to filter the data to the relevant subset? -> where/filter
			4. do you need to format or transform any columns? -> select/alter
			5. how to group the data to get the metrics you need? -> group(having)/stats
			6. do you need any rolling? -> window
		7. do you need to pivot or unpivot the data? -> pivot/union
		8. how to sort the final results with columns/fields? -> order/sort
		9. do you need to handle null for left join? -> iff/coalesce  */
/* Table Structures */
-- users(user_id BIGINT, signup_date DATE, country TEXT, device TEXT)
-- events(user_id BIGINT, event_time TIMESTAMP, event_type TEXT) 
-- orders(order_id BIGINT, user_id BIGINT, order_time TIMESTAMP, amount NUMERIC, status TEXT)

/* 1) DAU last 7 days (distinct users per day) */
select events.event_time::date as d, count(distinct events.user_id) as dau 
from events 
where events.event_time::date >= current_date - interval '6 days'
group by d 
order by d;

dataset=events
| filter event_time >= now() - 7d
| alter d = to_date(event_time)
| stats dau = count_distinct(user_id) by d
| sort d asc

/* 2) Rolling 7-day active users per day (distinct) — correlated subquery */
with all_days as (
	select d from generate_series(
		(select min(events.event_time::date) from events),
		(select max(events.event_time::date) from events),
		interval '1 day'
	)
), dau as (
	select events.event_time::date as d, count(distinct events.user_id) as users
	from events 
	group by d 
)
select all_days.d, 
	sum(coalesce(dau.users, 0)) over (order by all_days.d range between interval '6 days' preceding and current row) as rolling_users
from all_days 
left join dau using(d)
order by all_days.d;

dataset=events
| alter d = to_date(event_time)
| stats dau = count_distinct(user_id) by d
| moving_sum(dau, 7d, d) as rolling_users
| fields d, rolling_users

/* 3) D1 retention by signup cohort */
with d as (
	select users.signup_date as cohort,
			count(distinct users.user_id) filter(where events.event_time::date = users.signup_date) as d0,
			count(distinct users.user_id) filter(where events.event_time::date = users.signup_date + interval '1 day') as d1
	from users
	join events using(user_id)
	group by cohort
)
select cohort, d1::numeric / nullif(d0, 0) as d1_retention
from d 
order by d;

dataset=users
| join type=inner (dataset=events | alter d = to_date(event_time)) on user_id
| stats d0 = count_distinct(user_id) if(d == signup_date) as d0,
         d1 = count_distinct(user_id) if(d == signup_date + 1d) as d1
    by signup_date
| alter d1_retention = d1 / iff(d0 == 0, null, d0)
| sort signup_date asc

/* 4) First purchase date and days from signup */
with first_order as (
	select user_id, min(orders.order_time::date) as first_order_date
	from orders
	group by user_id
)
select users.user_id, users.signup_date as cohort, coalesce(first_order.first_order_date - users.signup_date, -1) as days
from users
left join first_order using(user_id)
order by users.signup_date, users.user_id;

dataset=users
| join type=left (
    dataset=orders
    | stats first_order_date = min(to_date(order_time)) by user_id
) on user_id
| alter days = iff(is_null(first_order_date), -1, first_order_date - signup_date)
| fields user_id, cohort = signup_date, days
| sort cohort asc, user_id asc

/* 5) Last 30d R/F/M metrics */
select user_id, max(order_time::date) as recency, count(orders.order_id) as frequency, sum(amount) as monetary
from orders 
where status = 'paid' and order_time::date >= current_date - interval '30 days'
group by user_id;

dataset=orders
| filter status == "paid" and order_time >= now() - 30d
| stats recency = max(to_date(order_time)), frequency = count(order_id), monetary = sum(amount) by user_id

/* 6) Top-3 revenue users per country (group top-K) */
WITH o AS (
  SELECT users.country, users.user_id, SUM(orders.amount) AS amount
  FROM users
  JOIN orders ON users.user_id = orders.user_id AND orders.status = 'paid'
  GROUP BY users.country, users.user_id
), r AS (
  SELECT country, user_id, amount, ROW_NUMBER() OVER (PARTITION BY country ORDER BY amount DESC) AS ranking
  FROM o 
)
SELECT country, user_id
FROM r 
WHERE ranking <= 3
ORDER BY country, ranking;

dataset=orders
| filter status == "paid"
| join type=inner (dataset=users) on user_id
| stats amount = sum(amount) by country, user_id
| sort country asc, amount desc
| stats top3 = array_agg(user_id, 3) by country

/* 7) Sessionization (30-min timeout) + session count */
with prev as (
	select user_id, events.event_time, lag(events.event_time) over (partition by user_id order by events.event_time) as prev_time
	from events
), diff as (
	select user_id, case when (prev_time is null) or (event_time - prev_time > interval '30 minutes') then 1 end as new_session
	from prev
)
select user_id, sum(new_session) as sessions
from diff
group by user_id
order by user_id; 

dataset=events
| sort user_id asc, event_time asc
| alter prev_time = lag(event_time, 1, user_id)
| alter new_session = iff(is_null(prev_time) or event_time - prev_time > 30m, 1, 0)
| stats sessions = sum(new_session) by user_id
| sort user_id asc

/* 8) Conversion funnel (view → add_to_cart → purchase) */
with stats as (
	select count(distinct user_id) filter(where event_type = 'view') as viewed,
		count(distinct user_id) filter(where event_type = 'add_to_cart') as added,
		count(distinct user_id) filter(where event_type = 'purchase') as purchased
	from events
)
select added::numeric / nullif(viewed, 0) as add_view_ratio,
		purchased::numeric / nullif(viewed, 0) as pur_view_ratio,
		purchased::numeric / nullif(added, 0) as pur_add_ratio
from stats;

dataset=events
| stats viewed = count_distinct(user_id) if(event_type == "view"),
         added = count_distinct(user_id) if(event_type == "add_to_cart"),
         purchased = count_distinct(user_id) if(event_type == "purchase")
| alter add_view_ratio = added / iff(viewed == 0, null, viewed),
        pur_view_ratio = purchased / iff(viewed == 0, null, viewed),
        pur_add_ratio = purchased / iff(added == 0, null, added)
| fields add_view_ratio, pur_view_ratio, pur_add_ratio

/* 9) Other operations in PostgreSQL */
INSERT INTO users (user_id, signup_date, country, device)
	VALUES (2, '2025-09-18', 'US', 'Android'), (3, '2025-09-17', 'JP', 'Web');
TRUNCATE TABLE users CASCADE; 
DELETE FROM orders
	USING users
	WHERE orders.user_id = users.user_id
	  AND users.country = 'JP';
UPDATE orders
	SET status = 'canceled'
	FROM users
	WHERE orders.user_id = users.user_id
	  AND users.country = 'JP';

/* 10) P90 purchase amount (exact) */
select percentile_cont(0.9) within group (order by amount) as P90
from orders
where status = 'paid'; 

dataset=orders
| filter status == "paid"
| stats p90_amount = percentile(amount, 90)

/* 11) Churned: active in days 31–60, not active in last 30 days */
with prev as (
	select distinct user_id
	from events
	where events.event_time::date >= current_date - interval '60 days'
		and events.event_time::date <= current_date - interval '31 days'
), cur as (
	select distinct user_id
	from events
	where events.event_time::date >= current_date - interval '30 days'
)
select prev.user_id 
from prev 
left join cur using(user_id)
where cur.user_id is null;

dataset=events
| alter d = to_date(event_time)
| filter d >= now() - 60d and d <= now() - 31d
| stats by user_id
| join type=left (dataset=events
				  | alter d = to_date(event_time)
				  | filter d >= now() - 30d
				  | stats by user_id
				  ) on user_id
| filter is_null(cur.user_id)
| fields user_id

/* 12) Repurchase in 30 days after first purchase */
with first_purchase as (
	select user_id, min(orders.order_time::date) as first_date, min(orders.order_id)
	from orders 
	where status = 'paid'
	group by user_id
), repurchase as (
	select first_purchase.user_id, count(order.order_id) as cnt
	from first_purchase
	left join orders using(user_id)
	where orders.order_id <> first_purchase.order_id 
		and orders.order_time::date - first_purchase.first_date <= 30 
	group by first_purchase.user_id
)
select count(distinct user_id) filter(where cnt > 0)::numeric / nullif(count(distinct user_id), 0) as repur_ratio
from repurchase;

dataset=orders
| filter status == "paid"
| stats first_date = min(to_date(order_time)) by user_id
| join type=left (
    dataset=orders
    | filter status == "paid"
    | alter d = to_date(order_time)
) on user_id
| stats repurchase_users = count_distinct(user_id) if(d is not null and d > first_date and d <= first_date + 30d), total_users = count_distinct(user_id)
| alter repur_ratio = repurchase_users / iff(total_users == 0, null, total_users)
| fields repur_ratio

/* 13) Long <-> Wide */
--| user_id	event_type	done   <->   user_id	viewed	added	purchased
--| 1	      viewed	   2              1	       2	    0	    1
--| 1	      added	       0              2	       1	    1	    0
--| 1	      purchased	   1
--| 2	      viewed	   1
--| 2	      added	       1
--| 2	      purchased	   0
-- Long -> Wide: sum + group by
SELECT user_id, 
  sum(coalesce(done, 0)) FILTER (WHERE event_type = 'view') AS viewed,
  sum(coalesce(done, 0)) FILTER (WHERE event_type = 'add_to_cart') AS added,
  sum(coalesce(done, 0)) FILTER (WHERE event_type = 'purchase') AS purchased
FROM long_table
GROUP BY user_id;
-- Wide -> Long: union
select user_id, 'view' as event_type, viewed as done from wide_table
union all 
select user_id, 'add_to_cart' as event_type, added as done from wide_table
union all 
select user_id, 'purchase' as event_type, purchased as done from wide_table;

/* 14) Weekly GMV and ARPPU */
select date_trunc('week', order_time) as wk,
		sum(amount) as gmv, 
		sum(amount)::numeric / nullif(count(distinct user_id), 0) as arppu
from orders 
where status = 'paid'
group by wk
order by wk;

dataset=orders
| filter status == "paid"
| alter wk = bin(order_time, 7d) 
| stats gmv = sum(amount), arppu = sum(amount) / count_distinct(user_id) by wk
| sort wk asc

/* 15) Rolling 7-day spend per user */
with all_days as (
	select user_id, d
	from users, generate_series(
		(select min(order_time::date) from orders),
		(select max(order_time::date) from orders),
		interval '1 day'
	) as d
), dedup as (
	select user_id, order_time::date as d, sum(amount) as amount
	from orders 
	where status = 'paid'
	group by user_id, d 
)
select all_days.user_id, all_days.d, 
		sum(coalesce(dedup.amount, 0)) over (partition by user_id order by all_days.d range between interval '6 days' preceding and current row) as rolling
from all_days
left join dedup using(user_id, d) 
order by all_days.d; 

dataset=orders
| filter status == "paid"
| alter d = to_date(order_time)
| stats amount = sum(amount) by user_id, d
| moving_sum(amount, 7d, d, user_id) as rolling
| fields user_id, d, rolling

/* 16) DAU WoW change (compare to same weekday previous week) */
with all_days as (
	select d
	from generate_series(
		(select min(order_time::date) from orders),
		(select max(order_time::date) from orders),
		interval '1 day'
	) as d
), dedup as (
	select event_time::date as d, count(distinct user_id) as users
	from events
	group by d 
), prev as (
	select all_days.d, coalesce(dedup.users, 0) as cur_users, 
			lag(dedup.users, 7, 0) over (order by all_days.d) as prev_users
	from all_days
	left join dedup using(d)
)
select d, (cur_users - prev_users)::numeric / nullif(prev_users, 0) as WoW
from prev
order by d;

dataset=events
| alter d = to_date(event_time)
| stats users = count_distinct(user_id) by d
| sort d asc
| alter prev_users = lag(users, 7)
| alter WoW = iff(prev_users == 0, null, (users - prev_users) / prev_users)
| fields d, WoW

/* 17) Longest active-day streak (gaps & islands) */
with dedup as (
	select distinct user_id, event_time::date as d
	from events 
), ranking as (
	select dedup.user_id, dedup.d - row_number() over (partition by dedup.user_id order by dedup.d) * interval '1 day' as grp_date
	from dedup
), counting as (
	select user_id, count(*) as cnt
	from ranking 
	group by user_id, grp_date
)
select user_id, max(cnt) as max_consective_days
from counting
group by user_id;

dataset=events
| alter d = to_date(event_time)
| stats days = array_distinct(d) by user_id
| alter streaks = [_max_consecutive_](days)
| fields user_id, streaks

/* 18) Monthly cohort revenue retention matrix (k-months since signup) */
with months as (
	select date_trunc('month', users.signup_date) as cohort, 
		date_trunc('month', orders.order_time) as order_month,
		sum(orders.amount) as month_amount
	from users 
	join orders using(user_id)
	group by cohort, order_month
), ranking as (
	select cohort, month_amount, 
		(date_part('year',order_month) - date_part('year',cohort))*12 + (date_part('month',order_month) - date_part('month',cohort)) as k
	from months 
)
select cohort,
		sum(case when k=0 then month_amount else 0 end) as m0,
		sum(case when k=1 then month_amount else 0 end) as m1,
		sum(case when k=2 then month_amount else 0 end) as m2,
		sum(case when k=3 then month_amount else 0 end) as m3
from ranking
group by cohort
order by cohort;

dataset=users
| join type=inner (dataset=orders) on user_id
| alter cohort = bin(signup_date, 1M)
| alter order_month = bin(order_time, 1M)
| alter k = months_between(order_month, cohort)
| stats month_amount = sum(amount) by cohort, k
| pivot cohort, k, month_amount