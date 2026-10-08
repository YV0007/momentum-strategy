# Replication: final strategy vs the paper's published monthly returns

_Generated from the saved train and test backtests. Paper: FAQ Q24 table, net of costs. Feb 2025 excluded as possibly partial. Figure: results/figures/replication.png._

- Months compared: 109 (Jan 2016 to Jan 2025)
- Correlation of monthly returns: 0.988
- Regression ours = a + b x paper: slope 0.999 (t vs 1: -0.08)
- Mean monthly difference: -0.02% (t -0.26): no bias
- Tracking error: 2.21% per year; median absolute monthly difference 0.29%
- Months with the same sign: 95%; months differing by more than 1%: 13, more than 2%: 1

## Cumulative and annual returns

|                        | cumulative_ours   | cumulative_paper   | annual_ours   | annual_paper   |
|:-----------------------|:------------------|:-------------------|:--------------|:---------------|
| Train 2016–2022        | 178.8%            | 196.1%             | 15.8%         | 16.8%          |
| Test Jan 2023–Jan 2025 | 86.7%             | 79.1%              | 34.9%         | 32.3%          |
| All shared months      | 420.5%            | 430.5%             | 19.9%         | 20.2%          |

## By year

|      | ours   | paper   | difference   |
|-----:|:-------|:--------|:-------------|
| 2016 | -12.7% | -12.8%  | 0.1%         |
| 2017 | -6.7%  | -7.0%   | 0.3%         |
| 2018 | 54.3%  | 60.9%   | -6.6%        |
| 2019 | 9.6%   | 6.8%    | 2.8%         |
| 2020 | 23.1%  | 26.6%   | -3.6%        |
| 2021 | 30.3%  | 34.9%   | -4.6%        |
| 2022 | 26.2%  | 24.3%   | 1.9%         |
| 2023 | 41.3%  | 37.3%   | 4.0%         |
| 2024 | 34.0%  | 32.1%   | 2.0%         |
| 2025 | -1.4%  | -1.2%   | -0.2%        |

## Largest monthly differences

|         | ours   | paper   | difference   |
|:--------|:-------|:--------|:-------------|
| 2024-12 | 8.2%   | 5.7%    | 2.5%         |
| 2016-07 | -4.9%  | -3.1%   | -1.8%        |
| 2019-06 | 1.2%   | -0.5%   | 1.7%         |
| 2016-01 | 0.5%   | -1.0%   | 1.5%         |
| 2021-11 | -4.7%  | -3.2%   | -1.5%        |
| 2022-12 | 2.1%   | 0.7%    | 1.4%         |
