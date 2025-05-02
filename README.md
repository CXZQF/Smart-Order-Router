# Cont & Kukanov Smart Order Router Backtest

## Project Overview

This project implements and backtests a smart order router (SOR) based on the static allocation model proposed by Cont & Kukanov (2014). The implementation makes efficient decisions on splitting orders between market orders and limit orders across multiple venues to minimize execution costs.

## Code Structure

- **`Venue` Class**: Represents a trading venue with ask price, size, fee, and rebate information.
- **Data Processing Functions**:
  - `load_and_preprocess_data()`: Handles data cleaning and initial processing of L1 data.
  - `create_snapshots()`: Converts data into chronological market snapshots.
- **Core Allocation Logic**:
  - `compute_cost()`: Calculates expected cost for a given order split.
  - `allocate()`: Implements the Cont-Kukanov static allocation algorithm.
  - `simulate_execution()`: Backtests the allocator over a sequence of snapshots.
- **Baseline Strategies**:
  - `best_ask_strategy()`: Naïvely sends orders to the venue with the lowest ask price.
  - `twap_strategy()`: Time-Weighted Average Price strategy with 60-second buckets.
  - `vwap_strategy()`: Volume-Weighted Average Price strategy weighted by displayed ask size.
- **Parameter Optimization**:
  - `evaluate_parameter_set()`: Evaluates a single set of risk parameters.
  - `parameter_search()`: Two-phase adaptive search to find optimal parameters.
- **Results Generation**:
  - `generate_results()`: Formats final results as specified in the task description.

## Parameter Tuning Approach

The algorithm searches for optimal values of three risk parameters:
- `lambda_over`: Penalty for executing more shares than the target slice (0.001-0.2).
- `lambda_under`: Penalty for executing fewer shares than the target slice (0.001-0.2).
- `theta_queue`: Queue-risk penalty related to mis-execution (0.0001-0.05).

The parameter search employs a two-phase approach:
1. **Exploration Phase**: Stratified sampling to broadly identify promising regions.
2. **Refinement Phase**: Focused random search within a narrower region around the best parameters found in phase 1.

The search leverages multiprocessing to parallelize parameter evaluations, significantly reducing runtime while maintaining solution quality.



## Suggested Improvement: Realistic Queue Position Modeling

The current implementation assumes that allocated limit orders up to the displayed size are filled instantly at the ask price. A significant improvement would be to model the queue position of the simulated order:

1. **Position Estimation**: When an order slice is allocated to a venue, estimate its position in the queue based on the displayed size already present.

2. **Fill Dynamics**: Track incoming market sell orders (aggressor flow) from subsequent data points. The simulated order slice only gets filled if the aggressor volume consumes the queue up to and including the estimated position before the price moves or the order is implicitly cancelled.

3. **Implementation Approach**: This would require tracking a more detailed state of each order and refining the fill simulation logic to account for partial fills at various queue positions. The order outflow model would need to be updated to include queue position information.

This enhancement would more accurately reflect real-world execution behavior, particularly in conditions where queue positioning greatly affects fill probability, and would likely improve the performance of the already cost-effective router.




## Output Format

The script prints a single JSON object to standard output upon completion. 
```json
{
  "best_parameters": {
    "lambda_over": 0.0554771892225554,
    "lambda_under": 0.09696509114785437,
    "theta_queue": 0.0030412833049502526
  },
  "allocator": {
    "cash_spent": 1113700.0,
    "avg_price": 222.74
  },
  "baselines": {
    "best_ask": {
      "cash_spent": 1114102.2800000003,
      "avg_price": 222.82045600000006
    },
    "twap": {
      "cash_spent": 1115250.0299999998,
      "avg_price": 223.05000599999997
    },
    "vwap": {
      "cash_spent": 1114102.2800000003,
      "avg_price": 222.82045600000006
    }
  },
  "savings_bp": {
    "vs_best_ask": 3.610799539879539,
    "vs_twap": 13.89849772072899,
    "vs_vwap": 3.610799539879539
  }
}

