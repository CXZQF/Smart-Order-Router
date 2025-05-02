import pandas as pd
import numpy as np
import json
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count


@dataclass
class Venue:
    """
    Represents a trading venue with associated price, size, fee, and rebate information.

    Follows the structure expected by the allocator function as described in the pseudocode.
    """
    id: int
    ask: float
    ask_size: int
    fee: float = 0.0
    rebate: float = 0.0

    def __repr__(self):
        return f"Venue(id={self.id}, ask={self.ask}, ask_size={self.ask_size}, fee={self.fee}, rebate={self.rebate})"


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the L1 market data.

    For each unique ts_event, keep only the first message per publisher_id (venue).
    Organize data chronologically for back-testing.

    Args:
        file_path: Path to the CSV file with market data

    Returns:
        Preprocessed DataFrame with market snapshots
    """
    df = pd.read_csv(file_path)
    if not pd.api.types.is_datetime64_dtype(df['ts_event']):
        df['ts_event'] = pd.to_datetime(df['ts_event'])

    df = df.sort_values('ts_event')
    df = df.drop_duplicates(subset=['ts_event', 'publisher_id'], keep='first')
    df = df.dropna(subset=['ask_px_00', 'ask_sz_00'])
    df = df[df['ask_sz_00'] > 0]
    df = df[['ts_event', 'publisher_id', 'ask_px_00', 'ask_sz_00']]

    return df


def create_snapshots(df: pd.DataFrame, fee: float = 0.0, rebate: float = 0.0) -> List[Dict[str, Any]]:
    """
    Convert preprocessed DataFrame to a list of market snapshots.
    
    Each snapshot contains the timestamp and venue objects with ask price, size, fee, and rebate.
    
    Args:
        df: Preprocessed DataFrame from load_and_preprocess_data
        fee: Fee value to use for all venues
        rebate: Rebate value to use for all venues
    
    Returns:
        List of snapshot dictionaries ordered by timestamp
    """
    
    valid_df = df.dropna(subset=['ask_px_00', 'ask_sz_00'])
    valid_df = valid_df[valid_df['ask_sz_00'] > 0]
    valid_df['ask_px_00'] = valid_df['ask_px_00'].astype(float)
    valid_df['ask_sz_00'] = valid_df['ask_sz_00'].astype(int)
    
    snapshots = []
    for ts, group in valid_df.groupby('ts_event'):
        venues = [
            Venue(
                id=row.publisher_id,
                ask=row.ask_px_00,
                ask_size=row.ask_sz_00,
                fee=fee,
                rebate=rebate
            )
            for row in group.itertuples(index=False)
        ]
        
        if venues:
            snapshots.append({
                'timestamp': ts,
                'venues': venues
            })
    
    return snapshots


def compute_cost(split: List[int], venues: List[Venue], order_size: int,
                 lambda_over: float, lambda_under: float, theta_queue: float) -> float:
    """
    Computes the cost of a given order split as described in the allocator pseudocode.

    Args:
        split: List of order quantities for each venue
        venues: List of venue objects with ask price, size, fee, and rebate information
        order_size: Target number of shares to execute
        lambda_over: Cost penalty per extra share bought
        lambda_under: Cost penalty per unfilled share
        theta_queue: Queue-risk penalty

    Returns:
        Total expected cost of the split
    """
    executed = 0
    cash_spent = 0

    for i in range(len(venues)):
        exe = min(split[i], venues[i].ask_size)
        executed += exe
        cash_spent += exe * (venues[i].ask + venues[i].fee)
        maker_rebate = max(split[i] - exe, 0) * venues[i].rebate
        cash_spent -= maker_rebate

    underfill = max(order_size - executed, 0)
    overfill = max(executed - order_size, 0)
    risk_pen = theta_queue * (underfill + overfill)
    cost_pen = lambda_under * underfill + lambda_over * overfill

    return cash_spent + risk_pen + cost_pen


def allocate(order_size: int, venues: List[Venue],
             lambda_over: float, lambda_under: float, theta_queue: float) -> Tuple[List[int], float]:
    """
    Allocates an order across multiple venues to minimize cost.
    Implements the algorithm as described in the allocator pseudocode.
    """
    step = 100
    splits = [[]]  # start with an empty allocation list
    
    for v in range(len(venues)):
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size-used, venues[v].ask_size)
            
            for q in range(0, max_v+1, step):
                new_splits.append(alloc + [q])
        
        splits = new_splits
    
    best_cost = float('inf')
    best_split = []
    
    for alloc in splits:
        if sum(alloc) != order_size:
            continue
            
        cost = compute_cost(alloc, venues, order_size, lambda_over, lambda_under, theta_queue)
        if cost < best_cost:
            best_cost = cost
            best_split = alloc
    
    return best_split, best_cost


def simulate_execution(snapshots: List[Dict[str, Any]], order_size: int,
                       lambda_over: float, lambda_under: float, theta_queue: float) -> Dict[str, Any]:
    """
    Simulates the execution of the allocator strategy over a series of market snapshots.

    Args:
        snapshots: List of market snapshots
        order_size: Target number of shares to execute
        lambda_over: Cost penalty per extra share bought
        lambda_under: Cost penalty per unfilled share
        theta_queue: Queue-risk penalty

    Returns:
        Dictionary with execution results including filled shares, cash spent, and average price
    """
    remaining_shares = order_size
    total_cash_spent = 0
    total_shares_filled = 0
    execution_history = []

    for snapshot in snapshots:
        if remaining_shares <= 0:
            break

        venues = snapshot['venues']
        split, _ = allocate(remaining_shares, venues, lambda_over, lambda_under, theta_queue)

        snapshot_executed = 0
        snapshot_cash_spent = 0

        for i, qty in enumerate(split):
            venue = venues[i]
            exe = min(qty, venue.ask_size)
            snapshot_executed += exe
            snapshot_cash_spent += exe * venue.ask 

        total_shares_filled += snapshot_executed
        total_cash_spent += snapshot_cash_spent
        remaining_shares -= snapshot_executed

        execution_history.append({
            'timestamp': snapshot['timestamp'],
            'allocated': split,
            'executed': snapshot_executed,
            'cash_spent': snapshot_cash_spent
        })

    avg_price = total_cash_spent / total_shares_filled if total_shares_filled > 0 else 0

    return {
        'shares_filled': total_shares_filled,
        'cash_spent': total_cash_spent,
        'avg_price': avg_price,
        'remaining_shares': remaining_shares,
        'execution_history': execution_history
    }


def best_ask_strategy(snapshots: List[Dict[str, Any]], order_size: int) -> Dict[str, Any]:
    """
    Implements the "take the best ask" baseline strategy.

    At each snapshot, routes the entire order to the venue with the best (lowest) ask price.

    Args:
        snapshots: List of market snapshots
        order_size: Target number of shares to execute

    Returns:
        Dictionary with execution results
    """
    remaining_shares = order_size
    total_cash_spent = 0
    total_shares_filled = 0
    execution_history = []

    for snapshot in snapshots:
        if remaining_shares <= 0:
            break

        venues = snapshot['venues']
        if not venues:
            continue

        best_venue_idx = min(range(len(venues)), key=lambda i: venues[i].ask)
        best_venue = venues[best_venue_idx]

        exe = min(remaining_shares, best_venue.ask_size)
        cash_spent = exe * best_venue.ask

        total_shares_filled += exe
        total_cash_spent += cash_spent
        remaining_shares -= exe

        execution_history.append({
            'timestamp': snapshot['timestamp'],
            'venue_id': best_venue.id,
            'executed': exe,
            'cash_spent': cash_spent
        })

    avg_price = total_cash_spent / total_shares_filled if total_shares_filled > 0 else 0

    return {
        'shares_filled': total_shares_filled,
        'cash_spent': total_cash_spent,
        'avg_price': avg_price,
        'remaining_shares': remaining_shares,
        'execution_history': execution_history
    }


def twap_strategy(snapshots: List[Dict[str, Any]], order_size: int) -> Dict[str, Any]:
    """
    Implements the 60-second-bucket TWAP baseline strategy.
    
    Divides the order evenly across 60-second time buckets with a single pass through snapshots.
    
    Args:
        snapshots: List of market snapshots
        order_size: Target number of shares to execute
    
    Returns:
        Dictionary with execution results
    """
    if not snapshots:
        return {
            'shares_filled': 0,
            'cash_spent': 0,
            'avg_price': 0,
            'remaining_shares': order_size,
            'execution_history': []
        }
    
    start_time = snapshots[0]['timestamp']
    end_time = snapshots[-1]['timestamp']
    time_range = (end_time - start_time).total_seconds()
    num_buckets = max(1, int(time_range / 60))
    bucket_duration = pd.Timedelta(seconds=60)
    
    # Shares per bucket
    shares_per_bucket = order_size // num_buckets
    extra_shares = order_size % num_buckets
    bucket_shares = [shares_per_bucket + (1 if i < extra_shares else 0) for i in range(num_buckets)]
    
    remaining_shares = order_size
    total_cash_spent = 0
    total_shares_filled = 0
    execution_history = []
    current_bucket_targets = bucket_shares.copy()  # Track remaining shares for each bucket
    
    for snapshot in snapshots:
        bucket_idx = min(num_buckets - 1, int((snapshot['timestamp'] - start_time).total_seconds() / 60))
        
        if current_bucket_targets[bucket_idx] <= 0 or remaining_shares <= 0:
            continue
        
        venues = snapshot['venues']
        if not venues:
            continue
        
        best_venue_idx = min(range(len(venues)), key=lambda i: venues[i].ask)
        best_venue = venues[best_venue_idx]
        
        target_shares = current_bucket_targets[bucket_idx]
        exe = min(target_shares, best_venue.ask_size)
        cash_spent = exe * best_venue.ask
        
        current_bucket_targets[bucket_idx] -= exe
        total_shares_filled += exe
        total_cash_spent += cash_spent
        remaining_shares -= exe
        
        execution_history.append({
            'timestamp': snapshot['timestamp'],
            'bucket': bucket_idx,
            'venue_id': best_venue.id,
            'executed': exe,
            'cash_spent': cash_spent
        })
    
    avg_price = total_cash_spent / total_shares_filled if total_shares_filled > 0 else 0
    
    return {
        'shares_filled': total_shares_filled,
        'cash_spent': total_cash_spent,
        'avg_price': avg_price,
        'remaining_shares': remaining_shares,
        'execution_history': execution_history
    }


def vwap_strategy(snapshots: List[Dict[str, Any]], order_size: int) -> Dict[str, Any]:
    """
    Implements the VWAP baseline strategy that weights orders by displayed ask size.

    Args:
        snapshots: List of market snapshots
        order_size: Target number of shares to execute

    Returns:
        Dictionary with execution results
    """
    remaining_shares = order_size
    total_cash_spent = 0
    total_shares_filled = 0
    execution_history = []

    for snapshot in snapshots:
        if remaining_shares <= 0:
            break

        venues = snapshot['venues']
        if not venues:
            continue

        total_size = sum(venue.ask_size for venue in venues)
        if total_size == 0:
            continue

        # Allocate based on displayed size
        allocations = []
        for venue in venues:
            weight = venue.ask_size / total_size
            shares = int(remaining_shares * weight)
            allocations.append(shares)

        total_allocated = sum(allocations)
        if total_allocated < remaining_shares:
            max_idx = allocations.index(max(allocations))
            allocations[max_idx] += remaining_shares - total_allocated

        # Execute allocations
        snapshot_executed = 0
        snapshot_cash_spent = 0

        for i, alloc in enumerate(allocations):
            venue = venues[i]
            exe = min(alloc, venue.ask_size)
            cash_spent = exe * venue.ask

            snapshot_executed += exe
            snapshot_cash_spent += cash_spent

        total_shares_filled += snapshot_executed
        total_cash_spent += snapshot_cash_spent
        remaining_shares -= snapshot_executed

        execution_history.append({
            'timestamp': snapshot['timestamp'],
            'allocated': allocations,
            'executed': snapshot_executed,
            'cash_spent': snapshot_cash_spent
        })

    # Calculate metrics
    avg_price = total_cash_spent / total_shares_filled if total_shares_filled > 0 else 0

    return {
        'shares_filled': total_shares_filled,
        'cash_spent': total_cash_spent,
        'avg_price': avg_price,
        'remaining_shares': remaining_shares,
        'execution_history': execution_history
    }


def evaluate_parameter_set(args):
    """Evaluate a single parameter set (designed for multiprocessing)"""
    params, snapshots, order_size = args
    lambda_over, lambda_under, theta_queue = params
    
    result = simulate_execution(snapshots, order_size, lambda_over, lambda_under, theta_queue)
    
    if result['shares_filled'] == 0:
        return (params, float('inf'))
    
    return (params, result['avg_price'])


def parameter_search(snapshots, order_size, n_iterations=40):
    """
    Performs an efficient parameter search to find optimal parameters.
    Uses a two-phase adaptive search and parallel processing.

    Args:
        snapshots: List of market snapshots
        order_size: Target number of shares to execute
        n_iterations: Total number of iterations for the optimization

    Returns:
        Dictionary with best parameters and their performance
    """
    
    # Parameter bounds
    param_bounds = {
        'lambda_over': (0.001, 0.2),
        'lambda_under': (0.001, 0.2),
        'theta_queue': (0.0001, 0.05)
    }
    
    n_first_phase = int(n_iterations * 0.6)
    
    # Generate quasi-random points using a stratified approach
    def generate_stratified_samples(n_samples, param_bounds):
        samples = []
        lambda_over_range = np.linspace(param_bounds['lambda_over'][0], 
                                        param_bounds['lambda_over'][1], 
                                        int(np.cbrt(n_samples))+1)
        lambda_under_range = np.linspace(param_bounds['lambda_under'][0], 
                                         param_bounds['lambda_under'][1], 
                                         int(np.cbrt(n_samples))+1)
        theta_queue_range = np.linspace(param_bounds['theta_queue'][0], 
                                        param_bounds['theta_queue'][1], 
                                        int(np.cbrt(n_samples))+1)
        
        # Add some random perturbation to avoid grid pattern
        for i in range(len(lambda_over_range)-1):
            for j in range(len(lambda_under_range)-1):
                for k in range(len(theta_queue_range)-1):
                    if len(samples) >= n_samples:
                        break
        
                    sample = [
                        lambda_over_range[i] + (lambda_over_range[i+1] - lambda_over_range[i]) * np.random.random(),
                        lambda_under_range[j] + (lambda_under_range[j+1] - lambda_under_range[j]) * np.random.random(),
                        theta_queue_range[k] + (theta_queue_range[k+1] - theta_queue_range[k]) * np.random.random()
                    ]
                    samples.append(sample)
        
        # If more samples needed, add pure random ones
        while len(samples) < n_samples:
            samples.append([
                param_bounds['lambda_over'][0] + (param_bounds['lambda_over'][1] - param_bounds['lambda_over'][0]) * np.random.random(),
                param_bounds['lambda_under'][0] + (param_bounds['lambda_under'][1] - param_bounds['lambda_under'][0]) * np.random.random(),
                param_bounds['theta_queue'][0] + (param_bounds['theta_queue'][1] - param_bounds['theta_queue'][0]) * np.random.random()
            ])
            
        return np.array(samples)
    
    # Generate first phase samples
    X_first_phase = generate_stratified_samples(n_first_phase, param_bounds)
    
    # Parallel evaluation
    args_first_phase = [(params, snapshots, order_size) for params in X_first_phase]
    with Pool(processes=min(cpu_count(), n_first_phase)) as pool:
        results_first_phase = pool.map(evaluate_parameter_set, args_first_phase)
    
    # Find best parameters from first phase
    best_idx = np.argmin([result[1] for result in results_first_phase])
    best_params_first = results_first_phase[best_idx][0]
    best_cost_first = results_first_phase[best_idx][1]
    
    # Second phase: Focused search around best point from first phase
    n_second_phase = n_iterations - n_first_phase
    
    # Narrower bounds around best point (25% of original range)
    local_range_factor = 0.25
    local_param_bounds = {
        'lambda_over': (
            max(param_bounds['lambda_over'][0], best_params_first[0] - local_range_factor * (param_bounds['lambda_over'][1] - param_bounds['lambda_over'][0])),
            min(param_bounds['lambda_over'][1], best_params_first[0] + local_range_factor * (param_bounds['lambda_over'][1] - param_bounds['lambda_over'][0]))
        ),
        'lambda_under': (
            max(param_bounds['lambda_under'][0], best_params_first[1] - local_range_factor * (param_bounds['lambda_under'][1] - param_bounds['lambda_under'][0])),
            min(param_bounds['lambda_under'][1], best_params_first[1] + local_range_factor * (param_bounds['lambda_under'][1] - param_bounds['lambda_under'][0]))
        ),
        'theta_queue': (
            max(param_bounds['theta_queue'][0], best_params_first[2] - local_range_factor * (param_bounds['theta_queue'][1] - param_bounds['theta_queue'][0])),
            min(param_bounds['theta_queue'][1], best_params_first[2] + local_range_factor * (param_bounds['theta_queue'][1] - param_bounds['theta_queue'][0]))
        )
    }
    
    # Generate second phase samples
    X_second_phase = np.random.uniform(
        low=[local_param_bounds['lambda_over'][0], local_param_bounds['lambda_under'][0], local_param_bounds['theta_queue'][0]],
        high=[local_param_bounds['lambda_over'][1], local_param_bounds['lambda_under'][1], local_param_bounds['theta_queue'][1]],
        size=(n_second_phase, 3)
    )
    
    # Parallel evaluation
    args_second_phase = [(params, snapshots, order_size) for params in X_second_phase]
    with Pool(processes=min(cpu_count(), n_second_phase)) as pool:
        results_second_phase = pool.map(evaluate_parameter_set, args_second_phase)
    
    # Combine both phases
    all_params = []
    for params, cost in results_first_phase:
        all_params.append({
            'lambda_over': params[0],
            'lambda_under': params[1],
            'theta_queue': params[2],
            'avg_price': cost,
            'phase': 'exploration'
        })
    
    for params, cost in results_second_phase:
        all_params.append({
            'lambda_over': params[0],
            'lambda_under': params[1],
            'theta_queue': params[2],
            'avg_price': cost,
            'phase': 'refinement'
        })
    
    # Find the best parameters across both phases
    all_results = results_first_phase + results_second_phase
    best_idx = np.argmin([result[1] for result in all_results])
    best_params_tuple = all_results[best_idx][0]
    
    best_params = {
        'lambda_over': best_params_tuple[0],
        'lambda_under': best_params_tuple[1],
        'theta_queue': best_params_tuple[2]
    }
    
    # Run simulation with best parameters for full results
    best_result = simulate_execution(
        snapshots, order_size,
        best_params['lambda_over'],
        best_params['lambda_under'],
        best_params['theta_queue']
    )

    return {
        'best_params': best_params,
        'best_result': best_result,
        'all_evaluated_params': all_params,
        'exploration_phase_best': {
            'lambda_over': best_params_first[0],
            'lambda_under': best_params_first[1],
            'theta_queue': best_params_first[2],
            'avg_price': best_cost_first
        }
    }
    

def generate_results(best_params: Dict[str, float], allocator_results: Dict[str, Any],
                     best_ask_results: Dict[str, Any], twap_results: Dict[str, Any],
                     vwap_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates the final results JSON as specified in the task description.
    """

    # Calculate savings in basis points
    def calculate_savings(baseline_price, allocator_price):
        if baseline_price == 0:
            return 0
        return (baseline_price - allocator_price) / baseline_price * 10000  # 10000 bp = 100%

    best_ask_savings = calculate_savings(best_ask_results['avg_price'], allocator_results['avg_price'])
    twap_savings = calculate_savings(twap_results['avg_price'], allocator_results['avg_price'])
    vwap_savings = calculate_savings(vwap_results['avg_price'], allocator_results['avg_price'])

    results = {
        'best_parameters': {
            'lambda_over': best_params['lambda_over'],
            'lambda_under': best_params['lambda_under'],
            'theta_queue': best_params['theta_queue']
        },
        'allocator': {
            'cash_spent': allocator_results['cash_spent'],
            'avg_price': allocator_results['avg_price']
        },
        'baselines': {
            'best_ask': {
                'cash_spent': best_ask_results['cash_spent'],
                'avg_price': best_ask_results['avg_price']
            },
            'twap': {
                'cash_spent': twap_results['cash_spent'],
                'avg_price': twap_results['avg_price']
            },
            'vwap': {
                'cash_spent': vwap_results['cash_spent'],
                'avg_price': vwap_results['avg_price']
            }
        },
        'savings_bp': {
            'vs_best_ask': best_ask_savings,
            'vs_twap': twap_savings,
            'vs_vwap': vwap_savings
        }
    }

    return results


def main():
    
    ORDER_SIZE = 5000
    DATA_FILE = "l1_day.csv"

    # Default fee and rebate values
    FEE = 0.003
    REBATE = 0.002

    # Load and preprocess data
    try:
        df = load_and_preprocess_data(DATA_FILE)
        snapshots = create_snapshots(df, fee=FEE, rebate=REBATE)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    search_results = parameter_search(snapshots, ORDER_SIZE, n_iterations=20)

    best_params = search_results['best_params']
    allocator_results = search_results['best_result']

    print("Best parameters found:")
    print(json.dumps(best_params, indent=2))

    # Run baseline strategies
    best_ask_results = best_ask_strategy(snapshots, ORDER_SIZE)
    twap_results = twap_strategy(snapshots, ORDER_SIZE)
    vwap_results = vwap_strategy(snapshots, ORDER_SIZE)

    # Generate output
    results = generate_results(best_params, allocator_results, best_ask_results, twap_results, vwap_results)

    # Print JSON to stdout
    print("Final results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
