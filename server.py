import flwr as fl
from typing import List, Tuple, Dict

# Define an aggregation function for evaluation metrics


def aggregate_accuracy(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregate accuracy across clients by computing a weighted average."""
    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0:
        return {"accuracy": 0.0}
    accuracy_sum = sum(num_samples * m["accuracy"]
                       for num_samples, m in metrics)
    aggregated_accuracy = accuracy_sum / total_samples
    return {"accuracy": aggregated_accuracy}


# Start the federated server with FedAvg strategy
if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=3,
        # Use the custom accuracy aggregation
        evaluate_metrics_aggregation_fn=aggregate_accuracy  # type: ignore
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )
