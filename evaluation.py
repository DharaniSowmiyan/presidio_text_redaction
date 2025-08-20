import re
import os
from collections import defaultdict

def parse_ground_truth(file_path):
    """
    Parses a file with XML-like tags to extract ground truth entities.
    Example: "My name is <PERSON>John Doe</PERSON>."
    Returns a list of tuples: (entity_type, text)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Regex to find all tagged entities, e.g., <PERSON>...</PERSON>
        pattern = re.compile(r"<([A-Z_]+)>(.*?)</\1>")
        matches = pattern.findall(text)
        return matches
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at '{file_path}'")
        return None

def parse_predictions(file_path):
    """
    Parses a redacted file to extract predicted entities.
    Example: "My name is <PERSON>."
    Returns a list of entity types found.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Regex to find all replacement tags, e.g., <PERSON>
        pattern = re.compile(r"<([A-Z_]+)>")
        matches = pattern.findall(text)
        return matches
    except FileNotFoundError:
        print(f"Error: Prediction file not found at '{file_path}'")
        return None

def evaluate_performance(ground_truth, predictions):
    """
    Compares ground truth entities with predictions to calculate performance metrics.
    """
    gt_counts = defaultdict(int)
    pred_counts = defaultdict(int)
    true_positives = defaultdict(int)

    # Count occurrences in ground truth
    for entity_type, _ in ground_truth:
        gt_counts[entity_type] += 1

    # Count occurrences in predictions
    for entity_type in predictions:
        pred_counts[entity_type] += 1
    
    # Create mutable copies for matching
    gt_copy = list(ground_truth)
    pred_copy = list(predictions)

    # Simple matching based on type. For a more advanced evaluation,
    # you would also check character offsets.
    for pred_entity in pred_copy:
        # Find a matching ground truth entity of the same type
        match_found = False
        for i, (gt_entity_type, _) in enumerate(gt_copy):
            if pred_entity == gt_entity_type:
                true_positives[pred_entity] += 1
                gt_copy.pop(i) # Remove matched item to prevent re-matching
                match_found = True
                break
    
    # --- Calculate Metrics ---
    results = {}
    all_entity_types = set(gt_counts.keys()) | set(pred_counts.keys())

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for entity_type in sorted(list(all_entity_types)):
        tp = true_positives[entity_type]
        fp = pred_counts[entity_type] - tp
        fn = gt_counts[entity_type] - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[entity_type] = {
            "True Positives": tp,
            "False Positives": fp,
            "False Negatives": fn,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_score
        }
    
    # Calculate overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    results["OVERALL"] = {
        "True Positives": total_tp,
        "False Positives": total_fp,
        "False Negatives": total_fn,
        "Precision": overall_precision,
        "Recall": overall_recall,
        "F1-Score": overall_f1
    }

    return results

def print_results(results):
    """Prints the evaluation results in a formatted table."""
    print("\n" + "="*80)
    print("PII Redaction Performance Evaluation")
    print("="*80)
    
    header = "| {:<20} | {:<5} | {:<5} | {:<5} | {:<10} | {:<10} | {:<10} |".format(
        "Entity Type", "TP", "FP", "FN", "Precision", "Recall", "F1-Score"
    )
    print(header)
    print("|" + "-"*78 + "|")

    for entity_type, metrics in results.items():
        if entity_type == "OVERALL":
            print("|" + "-"*78 + "|")
        
        row = "| {:<20} | {:<5} | {:<5} | {:<5} | {:<10.3f} | {:<10.3f} | {:<10.3f} |".format(
            entity_type,
            metrics["True Positives"],
            metrics["False Positives"],
            metrics["False Negatives"],
            metrics["Precision"],
            metrics["Recall"],
            metrics["F1-Score"]
        )
        print(row)
    
    print("="*80)


if __name__ == "__main__":
    # Get file paths from user
    gt_path = input("Please enter the path to the ground truth file: ")
    pred_path = input("Please enter the path to the prediction (output) file: ")

    # Check that files exist
    if not os.path.exists(gt_path):
        print(f"\nError: Ground truth file not found at '{gt_path}'")
    elif not os.path.exists(pred_path):
        print(f"\nError: Prediction file not found at '{pred_path}'")
    else:
        # Run the evaluation
        ground_truth_data = parse_ground_truth(gt_path)
        prediction_data = parse_predictions(pred_path)
        
        if ground_truth_data is not None and prediction_data is not None:
            evaluation_results = evaluate_performance(ground_truth_data, prediction_data)
            print_results(evaluation_results)
