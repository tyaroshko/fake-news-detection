"""Main execution script for fake news detection"""

import argparse
import warnings

warnings.filterwarnings("ignore")

from config import Config
from data_loader import FakeNewsDataLoader
from evaluator import FakeNewsEvaluator
from model import FakeNewsModel
from predictor import FakeNewsPredictor
from utils import ensure_dir, get_device, save_json, set_seed
from visualizer import FakeNewsVisualizer


def train_model(config: Config):
    """
    Train the fake news detection model

    Args:
        config: Configuration object
    """
    # Set seed
    set_seed(config.training.seed)

    # Get device
    device = get_device()

    # Initialize components
    print("" + "=" * 60)
    print("INITIALIZING COMPONENTS")
    print("=" * 60)

    data_loader = FakeNewsDataLoader(config.data)
    model = FakeNewsModel(config.model, config.training)
    evaluator = FakeNewsEvaluator()
    visualizer = FakeNewsVisualizer()

    # Load and prepare data
    print("" + "=" * 60)
    print("LOADING AND PREPARING DATA")
    print("=" * 60)

    data_loader.load_data()
    datasets = data_loader.create_datasets()

    # Visualize data distribution
    visualizer.plot_label_distribution(datasets)

    # Initialize model and tokenizer
    model.initialize_model_and_tokenizer()

    # Tokenize datasets
    tokenized_datasets = data_loader.tokenize_datasets(
        model.tokenizer, config.model.max_length
    )

    # Train model
    print("" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)

    train_result = model.train(
        tokenized_datasets["train"], tokenized_datasets["validation"]
    )

    # Plot training history
    visualizer.plot_training_history(model.trainer)

    # Evaluate on test set
    print("" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)

    test_results = model.evaluate(tokenized_datasets["test"])
    print("Test Set Metrics:")
    for key, value in test_results.items():
        if key.startswith("eval_"):
            metric_name = key.replace("eval_", "")
            print(f"  {metric_name}: {value:.4f}")

    # Get detailed predictions
    predictions = model.predict(tokenized_datasets["test"])
    y_pred, y_true, y_probs = evaluator.get_predictions_from_trainer(predictions)

    # Evaluate predictions
    detailed_results = evaluator.evaluate_predictions(y_true, y_pred, y_probs)
    evaluator.print_summary(detailed_results)

    # Generate visualizations
    print("" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    visualizer.plot_all_metrics(detailed_results)

    # Save results
    ensure_dir(config.training.output_dir)
    evaluator.save_results(f"{config.training.output_dir}/evaluation_results.json")
    save_json(
        train_result.metrics, f"{config.training.output_dir}/training_metrics.json"
    )

    # Save final model
    final_model_path = "./final_fake_news_detector"
    model.save_model(final_model_path)

    print("" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {final_model_path}")
    print(f"Results saved to: {config.training.output_dir}")

    return model, evaluator, visualizer


def test_predictions(model_path: str = "./final_fake_news_detector"):
    """
    Test the model with sample predictions

    Args:
        model_path: Path to saved model
    """
    print("" + "=" * 60)
    print("TESTING PREDICTIONS")
    print("=" * 60)

    predictor = FakeNewsPredictor(model_path)

    # Sample texts
    sample_texts = [
        "Scientists discover that chocolate is actually a vegetable and eating it daily will make you live to 200 years old!",
        "The Federal Reserve announced a quarter-point interest rate increase following the latest inflation data.",
        "BREAKING: Aliens have officially made contact with Earth and are demanding all our pizza!",
        "Local school district implements new STEM program to enhance student learning in science and technology.",
        "Study shows that people who sleep with their phones under their pillow develop telepathic abilities!",
        "The unemployment rate fell to 3.5% last month, according to the Bureau of Labor Statistics.",
    ]

    print("Single Predictions:")
    print("-" * 40)
    for text in sample_texts[:3]:
        result = predictor.predict_single(text)
        print(f"Text: {result['text']}")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Fake Probability: {result['fake_probability']:.2%}")
        print(f"Real Probability: {result['real_probability']:.2%}")

    print("Batch Predictions:")
    print("-" * 40)
    results_df = predictor.predict_batch(sample_texts)
    print(results_df.to_string())


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Fake News Detection System")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "predict"],
        default="train",
        help="Mode to run the system in",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="WELFake_Dataset.csv",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./final_fake_news_detector",
        help="Path to saved model (for test/predict modes)",
    )
    parser.add_argument("--text", type=str, help="Text to predict (for predict mode)")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run mode: use only 10%% of the dataset for faster training",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=None,
        help="Fraction of dataset to use (0-1). Overrides --quick flag. Example: 0.1 for 10%%",
    )

    args = parser.parse_args()

    if args.mode == "train":
        # Create configuration
        config = Config()
        config.data.data_path = args.data_path
        
        # Handle quick run / sample ratio
        if args.sample_ratio is not None:
            config.data.sample_ratio = args.sample_ratio
        elif args.quick:
            config.data.sample_ratio = 0.1

        # Train model
        model, evaluator, visualizer = train_model(config)

        # Test predictions
        test_predictions(config.training.output_dir)

    elif args.mode == "test":
        # Test predictions with saved model
        test_predictions(args.model_path)

    elif args.mode == "predict":
        if not args.text:
            print("Please provide text to predict using --text argument")
            return

        # Make prediction
        predictor = FakeNewsPredictor(args.model_path)
        result = predictor.predict_single(args.text)

        print("" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"Text: {result['text']}")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Fake Probability: {result['fake_probability']:.2%}")
        print(f"Real Probability: {result['real_probability']:.2%}")


if __name__ == "__main__":
    main()
