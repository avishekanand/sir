import argparse
from retriever import SIRRetriever
from feedback import FeedbackCollector
from improver import SIRImprover

def main():
    parser = argparse.ArgumentParser(description="SIR: Self-Improving Retrieval")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--feedback", action="store_true", help="Collect feedback for results")
    args = parser.parse_args()

    retriever = SIRRetriever()
    improver = SIRImprover()
    collector = FeedbackCollector()

    if args.query:
        print(f"\n--- Searching for: '{args.query}' ---")
        results = retriever.search(args.query)
        for i, res in enumerate(results):
            print(f"{i+1}. {res['content']} (Score: {res['score']:.4f})")

        if args.feedback:
            feedback_data = collector.get_user_feedback(results)
            improver.optimize(args.query, feedback_data)
            print("\nSIR has learned from your feedback.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
