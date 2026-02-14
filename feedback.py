class FeedbackCollector:
    """
    Module to collect use feedback on retrieved results.
    """
    def get_user_feedback(self, results):
        """
        Collects ratings for the retrieved results.
        """
        feedback_data = []
        print("\nPlease rate the relevance of the following results (1-5):")
        for res in results:
            while True:
                try:
                    rating = int(input(f"Result '{res['content'][:50]}...': "))
                    if 1 <= rating <= 5:
                        feedback_data.append({"id": res['id'], "rating": rating})
                        break
                    else:
                        print("Please enter a number between 1 and 5.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        return feedback_data
