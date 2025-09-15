from transformers import pipeline

# Create the summarization pipeline
# We will use the 'facebook/bart-large-cnn' model, which is a popular choice for this task.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define the text you want to summarize
# You can replace this with any text you choose, such as a news article.
text = """
In a significant move for space exploration, NASA has announced a new mission to Mars. The Perseverance rover, which has been exploring the red planet since February 2021, has discovered evidence of a large, ancient lake bed. This finding has excited scientists, as it suggests Mars may have once been capable of supporting life. The new mission aims to bring back rock and soil samples from this region for further analysis on Earth. The samples could provide definitive evidence of past life and revolutionize our understanding of planetary evolution.
"""

# Generate the summary with specified length constraints
summary = summarizer(text, max_length=50, min_length=20, do_sample=False)

# Print the result
print("Original Text:")
print(text)
print("\n---")
print("Summary:")
print(summary[0]['summary_text'])