from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import torch

# Download necessary NLTK data for better sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Check if GPU is available and set device accordingly
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Initialize summarization pipeline with better model and parameters
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",  # Better model for summarization
    device=device
)


# Function to split text into meaningful chunks by sentences
def chunk_text(text, max_chunk_size=1000):
    # Use NLTK for better sentence tokenization
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding the next sentence doesn't exceed max size, add it
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        # Otherwise start a new chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# Function to summarize conversation with improved parameters
def summarize_conversation(text, min_length=30, max_length=150):
    # Split conversation into chunks
    chunks = chunk_text(text, max_chunk_size=1000)

    # Summarize each chunk with appropriate parameters
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i + 1}/{len(chunks)}...")

        # Adjust summary length based on chunk size
        chunk_min_length = min(min_length, max(20, len(chunk) // 15))
        chunk_max_length = min(max_length, max(50, len(chunk) // 5))

        # Generate summary with more appropriate parameters
        summary = summarizer(
            chunk,
            max_length=chunk_max_length,
            min_length=chunk_min_length,
            do_sample=True,
            temperature=0.7,  # Add some variation
            top_p=0.9,  # Filter unlikely tokens
            num_beams=4  # Use beam search for better quality
        )[0]['summary_text']

        chunk_summaries.append(summary)

    return chunk_summaries


# Process the conversation
convo = """Alex: Hey guys, I'm trying to optimize my wheat farm. Any tips?
Blake: Are you using water properly? Each water block hydrates a 4-block radius.
Jordan: Also, make sure you're using a hoe on dirt before planting seeds. It speeds up growth.
Alex: Yeah, I did that. But my crops still take a long time to grow.
Blake: Try placing torches around the farm. Light helps crops grow even at night.
Jordan: Bone meal is a quick way to speed up growth. Just right-click on the crops with it.
Alex: Ah, where do I get bone meal?
Blake: You can craft it from bones. Just kill skeletons or find bones in chests.
Jordan: Or you can use a composter to turn plant materials into bone meal.
Alex: Cool, I'll try that. What about automated farms?
Blake: Water streams can push harvested crops into a hopper. It's useful for large farms.
Jordan: If you want full automation, use observers and pistons to detect and break mature crops.
Alex: Sounds complicated. Do you guys have a simple design?
Blake: You can start with a villager-based farm. Farmers harvest crops and replant them.
Jordan: Just trap a farmer in a 9x9 farm with a composter. He'll keep farming indefinitely.
Alex: Nice! Does this work for potatoes and carrots too?
Blake: Yep! The same concept applies. Just make sure to remove the village's main composter.
Jordan: Sugar cane farms are different, though. You need observers to detect when they grow.
Alex: Oh, how do I automate sugar cane farming?
Blake: Place observers on the second block of the sugar cane. When it grows, pistons break it.
Jordan: The broken cane falls into water streams that lead to a hopper and a chest.
Alex: That's genius! What about mob farms? I need XP and loot.
Blake: The easiest way is a dark room spawner. Mobs spawn in the dark and fall into a killing area.
Jordan: Or you can build a mob tower. Mobs fall from a height and take fall damage.
Alex: What's the best XP farm?
Blake: An Enderman farm in the End. They drop a ton of XP and pearls.
Jordan: If you don't have access to the End, a zombie spawner farm is a good alternative.
Alex: Thanks, guys! I'm going to work on my farms now.
Blake: No problem! Let us know if you need help.
Jordan: Yeah, happy farming!"""

# Get summaries
summaries = summarize_conversation(convo)


# Function to categorize and organize summary points
def organize_summary(summaries):
    print("\n=== ORGANIZED MINECRAFT FARMING GUIDE ===\n")

    # Combine all summaries
    full_summary = " ".join(summaries)

    # Extract topics from the conversation
    topics = {
        "Crop Basics": ["water", "hydrates", "hoe", "light", "torches", "growth"],
        "Bone Meal": ["bone meal", "bones", "skeletons", "composter"],
        "Wheat Farm": ["wheat", "crops", "dirt"],
        "Automated Farms": ["automated", "automation", "water streams", "hopper", "pistons", "observers"],
        "Villager Farms": ["villager", "farmer", "9x9", "indefinitely"],
        "Sugar Cane": ["sugar cane", "observers", "pistons", "water streams"],
        "Mob Farms": ["mob", "XP", "dark room", "spawner", "fall damage", "Enderman", "zombie"]
    }

    # Create organized sections
    organized_content = {}

    # Process each sentence and categorize it
    sentences = sent_tokenize(full_summary)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        added = False
        for topic, keywords in topics.items():
            if any(keyword.lower() in sentence.lower() for keyword in keywords):
                if topic not in organized_content:
                    organized_content[topic] = []
                if sentence not in organized_content[topic]:  # Avoid duplicates
                    organized_content[topic].append(sentence)
                added = True

        # If not categorized, add to general
        if not added:
            if "General Tips" not in organized_content:
                organized_content["General Tips"] = []
            if sentence not in organized_content["General Tips"]:
                organized_content["General Tips"].append(sentence)

    # Print organized summary
    for topic, sentences in organized_content.items():
        print(f"\n## {topic}")
        for sentence in sentences:
            print(f"- {sentence}")

    return organized_content


# Organize and print summary
organized_summary = organize_summary(summaries)