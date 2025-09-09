import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

# Download necessary NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


def extract_chat_bullets(text, num_bullets=5, is_chat=True):
    """
    Extract key points from text as bullet points, with special handling for chat transcripts.

    Args:
        text (str): The input text to summarize
        num_bullets (int): The number of bullet points to extract
        is_chat (bool): Whether the input is a chat transcript

    Returns:
        str: Bullet-pointed summary
    """
    # For chat transcripts, process by turns rather than sentences
    if is_chat:
        return summarize_chat(text, num_bullets)
    else:
        return summarize_document(text, num_bullets)


def summarize_document(text, num_bullets=5):
    """Standard document summarization using TF-IDF with enhancements"""
    sentences = sent_tokenize(text)

    # Ensure we have enough sentences to extract from
    if len(sentences) <= num_bullets:
        return "\n".join(f"- {sent}" for sent in sentences)

    # Clean sentences (remove extra whitespace)
    sentences = [re.sub(r'\s+', ' ', sent).strip() for sent in sentences]

    # Apply TF-IDF with bigrams to capture more context
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),  # Use unigrams and bigrams
        max_features=5000
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    except ValueError:  # Handle case with too few sentences
        return "\n".join(f"- {sent}" for sent in sentences)

    # Score sentences based on multiple factors
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        score = sentence_scores[i]

        # Position bias (first and last sentences often important)
        if i < len(sentences) * 0.2 or i > len(sentences) * 0.8:
            score *= 1.2

        # Length penalty (avoid very short or very long sentences)
        words = len(word_tokenize(sentence))
        if 5 <= words <= 25:
            score *= 1.2
        elif words < 3 or words > 40:
            score *= 0.7

        # Content indicators
        if any(keyword in sentence.lower() for keyword in [
            "important", "significant", "key", "main", "critical",
            "essential", "crucial", "fundamental", "vital"
        ]):
            score *= 1.3

        # Questions and instructions (as in original)
        if "?" in sentence or any(keyword in sentence.lower() for keyword in [
            "try", "use", "place", "build", "craft", "should", "must", "need to", "important to"
        ]):
            score *= 1.4

        scored_sentences.append((sentence, score, i))

    # Select top bullets and keep them in original order
    top_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:num_bullets]
    top_sentences = sorted(top_sentences, key=lambda x: x[2])

    return "\n".join(f"- {sentence}" for sentence, _, _ in top_sentences)


def summarize_chat(text, num_bullets=5):
    """Specialized function for summarizing chat transcripts"""
    # Split into turns/messages
    turns = re.split(r'\n(?=\w+:)', text)

    # Clean turns
    turns = [turn.strip() for turn in turns if turn.strip()]

    # Extract speaker and content
    processed_turns = []
    for turn in turns:
        match = re.match(r'^([^:]+):\s*(.*)', turn, re.DOTALL)
        if match:
            speaker, content = match.groups()
            processed_turns.append((speaker, content.strip()))

    # If we couldn't process properly, fall back to document summarization
    if not processed_turns:
        return summarize_document(text, num_bullets)

    # Use TF-IDF on the content of each turn
    contents = [content for _, content in processed_turns]

    # Apply TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(contents)
        turn_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    except ValueError:  # Handle case with too few turns
        # Just take the first num_bullets turns
        selected_turns = processed_turns[:num_bullets]
        return "\n".join(f"- {speaker}: {content}" for speaker, content in selected_turns)

    # Score turns based on multiple factors
    scored_turns = []
    for i, (speaker, content) in enumerate(processed_turns):
        base_score = turn_scores[i]
        score = base_score

        # Favor turns with:
        # 1. Questions from the main participant (assuming first speaker is main)
        if "?" in content and speaker == processed_turns[0][0]:
            score *= 1.4

        # 2. Answers (turns after questions)
        if i > 0 and "?" in processed_turns[i - 1][1]:
            score *= 1.3

        # 3. Turns with instructions or key content
        if any(keyword in content.lower() for keyword in [
            "try", "use", "place", "build", "craft", "should", "must", "need to",
            "important", "remember", "don't forget", "key", "best", "better"
        ]):
            score *= 1.4

        # 4. Avoid very short responses
        if len(word_tokenize(content)) > 5:
            score *= 1.2

        scored_turns.append((speaker, content, score, i))

    # Select top turns
    top_turns = sorted(scored_turns, key=lambda x: x[2], reverse=True)[:num_bullets]

    # Convert to topic-based bullets rather than showing full dialogue
    info_bullets = []
    for speaker, content, _, _ in top_turns:
        # Extract the key information rather than just repeating the turn
        sentences = sent_tokenize(content)
        if len(sentences) > 1:
            # For multi-sentence turns, pick the most informative sentence
            best_sentence = max(sentences, key=lambda s: len(s) if any(
                keyword in s.lower() for keyword in [
                    "try", "use", "place", "build", "craft", "should", "must",
                    "need to", "important", "best"
                ]) else 0
                                )
            info_bullets.append(f"- {best_sentence}")
        else:
            # Format as a clean bullet rather than showing the speaker
            # But keep the speaker for questions
            if "?" in content:
                info_bullets.append(f"- {speaker} asked about {content}")
            else:
                # Extract just the useful info
                info_bullets.append(f"- {content}")

    return "\n".join(info_bullets)


# Example usage
if __name__ == "__main__":
    # Sample transcript from your example
    text = """Alex: Hey guys, I'm trying to optimize my wheat farm. Any tips?
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

    print("CHAT SUMMARIZATION:")
    print(extract_chat_bullets(text, num_bullets=6))

    print("\nSTANDARD DOCUMENT SUMMARIZATION:")
    print(extract_chat_bullets(text, num_bullets=6, is_chat=False))