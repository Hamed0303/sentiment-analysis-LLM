"""Generates a PDF knowledge base for sentiment classification.

The PDF is structured so a fine-tuned model can retrieve focused guidance
when its own confidence is low. Heavy emphasis on disambiguating the neutral
class (the empirically weakest class in our experiments).
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

OUT = Path(__file__).parent / "sentiment_kb.pdf"


SECTIONS = [
    ("NEGATIVE SENTIMENT — Core Definition",
     "A tweet is NEGATIVE when the author expresses dissatisfaction, frustration, sadness, anger, "
     "complaint, disappointment, or any unpleasant emotional reaction. The author has a negative "
     "stance toward the topic, person, event, product, or situation being discussed."),

    ("NEGATIVE SENTIMENT — Linguistic Markers",
     "Explicit negative words: hate, terrible, awful, horrible, worst, sucks, disgusting, gross, "
     "annoying, frustrating, disappointing, useless, bad, sad, mad, angry, upset, broken, ruined. "
     "Complaint phrases: 'I can't stand', 'why does this always', 'fed up with', 'so done with', "
     "'this is the worst', 'never again', 'such a waste'. "
     "Negation flipping a positive: 'not good', 'no joy', 'didn't enjoy', 'not impressed', 'not worth it'. "
     "Emotional venting: 'ugh', 'wtf', 'smh', sighs, frustrated punctuation patterns."),

    ("NEGATIVE SENTIMENT — Tone Cues",
     "Even without explicit negative words, a tweet is negative if it expresses: "
     "regret ('shouldn't have', 'wish I hadn't'), loss ('miss', 'gone', 'lost'), "
     "physical/emotional pain ('hurts', 'sore', 'tired', 'exhausted', 'crying'), "
     "rejection ('left me', 'ignored', 'unfollowed'), or unfortunate events "
     "('got fired', 'car broke down', 'missed the flight'). "
     "Sarcasm signaling negativity often uses positive words ironically — "
     "'oh GREAT another Monday', 'just what I needed'."),

    ("POSITIVE SENTIMENT — Core Definition",
     "A tweet is POSITIVE when the author expresses happiness, excitement, gratitude, "
     "satisfaction, love, joy, enthusiasm, approval, or any pleasant emotional reaction. "
     "The author has a positive stance toward the subject — endorsement, praise, or celebration."),

    ("POSITIVE SENTIMENT — Linguistic Markers",
     "Explicit positive words: love, amazing, awesome, great, fantastic, wonderful, perfect, "
     "best, brilliant, beautiful, happy, glad, excited, thrilled, blessed, grateful, thankful, "
     "proud, delighted, fabulous, incredible, stunning. "
     "Enthusiastic phrases: 'so happy', 'can't wait', 'best day ever', 'made my day', "
     "'absolutely loved', 'highly recommend', 'so proud of'. "
     "Celebration markers: congratulations, congrats, yay, woohoo, finally, achieved, "
     "accomplished, won, succeeded. "
     "Affection: heart emojis, '<3', xoxo, hugs, kisses."),

    ("POSITIVE SENTIMENT — Subtle Cases",
     "Tweets can be positive without strong adjectives if they describe pleasant experiences: "
     "'morning coffee hits different today', 'just got the new album', 'finished my project', "
     "'kids made me breakfast'. The presence of a fortunate event or accomplishment is usually positive. "
     "Anticipation of something good: 'tomorrow's the day', 'flying out tonight', 'concert in 3 days'. "
     "Note: 'finally got my food' is positive (relief/satisfaction), while 'finally got my food after 2 hours' "
     "is negative (frustration)."),

    ("NEUTRAL SENTIMENT — Core Definition",
     "A tweet is NEUTRAL when the author makes a statement WITHOUT clear emotional valence. "
     "It is factual, informational, descriptive, or simply reports an event without expressing approval "
     "or disapproval. Neutral tweets do NOT lean toward either positive or negative."),

    ("NEUTRAL SENTIMENT — Indicator Patterns",
     "Factual statements: 'meeting at 3pm', 'new episode airs Thursday', 'temperature is 72F'. "
     "Questions seeking information: 'anyone know how to fix this?', 'when does the store open?', "
     "'has anyone tried the new update?'. "
     "Announcements without emotional framing: 'just updated my profile', 'posted a new video', "
     "'starting a new course'. "
     "Observations/descriptions: 'walking to work', 'reading a book', 'rain in NYC today'. "
     "News-style content: 'company X announced Y', 'study finds Z'. "
     "Mild routine reports: 'lunch was salad', 'spent the day at home', 'watching TV'."),

    ("NEUTRAL vs POSITIVE — Critical Disambiguation",
     "The most common error is classifying mild-positive tweets as neutral, or vice versa. "
     "Rule: if the tweet describes an event AND contains any positive emotional word (love, great, "
     "happy, excited, thanks), classify as POSITIVE even if mild. "
     "'Had coffee this morning' → NEUTRAL (pure description). "
     "'Had GREAT coffee this morning' → POSITIVE (positive adjective present). "
     "'Going to the gym' → NEUTRAL. "
     "'Excited to hit the gym today' → POSITIVE (emotion word present). "
     "Exclamation marks alone do NOT make a tweet positive — they only amplify whatever sentiment is there. "
     "'Meeting at 3!' is still neutral. 'Best meeting ever!' is positive."),

    ("NEUTRAL vs NEGATIVE — Critical Disambiguation",
     "Similarly, mild complaints often look neutral. "
     "Rule: if the tweet reports a fact AND includes any negative valuation, mild words, or implicit complaint, "
     "classify as NEGATIVE. "
     "'Stuck in traffic' → NEGATIVE (implicit complaint, 'stuck' is negative). "
     "'Driving on the highway' → NEUTRAL (descriptive). "
     "'Working late again' → NEGATIVE ('again' implies frustration). "
     "'Working late tonight' → NEUTRAL (no frustration marker). "
     "Questions can be negative if framed as complaints: "
     "'why is this app so slow?' → NEGATIVE. "
     "'how does this app work?' → NEUTRAL."),

    ("MIXED & AMBIGUOUS CASES — Decision Rules",
     "When a tweet contains both positive and negative elements, use the DOMINANT sentiment. "
     "'The food was great but the service was terrible' — which is stronger? "
     "If equally balanced, lean to negative (negative typically dominates perception). "
     "Sarcasm: positive words used ironically → flip to negative. "
     "'Oh perfect, my flight is delayed again' → NEGATIVE (sarcastic 'perfect'). "
     "Excitement about negative content can still be NEGATIVE: "
     "'omg I CANNOT believe how bad this movie is' → NEGATIVE. "
     "Politeness markers ('please', 'thanks') don't automatically make a tweet positive. "
     "Context: 'thanks for ruining my day' → NEGATIVE."),

    ("EDGE CASES — Specific Patterns",
     "1. Pure announcements without emotion: 'New blog post is live' → NEUTRAL. "
     "2. Self-deprecating humor: 'me trying to adult lol' → usually NEGATIVE (mild self-criticism). "
     "3. Comparison statements: 'this is better than X' → POSITIVE. 'this is worse than X' → NEGATIVE. "
     "4. Wish/hope statements: 'I wish it would stop raining' → NEGATIVE. 'I hope it gets sunny' → POSITIVE. "
     "5. Surprise can go either way: 'OMG this is amazing!' → POSITIVE. 'OMG this is awful!' → NEGATIVE. "
     "Surprise alone is NEUTRAL: 'just learned that X is true'. "
     "6. Achievement without emotion words: 'finished the marathon' → POSITIVE (achievement = positive). "
     "7. Routine without emotion: 'finished my coffee' → NEUTRAL (mundane fact). "
     "8. Tweets ending with positive emojis (😊, ❤️, 😍) → POSITIVE even if text is neutral. "
     "9. Tweets ending with negative emojis (😢, 😡, 😞) → NEGATIVE."),

    ("FINAL DECISION CHECKLIST",
     "When classifying a tweet, ask in order: "
     "(1) Is there any explicit positive OR negative emotion word? If yes, that determines the class. "
     "(2) If no explicit emotion words: does the tweet describe a fortunate event (POSITIVE) or "
     "unfortunate event (NEGATIVE)? "
     "(3) If neither: is it purely factual, informational, or a neutral question? Then NEUTRAL. "
     "(4) If sarcastic: flip the apparent sentiment. "
     "(5) When in genuine doubt between neutral and another class, prefer the non-neutral class IF "
     "any emotional cue is present, however mild."),
]


def main():
    doc = SimpleDocTemplate(str(OUT), pagesize=letter, topMargin=0.6*inch, bottomMargin=0.6*inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Heading1"], fontSize=16, spaceAfter=12)
    section_style = ParagraphStyle("section", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, spaceAfter=8, leading=14)

    story = []
    story.append(Paragraph("Sentiment Classification Knowledge Base", title_style))
    story.append(Paragraph(
        "Reference for retrieval-augmented sentiment classification. Used when the primary "
        "fine-tuned classifier reports low confidence (&lt; 70%).", body_style))
    story.append(Spacer(1, 0.1*inch))

    for heading, body in SECTIONS:
        story.append(Paragraph(heading, section_style))
        story.append(Paragraph(body, body_style))

    doc.build(story)
    print(f"Saved knowledge base: {OUT}")


if __name__ == "__main__":
    main()
