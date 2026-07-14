INSTRUCTION = """
To help you make more accurate and consistent judgments, here is an expanded explanation of how to interpret and assign support percentages. These examples are designed to cover a range of real-world cases you may encounter in the annotation task.
100% - /Full and unambiguous support:
The video clearly shows the exact event described in the claim. There is no need for guessing or interpretation. 
80-100% - Almost complete support:
The main content in the claim is shown, but there may be minor ambiguity in location, identity, or completeness. The overall claims are supported by the video.
60-80% - Strong partial support:
The video strongly suggests the claim is true, but some critical details may be missing, obscured, or ambiguous, limiting the ability to confirm the claim with certainty. The video gives strong but not definitive support.
40-60% - Moderate partial support:
There is some alignment with the claim, but large portions are either missing, unclear, or open to interpretation. While the footage may point in the same general direction as the claim, it lacks the clarity or completeness needed for confident verification.
20-40% - Minimal weak support:
There are small visual or audio cues that could hint at the claim, but they are insufficient to be confident. 
0-20% - Very weak or speculative support:
There may be the slightest indirect reference, such as a related object or setting, but nothing concrete happens. 
0% - No support or contradiction:
The video does not relate to the claim at all, or it directly shows something opposite. 
"""

PROMPT = """
Based on the provided video and text, evaluate the probability that the text is true.
Your answer must be a decimal number between 0 and 1, and you must strictly follow the format below:
<answer>probability_value</answer>
Where probability_value is the result you calculate.
The text to evaluate is:

{text}
"""

SYSTEM_PROMPT = """
You will be provided with a video and a text statement.  
Your task is to evaluate how likely it is that the statement is true based on the visual and audio evidence in the video.  

Your answer must be a single decimal number between 0 and 1, formatted exactly as follows:
<answer>probability_value</answer>

Where `probability_value` represents the probability.
"""





INSTRUCTION_AUDIO = """
To help you make more accurate and consistent judgments, here is an expanded explanation of how to interpret and assign support percentages for audio-based evidence. These examples cover a range of real-world cases you may encounter in this annotation task.

100% - /Full and unambiguous support:  
The audio clearly and directly contains the exact event, statement, or content described in the claim. There is no need for guessing or interpretation — the claim is fully verified by the audio.

80-100% - Almost complete support:  
The main content of the claim is clearly supported by the audio, though there may be minor ambiguity in speaker identity, context, or completeness (e.g., partial recording, mild noise). Overall, the claim is strongly supported.

60-80% - Strong partial support:  
The audio strongly suggests that the claim is true, but some key details may be missing, unclear, or ambiguous — such as incomplete phrases, background noise, or partial conversations. The evidence is strong but not definitive.

40-60% - Moderate partial support:  
There is some alignment between the audio and the claim, but large portions are missing, unclear, or open to interpretation. While the recording points in the same general direction as the claim, it lacks clarity or completeness for confident verification.

20-40% - Minimal weak support:  
There are small verbal cues or contextual hints that could relate to the claim, but they are insufficient to provide confidence in its truth.

0-20% - Very weak or speculative support:  
There may be the slightest indirect reference (such as a related topic or similar voice), but nothing concrete that verifies the claim.

0% - No support or contradiction:  
The audio does not relate to the claim at all, or it directly contradicts it.
"""

PROMPT_AUDIO = """
Based on the provided audio and text, evaluate the probability that the text statement is true.

Your answer must be a decimal number between 0 and 1, and you must strictly follow the format below:
<answer>probability_value</answer>

Where probability_value is the result you calculate.

The text to evaluate is:

{text}
"""


SYSTEM_PROMPT_AUDIO = """
You will be provided with an audio recording and a text statement.  
Your task is to evaluate how likely it is that the statement is true based on the spoken content, tone, and context within the audio.

Your answer must be a single decimal number between 0 and 1, formatted exactly as follows:
<answer>probability_value</answer>

Where `probability_value` represents the probability.
"""

INSTRUCTION_TEXT = """
To help you make more accurate and consistent judgments, here is an expanded explanation of how to interpret and assign support percentages based on textual evidence. These examples are designed to cover a range of logic and linguistic relationships you may encounter.

100% - Full and unambiguous support (Entailment):
The sentence explicitly states the information in the claim, or the claim is a direct paraphrase of the sentence. There is no need for guessing; the facts are identical.

80-100% - Almost complete support:
The main assertions in the claim are present in the sentence. There may be minor differences in wording, synonyms, or omission of non-essential details, but the core meaning is fully preserved and supported.

60-80% - Strong partial support (Strong Implication):
The sentence strongly implies the claim is true through logical inference or context, though it may not state it explicitly. A reasonable person would conclude the claim is likely true based on the sentence.

40-60% - Moderate partial support:
There is a topical alignment or shared keywords. The sentence discusses the same subject matter, but the specific assertion in the claim is neither confirmed nor denied. It is plausible but lacks definitive evidence in the text.

20-40% - Minimal weak support:
There are weak textual links, such as matching entity names or a general theme, but the specific context is different. The sentence provides very little basis to deduce the claim.

0-20% - Very weak or speculative support:
There may be a very distant connection (e.g., related vocabulary), but inferring the claim from the sentence would be highly speculative.

0% - No support or contradiction:
The sentence is completely unrelated to the claim, or it directly contradicts the claim (proves it false).
"""

PROMPT_TEXT = """
Based on the provided sentence and claim, evaluate the probability that the claim is supported by the sentence.
Your answer must be a decimal number between 0 and 1, and you must strictly follow the format below:
<answer>probability_value</answer>
Where probability_value is the result you calculate.

Sentence:
{sentence}

Claim:
{claim}
"""

SYSTEM_PROMPT_TEXT = """
You will be provided with a source sentence and a target claim.
Your task is to evaluate how likely it is that the claim is true based *only* on the information provided in the sentence.

Your answer must be a single decimal number between 0 and 1, formatted exactly as follows:
<answer>probability_value</answer>

Where `probability_value` represents the probability of supporting the claim.
"""



