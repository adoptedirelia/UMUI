INSTRUCTION = """
To help you make accurate judgments, here is the guideline for determining "Yes" or "No":

YES:
- Full and unambiguous support: The video clearly shows the exact event described in the claim.
- Almost complete support: The main content is shown; minor ambiguities exist but overall claims are supported.
- Strong partial support: The video strongly suggests the claim is true; despite some missing details, the evidence is sufficient for verification.

NO:
- Moderate/Weak support: While there may be alignment, large portions are missing, unclear, or open to interpretation. The footage is insufficient for confident verification.
- Speculative support: Only slight or indirect references (related objects/settings) without concrete action.
- No support or contradiction: The video does not relate to the claim or shows the opposite.
"""

PROMPT = """
Based on the provided video and text, determine if the text is supported by the visual and audio evidence.

Your answer must be strictly "Yes" or "No" and follow the format below:
<answer>Result</answer>
Where Result is either Yes or No.

The text to evaluate is:
{text}
"""

SYSTEM_PROMPT = """
You will be provided with a video and a text statement.  
Your task is to evaluate whether the statement is true based on the visual and audio evidence in the video.

Your answer must be strictly "Yes" or "No", formatted exactly as follows:
<answer>Yes</answer>
OR
<answer>No</answer>
"""


INSTRUCTION_AUDIO = """
To help you make accurate judgments, here is the guideline for determining "Yes" or "No" for audio-based evidence:

YES:
- Full and unambiguous support: The audio clearly and directly contains the exact event or statement described in the claim.
- Almost complete support: The main content is clearly supported; minor ambiguities (like background noise) exist but the claim is valid.
- Strong partial support: The audio strongly suggests the claim is true; despite some missing details or incomplete phrases, the evidence is sufficient for verification.

NO:
- Moderate/Weak support: While there is some alignment, large portions are missing, unclear, or open to interpretation. The recording lacks the clarity needed for confident verification.
- Speculative support: Only small verbal cues or indirect references without concrete confirmation.
- No support or contradiction: The audio does not relate to the claim at all, or it directly contradicts it.
"""

PROMPT_AUDIO = """
Based on the provided audio and text, determine if the text statement is supported by the audio evidence.

Your answer must be strictly "Yes" or "No" and follow the format below:
<answer>Result</answer>
Where Result is either Yes or No.

The text to evaluate is:
{text}
"""

SYSTEM_PROMPT_AUDIO = """
You will be provided with an audio recording and a text statement.  
Your task is to evaluate whether the statement is true based on the spoken content, tone, and context within the audio.

Your answer must be strictly "Yes" or "No", formatted exactly as follows:
<answer>Yes</answer>
OR
<answer>No</answer>
"""



INSTRUCTION_TEXT = """
To help you make accurate judgments, here is the guideline for determining "Yes" or "No" based on textual evidence:

YES:
- Full and unambiguous support (Entailment): The sentence explicitly states the information in the claim.
- Almost complete support: The main assertions are present; synonyms or paraphrasing are used, but the core meaning is fully preserved.
- Strong partial support (Strong Implication): The sentence strongly implies the claim is true through logical inference, even if not explicitly stated.

NO:
- Moderate support (Topical Alignment): The sentence discusses the same subject or shares keywords, but the specific assertion in the claim is neither confirmed nor denied (Neutral).
- Weak/Speculative support: Only weak textual links or distant connections exist; inferring the claim would be speculative.
- No support or contradiction: The sentence is completely unrelated or directly contradicts the claim.
"""

PROMPT_TEXT = """
Based on the provided sentence and claim, determine if the claim is supported by the sentence.

Your answer must be strictly "Yes" or "No" and follow the format below:
<answer>Result</answer>
Where Result is either Yes or No.

Sentence:
{sentence}

Claim:
{claim}
"""

SYSTEM_PROMPT_TEXT = """
You will be provided with a source sentence and a target claim.
Your task is to evaluate whether the claim is true based *only* on the information provided in the sentence.

Your answer must be strictly "Yes" or "No", formatted exactly as follows:
<answer>Yes</answer>
OR
<answer>No</answer>
"""