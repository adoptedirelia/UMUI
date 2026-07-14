INSTRUCTION = """
To help ensure accurate and consistent judgments, use the following guidelines when assigning a score from 0 to 9. Each level represents how strongly the video supports the claim:

9 — Full, unambiguous support
The video clearly and directly shows the exact event described in the claim, with no need for interpretation.

8 — Almost complete support
The main content of the claim is shown in the video, with only minor ambiguities (e.g., small uncertainties in identity or location). The claim is strongly supported overall.

6-7 — Strong partial support
The video strongly suggests the claim is true, but important details may be missing, unclear, or partially obscured. The support is substantial but not definitive.

4-5 — Moderate partial support
Some aspects of the video align with the claim, but large portions remain unclear, incomplete, or open to interpretation. The evidence points in the general direction of the claim but lacks clarity or completeness.

2-3 — Minimal weak support
Only small or indirect cues hint at the claim being true. The evidence is insufficient for confident verification.

1 — Very weak or speculative support
There may be extremely faint or indirect references, but nothing concrete supports the claim.

0 — No support or contradiction
The video does not relate to the claim at all or directly contradicts it.
"""

PROMPT = """
Based on the provided video and text, evaluate the probability that the text is true.
Your answer must be a single in number between 0 and 9.

The text to evaluate is:

{text}
"""

SYSTEM_PROMPT = """
You will be provided with a video and a text statement.  
Your task is to evaluate how likely it is that the statement is true based on the visual and audio evidence in the video.  

Your answer must be a single integer number between 0 and 9.

"""

INSTRUCTION_AUDIO = """
To help ensure accurate and consistent judgments, use the following guidelines when assigning a score from 0 to 9. Each level represents how strongly the audio supports the claim:

9 — Full, unambiguous support
The audio clearly and directly captures the exact event, speech, or sound described in the claim, with no need for inference.
8 — Almost complete support
The main content of the claim is audible in the recording, with only minor ambiguities (e.g., slight background noise or minor uncertainty regarding the speaker's identity). The claim is strongly supported overall.
6-7 — Strong partial support
The audio strongly suggests the claim is true, but important details may be missing, muffled, or partially obscured by interference. The support is substantial but not definitive.
4-5 — Moderate partial support
Some aspects of the audio align with the claim, but large portions remain unintelligible, incomplete, or open to interpretation. The auditory evidence points in the general direction of the claim but lacks clarity.
2-3 — Minimal weak support
Only small or indirect acoustic cues hint at the claim being true. The evidence is insufficient for confident verification.
1 — Very weak or speculative support
There may be extremely faint or indirect auditory references, but nothing concrete supports the claim.
0 — No support or contradiction
The audio does not relate to the claim at all or directly contradicts it (e.g., a different language is spoken or a different sound occurs).
"""



PROMPT_AUDIO = """
Based on the provided audio and text, evaluate the probability that the text is true.
Your answer must be a single integer number between 0 and 9.

The text to evaluate is:

{text}

"""

SYSTEM_PROMPT_AUDIO = """
You will be provided with an audio file and a text statement.
Your task is to evaluate how likely it is that the statement is true based on the auditory evidence, including speech content, tone of voice, ambient sounds, and background noise.
Your answer must be a single integer number between 0 and 9.
"""