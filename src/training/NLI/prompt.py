NLI_SYSTEM_PROMPT = """
You are a probability estimation model.
Analyze the video and the claim.
Output ONLY one single probability token representing the likelihood that the claim is true.
Do not output any other text or explanation.
"""

NLI_PROMPT = """
Claim: {text}
"""

NLI_SYSTEM_PROMPT_AUDIO = """
You are a probability estimation model.
Analyze the audio and the claim.
Output ONLY one single probability token representing the likelihood that the claim is true.
Do not output any other text or explanation.
"""

NLI_PROMPT_AUDIO = """
Claim: {text}
"""

NLI_SYSTEM_PROMPT_TEXT = """
You are a probability estimation model.
Analyze the sentence and the claim.
Output ONLY one single probability token representing the likelihood that the claim is true.
Do not output any other text or explanation.
"""

NLI_PROMPT_TEXT = """
Sentence: {sentence}
Claim: {claim}
"""