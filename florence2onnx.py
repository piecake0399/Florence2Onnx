if task == "<CAPTION_TO_PHRASE_GROUNDING>":
    prompt = f"{task}: {expr}. Identify the region."
else:
    prompt = f"{task} {expr}"
