"""LLM Prompt 模板"""

SEMANTIC_EXTRACTION_PROMPT = """You are a movie analyst. Given the following movie information, extract structured semantic tags.

**Movie Title:** {title}
**Overview:** {overview}
**Genres:** {genres}
**Director:** {director}
**Main Cast:** {cast}
**Release Year:** {year}

Output a JSON object with EXACTLY these fields:
{{
  "genre_fine": ["<up to 3 fine-grained genre tags, e.g. psychological-thriller, romantic-comedy>"],
  "mood": "<one of: dark, tense, uplifting, melancholic, whimsical, intense, serene, humorous, dramatic, mysterious>",
  "theme": ["<up to 3 core themes, e.g. redemption, coming-of-age, survival>"],
  "pace": "<one of: fast, moderate, slow, varied>",
  "audience": "<one of: general, family, mature, cinephile, teen>",
  "narrative_style": "<one of: linear, nonlinear, ensemble, character-study, documentary-style, anthology>",
  "visual_style": "<one of: realistic, stylized, gritty, colorful, minimalist, epic>",
  "era_setting": "<time period of the story, e.g. modern, 1940s, futuristic, medieval>",
  "emotion_arc": "<brief 5-word description of emotional arc>"
}}

Output ONLY the JSON, no explanation."""


CAPTION_PROMPT = """Based on the following movie information, describe what its movie poster would likely look like in ONE detailed sentence.

Focus on: dominant color palette, visual composition style, mood/atmosphere, and genre visual cues.

Movie: {title} ({year})
Genre: {genres}
Overview: {overview}
Director: {director}

Respond with ONLY the poster description sentence."""
