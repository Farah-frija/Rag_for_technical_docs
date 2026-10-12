"""
manual_rag.prompts
===================
Every LLM prompt string + its JSON schema lives here, grouped by pipeline
stage, and NOWHERE else. Indexing and retrieval code imports prompts from
here; it never inlines an f-string prompt. This is what makes prompts
reviewable/tunable in one place and testable independent of the pipeline
logic that fills them in.
"""
