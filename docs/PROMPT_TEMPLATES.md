# BatterySense AI prompt templates

BatterySense AI supports editable AI prompts for OpenAI and local Ollama models.

Prompt templates are stored in:

```text
config/prompt_templates.json
```

Inside the Streamlit sidebar, use **AI prompt settings** to:

- choose a prompt template,
- edit the free-text prompt,
- enable or disable custom prompting.

The app still sends only calculated BatterySense AI evidence to the AI backend. It does not send the full raw dataframe for interpretation.

Recommended local model instruction:

```text
Do not explain JSON. Do not invent values. Use only BatterySense AI evidence.
```

For lab QC reports, use the **Lab QC progress report** template.
