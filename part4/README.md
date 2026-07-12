# Part 4 – LLM Integration (Track C)

## Overview

In this part, the trained Machine Learning model was integrated with a Large Language Model (LLM) to generate human-readable explanations for spam email predictions. The LLM receives the prediction result and important feature values, then returns a structured JSON explanation.

---

## LLM Provider

- **Provider:** OpenRouter
- **Model:** google/gemma-4-26b-a4b-it:free

---

## API Key Security

The API key was **not hardcoded** in the notebook.

Instead, it was securely stored as an **environment variable** using:

```python
import os
from getpass import getpass

os.environ["LLM_API_KEY"] = getpass("Enter API Key: ")
```

This prevents sensitive credentials from being exposed in the source code or GitHub repository.

---

## System Prompt

A system prompt was created to instruct the LLM to:

- Explain why the email was predicted as Spam or Ham.
- Mention important features such as spam score, links, exclamation marks, and capital letters.
- Keep explanations concise.
- Never reveal personal information.
- Return only valid JSON.

---

## JSON Output Schema

The LLM response follows this JSON schema:

```json
{
  "prediction": "Spam",
  "confidence": 0.88,
  "explanation": "The email is classified as Spam because it contains many suspicious characteristics."
}
```

The response was validated using the **jsonschema** library before being used.

---

## Machine Learning Prediction

The saved model (`best_model.pkl`) was loaded using Joblib.

The prediction was generated using:

- `predict()` for the predicted class.
- `predict_proba()` for prediction confidence.

Example:

- Prediction: Spam
- Confidence: 87.58%

---

## Temperature Comparison

Two temperatures were tested.

### Temperature = 0

- Deterministic output.
- Same explanation every time.
- Suitable for production systems.

### Temperature = 0.7

- More creative explanations.
- Slight wording variation.
- Useful for conversational responses.

---

## PII Guardrail

The prompt instructed the LLM not to reveal any personally identifiable information (PII).

Examples include:

- Email address
- Phone number
- Personal identifiers

The model generated explanations without exposing sensitive information.

---

## Sample Predictions

Three sample emails were evaluated.

| Sample | Prediction | Confidence |
|--------|------------|------------|
| Sample 1 | Spam | 0.88 |
| Sample 2 | Spam | 0.98 |
| Sample 3 | Spam | 0.98 |

---

## Workflow

1. Load the trained model (`best_model.pkl`).
2. Load the cleaned dataset.
3. Perform prediction using `predict()`.
4. Obtain confidence using `predict_proba()`.
5. Send prediction details to the LLM.
6. Receive explanation in JSON format.
7. Validate the JSON response.
8. Display the prediction and explanation.

---

## Conclusion

The Machine Learning model was successfully integrated with an LLM to provide clear and structured explanations for spam predictions. The API key was securely managed using environment variables, JSON outputs were validated, and prompt engineering techniques were used to produce reliable, human-readable explanations. This pipeline demonstrates how traditional Machine Learning models can be enhanced with Generative AI to improve interpretability and user understanding.

