---
title: Meraki Tagger App
emoji: ✨
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Meraki Tagger Web App

This is the official web interface for the **Meraki Tagger**, a humanitarian sentence classification model.

## How it Works
- **Frontend**: Custom HTML/CSS/JS UI for a clean user experience.
- **Backend**: FastAPI server running inside a Docker container.
- **Inference**: Runs locally using the `AaranNihalani/MerakiTagger` model (DeBERTa-v3-large).

## Usage
Paste a paragraph of text (e.g., daily report, needs assessment) into the input box. The app will split it into sentences and tag each one with relevant humanitarian sectors and indicators.

## Thresholds
This app applies custom per-label thresholds optimized for F1 score, ensuring high-quality predictions even for rare tags.
