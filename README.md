# Production-Grade Customer Support LLM

## Overview

This project demonstrates fine-tuning and deployment of a lightweight open-source Large Language Model for the Customer Support domain.

The system is capable of:

* Handling refund and cancellation requests
* Generating professional customer responses
* Toxicity filtering
* Prompt guardrails
* FastAPI deployment
* Latency benchmarking

---

## Tech Stack

* Python
* Transformers
* PEFT / LoRA
* TinyLlama 1.1B
* FastAPI
* ngrok
* Hugging Face

---

## Features

### Fine-Tuning

* Parameter-efficient LoRA fine-tuning
* Lightweight customer support adaptation

### AI Safety

* Toxicity filtering using Detoxify
* Prompt injection protection
* Unsafe prompt blocking

### Deployment

* FastAPI backend
* Public API deployment using ngrok
* Swagger API documentation

### Benchmarking

* Latency measurement
* GPU memory monitoring
* Tokens/sec benchmarking

---

## Dataset

Bitext Customer Support Dataset from Hugging Face.

---

## Model

Base Model:

* TinyLlama/TinyLlama-1.1B-Chat-v1.0

Fine-Tuning:

* LoRA adaptation
* 100 training samples
* 1 epoch

---

## API Endpoints

### GET /

Health check endpoint.

### POST /chat

Accepts customer support queries.

---

## Hugging Face Model

https://huggingface.co/satyadube1/customer-support-llm

---

## Future Improvements

* vLLM benchmarking
* llama.cpp deployment
* RAG integration
* Quantized inference
* Kubernetes deployment

---

## Author

Satyanarayan Dubey
