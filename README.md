# CrediTrust Complaint Answering Chatbot

This project aims to develop an internal AI tool for CrediTrust Financial that transforms raw, unstructured customer complaint data into a strategic asset. The tool, an intelligent complaint-answering chatbot, empowers product managers, support teams, and compliance officers to quickly understand customer pain points across five major product categories: Credit Cards, Personal Loans, Buy Now, Pay Later (BNPL), Savings Accounts, and Money Transfers.

## Key Performance Indicators (KPIs)

* **Decrease the time:** Reduce the time for a Product Manager to identify a major complaint trend from days to minutes.
* **Empower non-technical teams:** Enable non-technical teams (like Support and Compliance) to get answers without needing a data analyst.
* **Proactive problem-solving:** Shift the company from reacting to problems to proactively identifying and fixing them based on real-time customer feedback.

## Motivation

CrediTrust’s internal teams face serious bottlenecks:
* Customer Support is overwhelmed by the volume of incoming complaints.
* Product Managers struggle to identify the most frequent or critical issues.
* Compliance & Risk teams are reactive rather than proactive.
* Executives lack visibility into emerging pain points.

This chatbot addresses these issues by providing quick, evidence-backed insights from customer feedback.

## Solution: Retrieval-Augmented Generation (RAG) Chatbot

The core of this solution is a Retrieval-Augmented Generation (RAG) agent that:
* Allows internal users to ask plain-English questions (e.g., “Why are people unhappy with BNPL?”).
* Uses semantic search (via a vector database like FAISS) to retrieve the most relevant complaint narratives.
* Feeds the retrieved narratives into a language model (LLM) that generates concise, insightful answers.
* Supports multi-product querying, allowing filtering or comparison across financial services.

## Data

The project uses complaint data from the Consumer Financial Protection Bureau (CFPB), which contains real customer complaints with a short issue label, a free-text narrative, and product/company information. The `Consumer complaint narrative` is the core input for embedding and retrieval.

**Dataset Link:** (Please ensure you download the CFPB dataset and place it in the `data/` directory as `complaints.csv`)

## Project Setup and Running Instructions

Follow these steps to set up and run the complaint-answering chatbot:

### 1. Clone the Repository

```bash
git clone https://github.com/seid2015a/Complaint-Analysis-for-Financial-Services-Week_6
cd cred_trust_chatbot.
