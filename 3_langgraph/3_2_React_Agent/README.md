# Intelligent Multi-Agent Workflow using LangGraph

## Description

This project implements a dynamic, agentic workflow using **LangGraph**, simulating intelligent decision-making across multiple AI tools. The workflow is centered around a **Supervisor node** that routes user queries to the appropriate processing tool based on the content and context of the input.

### Components:

- **Supervisor Node**  
  Analyzes the user query and dynamically routes it to one of three tools — **LLM**, **RAG**, or **Web Crawler**.

- **LLM Node**  
  Handles general reasoning and creative tasks using a language model.

- **RAG Node**  
  Performs retrieval-augmented generation to answer questions with factual relevance (e.g., USA-related queries).

- **Web Crawler Node**  
  Fetches live or up-to-date information from the web.

- **Validation Node**  
  Validates the output from any tool to ensure quality and accuracy. If validation fails, the query is reprocessed.

- **Final Output Node**  
  Presents the final response to the user once validation is passed.

This project showcases **agent coordination**, **tool routing**, **output validation**, and **resilient workflows** using **LangGraph** and **LangChain**.
