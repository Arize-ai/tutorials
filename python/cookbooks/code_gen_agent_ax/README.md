# Code Generation Agent Example for Arize AX

> Last updated: 2026-03-10

## Overview

This folder contains examples for building a Code Generation (Code-Gen) agent using the LangChain library. This agent is designed to generate, refine, and validate code using OpenAI models.

This version is intended to send traces to Arize AX; there is a [Arize Phoenix version](https://github.com/Arize-ai/phoenix/tree/main/examples/code_gen_agent) in the Phoenix repository.

## Features

* Construction of a Code-Gen agent workflow using LangChain
* Integration with OpenAI models for generating and refining code
* Example usage of tools such as code analysis, execution, and generation
* Auto-instrumentation with OpenInference decorators to fully instrument the agent
* End-to-end tracing with Arize AX to track agent performance

## Requirements

* LangChain library
* OpenAI API key
* Langgraph (for managing agent logic and workflows)
* Python >3.8, <=3.12
* Gradio (for UI)

## Installation

1. Install `uv`
1. Run directly with `uv run --isolated --with-requirements requirements.txt python app.py`
1. View the UI and input the required Keys(OpenAI, Arize AX API Key, Arize Space ID)

## Usage

1. Run the `app.py` script to start the RAG agent.
1. Click on the local host link provided in the output.
1. Interact with the agent by entering prompts and receiving generated code responses.

## Files

* `app.py`: The main script for starting the application, this will run the web server with default port(7860)
* `agent.py`: The main script for the code generation agent
* `tools.py`: Contains tools for code analysis, generation, execution, and merging
* `requirements.txt`: Lists the required libraries for the project

## Notes

* All the Key's must be inputted from the UI application.
* This application will support the HTML based sources.
