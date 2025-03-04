# Pico Code Agent

This is a simple code writing agent that allows the user to control a workspace and live edit code with an LLM. 

[demo](!assets/demo.png)

It uses OpenAI's advanced function calling capabilities along with a strong set of deterministic tools to be able to read and write codes with fairly high precision. Pinpointing small segments of code to have an LLM update it while understanding the full context is an ongoing challenge facing LLMs. 

This demo is my approach at simplifying this issue. 

## Disclaimer

__**This code can execute shell commands from inside the workspace directory without human approval. Use with caution and at your own discretion. You are responsible for your own actions**__

## Install 

```bash
git clone 
cd pico-code-agent
pip install -r requirements.txt
cp .env.copy .env
```

Add your OpenAI API key to the .env you just created.

**Usage**

```bash
python code_agent.py
```

If you want to change any of the settings for the agent you can do so in the `config.yaml`

#### Stretch Goals

- Self improvement
- Long term goals (several to many completions with no human intervention to complete a task)