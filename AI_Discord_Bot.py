import os
import discord
from discord.ext import commands
from openai import AsyncOpenAI
from dotenv import load_dotenv

# 1. Configuration: Load Environment Variables
# This looks for the .env file and loads the variables inside it
load_dotenv() 
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')

# Fail gracefully if the token is missing
if not DISCORD_TOKEN:
    raise ValueError("Error: DISCORD_TOKEN not found. Make sure your .env file is set up correctly.")

# 2. Initialize Clients
ai_client = AsyncOpenAI(
    base_url="http://localhost:8080/v1", 
    api_key="sk-no-key-required" 
)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# 3. The Generation Logic
async def generate_thought(topic):
    try:
        response = await ai_client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": "You are a helpful Discord bot. Respond directly to the prompt."},
                {"role": "user", "content": f"Topic: {topic}"}
            ],
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        
        # Safety net for reasoning models
        if not content:
            msg_dump = response.choices[0].message.model_dump()
            if msg_dump.get("reasoning_content"):
                content = f"**[Internal Thought Process]**\n{msg_dump['reasoning_content']}"
            else:
                content = "Error: The AI returned a completely empty response."
                
        return content

    except Exception as e:
        return f"Error communicating with local AI: {e}"

# 4. The Manual Command with Splitter Logic
@bot.command(name='prompt')
async def prompt_cmd(ctx, *, topic: str):
    print(f"-> Processing !prompt: '{topic}'")
    
    await ctx.send(f"🧠 Thinking about: `{topic}`...")
    
    thought = await generate_thought(topic)
    
    if thought:
        # Split into chunks of 1900 characters
        chunks = [thought[i:i+1900] for i in range(0, len(thought), 1900)]
        
        for index, chunk in enumerate(chunks):
            try:
                await ctx.send(chunk)
            except Exception as e:
                print(f"<- [ERROR] Failed to send chunk {index + 1}: {e}")
                
        print(f"<- Successfully sent response ({len(chunks)} chunks).")

# 5. Boot Up
@bot.event
async def on_ready():
    print('--------------------------------')
    print(f'Logged in as {bot.user.name}')
    print('Ready! Type !prompt <topic> in Discord.')
    print('--------------------------------')

# Run the bot
bot.run(DISCORD_TOKEN)
