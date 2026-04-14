import os
import json
import re
import discord
from discord.ext import commands
from openai import AsyncOpenAI
from dotenv import load_dotenv
from ddgs import DDGS # <-- Updated import

load_dotenv() 
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')

if not DISCORD_TOKEN:
    raise ValueError("Error: DISCORD_TOKEN not found in .env file.")

ai_client = AsyncOpenAI(
    base_url="http://localhost:8080/v1", 
    api_key="sk-no-key-required" 
)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

def perform_web_search(query: str, max_results=3) -> str:
    print(f"[Search] Searching the web for: '{query}'")
    try:
        # <-- Updated function call
        results = DDGS().text(query, max_results=max_results) 
        if not results:
            return "No web search results found."
        
        formatted_results = "\n\n".join([f"Source: {res['title']}\nSummary: {res['body']}" for res in results])
        return formatted_results
    except Exception as e:
        return f"Search failed: {e}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Searches the internet for up-to-date information, news, or facts you do not know.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to send to the search engine."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

async def generate_thought(topic):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful Discord bot. You have access to the internet. If asked about current events or facts you don't know, use the search_web tool."},
            {"role": "user", "content": topic}
        ]

        response = await ai_client.chat.completions.create(
            model="local-model",
            messages=messages,
            tools=tools,
            temperature=0.7
        )
        
        response_message = response.choices[0].message
        
        if response_message.tool_calls:
            print("-> [AI] Requested a tool call!")
            messages.append(response_message)
            
            for tool_call in response_message.tool_calls:
                if tool_call.function.name == "search_web":
                    args = json.loads(tool_call.function.arguments)
                    query = args.get("query")
                    
                    search_results = perform_web_search(query)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": search_results
                    })
            
            # --- NEW: Circuit Breaker System Prompt ---
            messages.append({
                "role": "system",
                "content": "You have received the search results. You MUST now provide a final, plain-text summary to the user. Do NOT use the search tool again. Do NOT output any raw <|tool_call|> tags."
            })

            print("-> [AI] Sending search results back for final analysis...")
            final_response = await ai_client.chat.completions.create(
                model="local-model",
                messages=messages,
                temperature=0.7
            )
            
            final_text = final_response.choices[0].message.content
            
            # --- NEW: Safety net to strip any rogue tags ---
            final_text = re.sub(r'<\|tool_call.*', '', final_text, flags=re.DOTALL).strip()
            return final_text
            
        else:
            print("-> [AI] Answered directly from internal knowledge.")
            return response_message.content

    except Exception as e:
        return f"Error communicating with AI: {e}"

@bot.command(name='prompt')
async def prompt_cmd(ctx, *, topic: str):
    print(f"\n--- Processing !prompt: '{topic}' ---")
    await ctx.send(f"🧠 Thinking about: `{topic}`...")
    
    thought = await generate_thought(topic)
    
    if thought:
        chunks = [thought[i:i+1900] for i in range(0, len(thought), 1900)]
        for chunk in chunks:
            await ctx.send(chunk)
        print(f"<- Successfully sent response.")

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')
    print('Ready! Try asking something about today’s news.')

bot.run(DISCORD_TOKEN)
