import discord
from discord import app_commands
from discord.ext import commands
import asyncio
import chromadb
from agent.loop import run_autonomous_loop
from query import answer_question
from config.settings import DISCORD_TOKEN, CHROMA_DB_PATH

# --- Setup ---
class ResearchBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self):
        # This is where we could load extensions, 
        # but for now we just prepare the tree.
        print("🔧 Setup hook active. Slash commands are ready to sync.")

bot = ResearchBot()

# --- Autocomplete Logic ---
async def topic_autocomplete(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:
    """Looks into ChromaDB and suggests existing collections."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collections = client.list_collections()
        # Filter collections based on what the user is currently typing
        choices = [
            app_commands.Choice(name=c.name, value=c.name)
            for c in collections if current.lower() in c.name.lower()
        ]
        return choices[:25] # Discord limit is 25 choices
    except Exception:
        return []

# --- Slash Commands ---

@bot.tree.command(name="research", description="Start an autonomous research run.")
@app_commands.describe(
    subject="What do you want to learn about?",
    iterations="Thoroughness: How many cycles of analysis? (1-10)",
    depth="Depth: How many sites to scrape per query? (1-5)",
    save_to="Optional: Existing collection to save into. (Leave empty to use subject name)"
)
@app_commands.autocomplete(save_to=topic_autocomplete) # Use the same autocomplete we built!
async def research(interaction: discord.Interaction, subject: str, iterations: int = 3, depth: int = 2, save_to: str = None):
    collection = save_to if save_to else subject
    await interaction.response.send_message(f"🏗️ **Building Research Environment for `{subject}`...**")
    
    # This is the "Bridge" function
    async def discord_logger(message):
        # We use the interaction channel to send live updates
        await interaction.channel.send(f"> {message}")

    try:
        # We pass our logger into the loop
        count = await run_autonomous_loop(
            subject=subject, 
            collection_name=collection, 
            max_iterations=iterations, 
            depth=depth, 
            log_func=discord_logger # THE BRIDGE
        )
        
        await interaction.followup.send(f"🏁 **Mission Complete.** {count} sources archived in `{collection}`.")
    except Exception as e:
        await interaction.channel.send(f"🚨 **CRITICAL SYSTEM ERROR:** ```{e}```")

@bot.tree.command(name="ask", description="Query your research database for answers.")
@app_commands.describe(
    topic="Select an existing research project.",
    question="What do you want to know?"
)
@app_commands.autocomplete(topic=topic_autocomplete) # THE MAGIC LINE
async def ask(interaction: discord.Interaction, topic: str, question: str):
    await interaction.response.defer() # Gives the AI time to think
    
    answer = await answer_question(topic, question, num_results=10, show_sources=False)
    
    response = (
        f"### 🧠 Intelligence Report: {topic}\n"
        f"> **Q:** {question}\n"
        f"---\n"
        f"{answer}\n"
        f"---\n"
        f"*Source: Local RAG Archive*"
    )
    await interaction.followup.send(response)

# --- Sync Command (Admin only) ---
@bot.command()
@commands.is_owner()
async def sync(ctx):
    """Run '!sync' in Discord to push the Slash Commands to the server."""
    await bot.tree.sync()
    await ctx.send("✅ **Slash Command Tree Synced!** (Note: It may take a few mins to show up)")

bot.run(DISCORD_TOKEN)
