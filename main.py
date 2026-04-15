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

@bot.tree.command(name="research", description="Start an autonomous research loop on a topic.")
@app_commands.describe(
    topic="The name of the research project (slug).",
    iterations="How many search/analyze cycles to run (Default: 3)."
)
async def research(interaction: discord.Interaction, topic: str, iterations: int = 3):
    await interaction.response.send_message(f"🚀 **Initializing Research Agent** for: `{topic}`\n*This will take a few minutes...*")
    
    # Run the research in the background so the bot doesn't hang
    try:
        await run_autonomous_loop(topic, max_iterations=iterations)
        await interaction.followup.send(f"✅ **Research Complete!** Data stored in collection: `{topic}`")
    except Exception as e:
        await interaction.followup.send(f"❌ **Research Failed:** {e}")

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
