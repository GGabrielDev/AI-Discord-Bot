import discord
from discord import app_commands
from discord.ext import commands
import asyncio
import io
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

from agent.checkpoint import load_checkpoint, load_chain_checkpoint, save_chain_checkpoint, delete_chain_checkpoint, request_soft_stop
from agent.planner import decompose_chain_prompt

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
    
    # Check for an interrupted session and notify the user
    checkpoint = load_checkpoint(subject)
    if checkpoint and checkpoint.get("status") == "in_progress":
        urls_done = len(checkpoint.get("seen_urls", set()))
        iter_at = checkpoint.get("current_iteration", 1)
        await interaction.response.send_message(
            f"⚡ **Resuming interrupted research on `{subject}`!**\n"
            f"> Previous session was interrupted at iteration {iter_at}/{iterations}.\n"
            f"> {urls_done} sources already processed — all prior work preserved.\n"
            f"> Picking up where we left off..."
        )
    else:
        await interaction.response.send_message(f"🏗️ **Building Research Environment for `{subject}`...**")
    
    status_message = None
    
    # This is the "Bridge" function
    async def discord_logger(message, is_sub_step=False):
        nonlocal status_message
        try:
            if not is_sub_step:
                status_message = await interaction.channel.send(f"> {message}")
            elif status_message:
                # Append to the dashboard message
                new_content = status_message.content + f"\n> ↳ {message}"
                # Guard against Discord's 2000 char message limit
                if len(new_content) > 1900:
                    status_message = await interaction.channel.send(f"> ↳ {message} (continued...)")
                else:
                    await status_message.edit(content=new_content)
        except Exception as e:
            print(f"Discord logger failed: {e}")

    try:
        # We pass our logger into the loop
        count = await run_autonomous_loop(
            subject=subject, 
            collection_name=collection, 
            max_iterations=iterations, 
            depth=depth, 
            log_func=discord_logger # THE BRIDGE
        )
        
        try:
            await interaction.followup.send(f"🏁 **Mission Complete.** {count} sources archived in `{collection}`.")
        except discord.errors.HTTPException:
            # The interaction token expired (15 minute limit for long researches)
            await interaction.channel.send(f"<@{interaction.user.id}> 🏁 **Mission Complete.** {count} sources archived in `{collection}`.")
            
    except Exception as e:
        await interaction.channel.send(f"🚨 **CRITICAL SYSTEM ERROR:** ```{e}```")

@bot.tree.command(name="finish", description="Soft-Stop: Gracefully wind down all running researches and chains after current iterations.")
async def finish(interaction: discord.Interaction):
    request_soft_stop()
    await interaction.response.send_message("🛑 **Soft Stop Requested:** The agent will gracefully wrap up its current active loop and finalize its reports. No new loops will be started.")

@bot.tree.command(name="chain_research", description="Decompose a massive prompt into multiple sub-topics and research them in an automated chain.")
@app_commands.describe(
    prompt="What is your massive, overarching research goal?",
    save_to="REQUIRED: Collection to pool all vectorized knowledge into.",
    iterations="Thoroughness per sub-topic? (1-10)",
    depth="Depth: How many sites to scrape per query? (1-5)"
)
@app_commands.autocomplete(save_to=topic_autocomplete)
async def chain_research(interaction: discord.Interaction, prompt: str, save_to: str, iterations: int = 10, depth: int = 5):
    # Check for macro chain checkpoint
    chain_state = load_chain_checkpoint(prompt)
    
    if chain_state and chain_state.get("status") == "in_progress":
        sub_topics = chain_state["sub_topics"]
        start_index = chain_state["current_topic_index"]
        await interaction.response.send_message(
            f"⛓️ **Resuming Master Chain:** `{prompt[:50]}...`\n"
            f"> Detected interrupted chain. Resuming at sub-topic {start_index+1}/{len(sub_topics)}."
        )
    else:
        await interaction.response.send_message(f"⛓️ **Analyzing Master Prompt:** `{prompt}`\n> Decomposing into exhaustive sub-topics...")
        sub_topics = await decompose_chain_prompt(prompt)
        start_index = 0
        
        # Announce the formulated plan
        plan_msg = "### 📋 Chain Research Plan\n"
        for i, t in enumerate(sub_topics, 1):
            plan_msg += f"{i}. {t}\n"
        await interaction.channel.send(plan_msg)
        
        # Save initial state
        save_chain_checkpoint(prompt, save_to, iterations, depth, sub_topics, start_index)
        
    for i in range(start_index, len(sub_topics)):
        current_topic = sub_topics[i]
        
        # Save progress at the start of each topic
        save_chain_checkpoint(prompt, save_to, iterations, depth, sub_topics, i)
        
        await interaction.channel.send(f"\n🚀 **Chain Progress {i+1}/{len(sub_topics)} — Starting sub-chain:** `{current_topic}`")
        
        status_message = None
        async def chain_logger(message, is_sub_step=False):
            nonlocal status_message
            try:
                if not is_sub_step:
                    status_message = await interaction.channel.send(f"> {message}")
                elif status_message:
                    new_content = status_message.content + f"\n> ↳ {message}"
                    if len(new_content) > 1900:
                        status_message = await interaction.channel.send(f"> ↳ {message} (cont...)")
                    else:
                        await status_message.edit(content=new_content)
            except Exception:
                pass
                
        try:
            await run_autonomous_loop(
                subject=current_topic,
                collection_name=save_to,
                max_iterations=iterations,
                depth=depth,
                log_func=chain_logger
            )
        except Exception as e:
            await interaction.channel.send(f"🚨 **CHAIN FAILED on `{current_topic}`:** ```{e}```")
            return # Stop the chain if a sub-loop critically fails
            
    # Chain complete
    delete_chain_checkpoint(prompt)
    await interaction.channel.send(f"🏁 **ALL CHAINS COMPLETE.** Pool `{save_to}` is rich with knowledge!")

@bot.tree.command(name="ask", description="Query your research database for answers.")
@app_commands.describe(
    topic="Select an existing research project.",
    question="What do you want to know?",
    mode="Select internal analysis strategy (Fast/Balanced/Thorough).",
    language="Optional: Force output language (e.g. Spanish, French). Defaults to English."
)
@app_commands.autocomplete(topic=topic_autocomplete)
@app_commands.choices(mode=[
    app_commands.Choice(name="Fast (Single query, 0 auto-loops, ~5s)", value="Fast"),
    app_commands.Choice(name="Balanced (3 queries, Max 1 auto-loop, ~15s+)", value="Balanced"),
    app_commands.Choice(name="Thorough (5 queries, Max 3 auto-loops, ~40s+)", value="Thorough"),
    app_commands.Choice(name="Omniscient (Uncapped gap-seeking, auto-loops until perfect)", value="Omniscient")
])
async def ask(interaction: discord.Interaction, topic: str, question: str, mode: app_commands.Choice[str] = None, language: str = "English"):
    # Acknowledge the command and set up for live edits
    await interaction.response.defer()
    
    # Extract string from choice, default to Balanced
    mode_val = mode.value if mode else "Balanced"
    
    async def discord_status_logger(message: str, is_sub_step: bool = False):
        # We edit the deferred response to show progress
        try:
            # We ignore is_sub_step for the ask dashboard, but we must accept the argument
            # since loop.py provides it.
            prefix = "> ↳ " if is_sub_step else "> "
            await interaction.edit_original_response(content=f"### 🧠 Intelligence Report: {topic}\n> **Q:** {question}\n\n{prefix}{message}")
        except discord.errors.HTTPException:
            # Interaction token expired during massive agentic gap loops. 
            pass
    
    # 1. Run the massive multi-query pipeline (this will take time)
    answer = await answer_question(topic, question, mode=mode_val, log_func=discord_status_logger, language=language)
    
    # 2. Package the response as a downloadable Markdown file
    markdown_bytes = io.BytesIO(answer.encode('utf-8'))
    file = discord.File(fp=markdown_bytes, filename=f"Report_{topic}.md")
    
    try:
        await interaction.edit_original_response(
            content=f"### 🧠 Intelligence Report: {topic}\n> **Q:** {question}\n\n✅ Analysis Complete. Generated report attached below.",
            attachments=[file]
        )
    except discord.errors.HTTPException:
        # Token expired (took heavily over 15 minutes). Send a standard message instead.
        await interaction.channel.send(
            content=f"<@{interaction.user.id}> ### 🧠 Intelligence Report: {topic}\n> **Q:** {question}\n\n✅ Analysis Complete. Generated report attached below.",
            file=file
        )

# --- Sync Command (Admin only) ---
@bot.command()
@commands.is_owner()
async def sync(ctx):
    """Run '!sync' in Discord to push the Slash Commands to the server."""
    await bot.tree.sync()
    await ctx.send("✅ **Slash Command Tree Synced!** (Note: It may take a few mins to show up)")

bot.run(DISCORD_TOKEN)
