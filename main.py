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
import os
import re
import shutil
from agent.wiki_builder import WIKI_ROOT, generate_index_page
from agent.summarizer import compress_raw_text, chunk_text
from tools.scraper import scrape_text_from_url
from storage.vectordb import VectorDB

# --- Slash Commands ---

@bot.tree.command(name="research", description="Start an autonomous research run.")
@app_commands.describe(
    subject="What do you want to learn about?",
    iterations="Thoroughness: How many cycles of analysis? (1-10)",
    depth="Depth: How many sites to scrape per query? (1-5)",
    topic="Optional: Existing collection to save into. (Leave empty to use subject name)"
)
@app_commands.autocomplete(topic=topic_autocomplete) # Use the same autocomplete we built!
async def research(interaction: discord.Interaction, subject: str, iterations: int = 3, depth: int = 2, topic: str = None):
    topic_val = topic if topic else subject
    
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
            topic=topic_val, 
            max_iterations=iterations, 
            depth=depth, 
            log_func=discord_logger # THE BRIDGE
        )
        
        try:
            await interaction.followup.send(f"🏁 **Mission Complete.** {count} sources archived in `{topic_val}`.")
        except discord.errors.HTTPException:
            # The interaction token expired (15 minute limit for long researches)
            await interaction.channel.send(f"<@{interaction.user.id}> 🏁 **Mission Complete.** {count} sources archived in `{topic_val}`.")
            
    except Exception as e:
        await interaction.channel.send(f"🚨 **CRITICAL SYSTEM ERROR:** ```{e}```")

@bot.tree.command(name="finish", description="Soft-Stop: Gracefully wind down all running researches and chains after current iterations.")
async def finish(interaction: discord.Interaction):
    request_soft_stop()
    await interaction.response.send_message("🛑 **Soft Stop Requested:** The agent will gracefully wrap up its current active loop and finalize its reports. No new loops will be started.")

@bot.tree.command(name="backfill_raw", description="Admin: Retroactively scrape and inject compressed RAW text for an entire past research topic.")
@app_commands.autocomplete(topic=topic_autocomplete)
async def backfill_raw(interaction: discord.Interaction, topic: str):
    await interaction.response.send_message(f"⚙️ **Backfill Initiated for `{topic}`.** Analyzing existing markdown index...")
    
    master_topic_dir = os.path.join(WIKI_ROOT, topic.replace(" ", "_").lower())
    os.makedirs(master_topic_dir, exist_ok=True)
    
    urls_to_scrape = set()
    files_moved = 0
    dirs_to_clean = set()
    
    # 1. Global sweep to extract URLs and structurally migrate files
    for root, dirs, files in os.walk(WIKI_ROOT):
        # Don't iterate over the target master directory itself to avoid recursive chaos
        if root.startswith(master_topic_dir):
            continue
            
        for filename in files:
            if not filename.endswith(".md") or filename == "index.md":
                continue
                
            filepath = os.path.join(root, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                match = re.search(r'\*\*Source URL:\*\* \[(.*?)\]', content)
                if match:
                    urls_to_scrape.add(match.group(1))
            
            # Physically migrate the file to the master directory
            new_filepath = os.path.join(master_topic_dir, filename)
            shutil.move(filepath, new_filepath)
            files_moved += 1
            dirs_to_clean.add(root)
            
    # Clean up empty prompt directories
    for empty_dir in dirs_to_clean:
        try:
            if not os.listdir(empty_dir): # Only delete if actually empty
                os.rmdir(empty_dir)
        except OSError:
            pass
            
    # Re-generate index since we broke the old paths
    if files_moved > 0:
        generate_index_page()
            
    if not urls_to_scrape:
        await interaction.channel.send("⚠️ No newly un-vacuumed URLs found in the overarching markdown index.")
        return
        
    await interaction.channel.send(f"🧹 Vacuumed **{files_moved}** `.md` files into `knowledge_base/{topic}/`.\n🔍 Found **{len(urls_to_scrape)}** unique overarching URLs. Igniting deep RAW compression backfill...")
    db = VectorDB(collection_name=topic)
    
    success = 0
    for dict_index, url in enumerate(urls_to_scrape):
        try:
            # We don't log to Discord heavily to avoid spam, just terminal
            print(f"[Backfill] Processing {dict_index+1}/{len(urls_to_scrape)}: {url}")
            text = await scrape_text_from_url(url)
            if len(text) > 300:
                compressed_raw = compress_raw_text(text)
                raw_chunks = chunk_text(compressed_raw)
                db.add_chunks(raw_chunks, url, chunk_type="raw")
                success += 1
        except Exception as e:
            print(f"[Backfill] Failed on {url}: {e}")
            
    await interaction.channel.send(f"✅ **Backfill Complete:** Successfully injected raw, compressed chunks for {success}/{len(urls_to_scrape)} sources into `{topic}`.")

@bot.tree.command(name="chain_research", description="Decompose a massive prompt into multiple sub-topics and research them in an automated chain.")
@app_commands.describe(
    prompt="What is your massive, overarching research goal?",
    topic="The unified database topic to intelligently pool all the sub-topic findings into",
    max_depth="Max depth of sub-research (Recommended: 3-5)"
)
@app_commands.autocomplete(topic=topic_autocomplete)
async def chain_research(interaction: discord.Interaction, prompt: str, topic: str, max_depth: int = 4):
    await interaction.response.send_message(f"⛓️ **Analyzing Master Prompt:** `{prompt}`\n> Decomposing into exhaustive sub-topics...")
    sub_topics = await decompose_chain_prompt(prompt)
    
    # Announce the formulated plan
    plan_msg = "### 📋 Chain Research Plan\n"
    for i, t in enumerate(sub_topics, 1):
        plan_msg += f"{i}. {t}\n"
    await interaction.channel.send(plan_msg)
        
    # We check if a chain checkpoint exists.
    checkpoint = load_chain_checkpoint(prompt, topic)
    if checkpoint:
        start_idx = checkpoint["current_index"]
        queries = checkpoint["queries"]
        sub_topic = queries[start_idx]
        await interaction.channel.send(f"⚠️ Resuming exactly from interrupted chain: {topic} - `{sub_topic}`")
    else:
        # Save a new checkpoint so if we crash during decomposition, we can't resume, we just start over
        save_chain_checkpoint(prompt, topic, sub_topics)
        start_idx = 0
        queries = sub_topics
        
    for i in range(start_idx, len(queries)):
        sub_topic = queries[i]
        
        # Save progress at the start of each topic
        save_chain_checkpoint(prompt, topic, queries, i)
        
        await interaction.channel.send(f"\n🚀 **Chain Progress {i+1}/{len(queries)} — Starting sub-chain:** `{sub_topic}`")
        
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
                subject=sub_topic,
                topic=topic,
                max_iterations=1, # One deep loop per sub-topic
                depth=max_depth,
                log_func=chain_logger
            )
        except Exception as e:
            await interaction.channel.send(f"🚨 **CHAIN FAILED on `{sub_topic}`:** ```{e}```")
            return # Stop the chain if a sub-loop critically fails
            
    # Chain complete
    delete_chain_checkpoint(prompt)
    await interaction.channel.send(f"🏁 **ALL CHAINS COMPLETE.** Pool `{topic}` is rich with knowledge!")

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
    # Acknowledge the command natively
    await interaction.response.send_message(f"### 🧠 Intelligence Report: {topic}\n> **Q:** {question}\n\n⚙️ **Initializing Agentic Analysis...**")
    
    # Extract string from choice, default to Balanced
    mode_val = mode.value if mode else "Balanced"
    
    status_message = None
    async def discord_status_logger(message: str, is_sub_step: bool = False):
        nonlocal status_message
        try:
            if not is_sub_step:
                status_message = await interaction.channel.send(f"> {message}")
            elif status_message:
                new_content = status_message.content + f"\n> ↳ {message}"
                if len(new_content) > 1900:
                    status_message = await interaction.channel.send(f"> ↳ {message} (continued...)")
                else:
                    await status_message.edit(content=new_content)
        except Exception:
            pass
            
    async def handle_draft(draft_text: str, iteration: int):
        markdown_bytes = io.BytesIO(draft_text.encode('utf-8'))
        file = discord.File(fp=markdown_bytes, filename=f"Draft_Iter_{iteration}_{topic}.md")
        try:
            await interaction.channel.send(f"⚠️ **Knowledge Gaps Found!** Intermediate draft (Iteration {iteration}) attached below. Agent is now pushing into Gap Trackers...", file=file)
        except Exception:
            pass
    
    # 1. Run the massive multi-query pipeline (this will take time)
    answer = await answer_question(
        topic, 
        question, 
        mode=mode_val, 
        log_func=discord_status_logger, 
        draft_callback=handle_draft, 
        language=language
    )
    
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
