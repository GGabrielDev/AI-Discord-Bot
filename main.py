import discord
from discord import app_commands
from discord.ext import commands
import asyncio
import io
import chromadb
from agent.loop import run_autonomous_loop
from agent.crawler import run_focused_crawler
from agent.wiki_builder import generate_index_page
from query import (
    answer_question,
    build_translated_report_filename,
    infer_report_topic_from_filename,
    translate_and_archive_report,
)
from config.settings import DISCORD_TOKEN, CHROMA_DB_PATH
from llm.client import LocalLLM
from tools.search import close_search_client
from tools.scraper import close_scraper_client

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

    async def close(self):
        await close_search_client()
        await close_scraper_client()
        await LocalLLM.aclose()
        await super().close()

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

from agent.checkpoint import load_checkpoint, load_chain_checkpoint, load_ask_checkpoint, save_chain_checkpoint, delete_chain_checkpoint, request_soft_stop
from agent.planner import decompose_chain_prompt
from storage.vectordb import VectorDB

# --- Focused Crawler Command ---
@bot.tree.command(name="crawl_site", description="Recursively crawls a single website to build a specialized knowledge base.")
@app_commands.describe(
    url="The starting URL of the website to crawl.",
    topic="The collection name (database) to store the data in.",
    max_pages="Maximum number of pages to ingest (Default: 20).",
    max_depth="How deep to follow internal links (Default: 3)."
)
async def crawl_site(interaction: discord.Interaction, url: str, topic: str, max_pages: int = 20, max_depth: int = 3):
    # Truncate long URLs if they exceed limits
    safe_url = (url[:100] + '...') if len(url) > 100 else url
    await interaction.response.send_message(f"### 🕷️ Site Crawler Initialized: {topic}\n> **Target:** <{safe_url}>\n\n⚙️ **Calibrating Domain Shield...**")
    
    status_message = None
    async def discord_logger(message: str, is_sub_step: bool = False):
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

    try:
        count = await run_focused_crawler(
            base_url=url,
            topic=topic,
            max_pages=max_pages,
            max_depth=max_depth,
            log_func=discord_logger
        )
        await interaction.channel.send(f"✅ **Crawl Complete!** Successfully ingested {count} pages from the target domain into `{topic}`.")
    except Exception as e:
        await interaction.channel.send(f"❌ **Fatal Crawler Error:** {e}")

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
        await interaction.response.send_message(f"⚡ **RESUMING INTERRUPTED SESSION** — Subject: `{subject}`")
    else:
        # Safe truncation for echoing back potentially long subjects
        safe_subject = (subject[:100] + '...') if len(subject) > 100 else subject
        await interaction.response.send_message(f"🏗️ **Building Research Environment for `{safe_subject}`...**")
    
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
    finally:
        from agent.checkpoint import clear_soft_stop
        clear_soft_stop()

@bot.tree.command(name="finish", description="Soft-Stop: Gracefully wind down all running researches and chains after current iterations.")
async def finish(interaction: discord.Interaction):
    request_soft_stop()
    await interaction.response.send_message("🛑 **Soft Stop Requested:** The agent will gracefully wrap up its current active loop and finalize its reports. No new loops will be started.")

@bot.tree.command(name="chain_research", description="Decompose a massive prompt into multiple sub-topics and research them in an automated chain.")
@app_commands.describe(
    prompt="What is your massive, overarching research goal?",
    topic="The unified database topic to intelligently pool all the sub-topic findings into",
    max_depth="Max depth of sub-research (Recommended: 3-5)"
)
@app_commands.autocomplete(topic=topic_autocomplete)
async def chain_research(interaction: discord.Interaction, prompt: str, topic: str, max_depth: int = 4):
    # Safe truncation for prompts which can be massive
    safe_prompt = (prompt[:1500] + '...') if len(prompt) > 1500 else prompt
    await interaction.response.send_message(f"⛓️ **Analyzing Master Prompt:** `{safe_prompt}`\n> Decomposing into exhaustive sub-topics...")
    sub_topics = await decompose_chain_prompt(prompt)
    
    # Announce the formulated plan
    plan_msg = "### 📋 Chain Research Plan\n"
    for i, t in enumerate(sub_topics, 1):
        plan_msg += f"{i}. {t}\n"
    await interaction.channel.send(plan_msg)
        
    # We check if a chain checkpoint exists.
    checkpoint = load_chain_checkpoint(prompt)
    if checkpoint:
        start_idx = checkpoint["current_topic_index"]
        queries = checkpoint["sub_topics"]
        sub_topic = queries[start_idx]
        await interaction.channel.send(f"⚠️ Resuming exactly from interrupted chain: {topic} - `{sub_topic}`")
    else:
        # Save a new checkpoint so if we crash during decomposition, we can't resume, we just start over
        save_chain_checkpoint(
            prompt=prompt,
            original_topic=topic,
            iterations=len(sub_topics),
            depth=max_depth,
            sub_topics=sub_topics,
            current_topic_index=0
        )
        start_idx = 0
        queries = sub_topics
        
    for i in range(start_idx, len(queries)):
        sub_topic = queries[i]
        
        # Save progress at the start of each topic
        save_chain_checkpoint(
            prompt=prompt,
            original_topic=topic,
            iterations=len(queries),
            depth=max_depth,
            sub_topics=queries,
            current_topic_index=i
        )
        
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
        finally:
            from agent.checkpoint import clear_soft_stop
            clear_soft_stop()
            
    # Chain complete
    delete_chain_checkpoint(prompt)
    await interaction.channel.send(f"🏁 **ALL CHAINS COMPLETE.** Pool `{topic}` is rich with knowledge!")

@bot.tree.command(name="ask", description="Query your research database for English answers.")
@app_commands.describe(
    topic="Select an existing research project.",
    question="What do you want to know?",
    mode="Select internal analysis strategy (Fast/Balanced/Thorough).",
    resume_from="Optional: Attach a previous .md report to resume research and fill its gaps.",
    local_only="If True, skips all web searching/scraping and uses ONLY existing database data."
)
@app_commands.autocomplete(topic=topic_autocomplete)
@app_commands.choices(mode=[
    app_commands.Choice(name="Fast (Single query, 0 auto-loops, ~5s)", value="Fast"),
    app_commands.Choice(name="Balanced (3 queries, Max 1 auto-loop, ~15s+)", value="Balanced"),
    app_commands.Choice(name="Thorough (5 queries, Max 3 auto-loops, ~40s+)", value="Thorough"),
    app_commands.Choice(name="Omniscient (Uncapped gap-seeking, auto-loops until perfect)", value="Omniscient")
])
@app_commands.choices(style=[
    app_commands.Choice(name="Concise (Direct, efficient summaries)", value="Concise"),
    app_commands.Choice(name="Investigative (Exhaustive deep-dive, forensic analysis)", value="Investigative")
])
async def ask(interaction: discord.Interaction, topic: str, question: str, 
            mode: app_commands.Choice[str] = None, 
            style: app_commands.Choice[str] = None, 
            resume_from: discord.Attachment = None,
            local_only: bool = False):
    # Extract string from choice, default to Balanced
    mode_val = mode.value if mode else "Balanced"
    style_val = style.value if style else "Concise"

    # Acknowledge the command natively - Truncate long questions to stay within Discord's 2000 char limit
    safe_q = (question[:1500] + '...') if len(question) > 1500 else question
    existing_ask_checkpoint = None if resume_from else load_ask_checkpoint(topic, question, mode_val, style_val, local_only)
    if existing_ask_checkpoint and existing_ask_checkpoint.get("status") == "in_progress":
        await interaction.response.send_message(f"### 🧠 Intelligence Report: {topic}\n> **Q:** {safe_q}\n\n⚡ **Resuming Interrupted Ask Session...**")
    else:
        await interaction.response.send_message(f"### 🧠 Intelligence Report: {topic}\n> **Q:** {safe_q}\n\n⚙️ **Initializing Agentic Analysis...**")
    
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
    
    # --- Resume Logic ---
    resume_draft = None
    if resume_from:
        if not resume_from.filename.endswith(".md"):
            await interaction.channel.send("❌ **Error:** `resume_from` must be a `.md` Markdown file.")
            return
        
        try:
            await discord_status_logger(f"📥 **Downloading report for resumption:** `{resume_from.filename}`...")
            resume_bytes = await resume_from.read()
            resume_draft = resume_bytes.decode('utf-8')
            # If the user is resuming, we default to a deeper mode if possible
            if not mode: mode_val = "Omniscient"
        except Exception as e:
            await interaction.channel.send(f"❌ **Failed to read report:** {e}")
            return

    try:
        answer_data = await answer_question(
            topic, 
            question, 
            mode=mode_val, 
            style=style_val,
            log_func=discord_status_logger, 
            draft_callback=handle_draft, 
            no_web=local_only,
            _draft=resume_draft 
        )
        
        # 2. Package the response
        files = []
        
        en_text = answer_data.get("english", "")
        if not en_text:
            en_text = "⚠️ **Warning:** The English report synthesis failed or returned empty content."
        
        en_bytes = io.BytesIO(en_text.encode('utf-8'))
        en_filename = f"Report_{topic}.md"
        files.append(discord.File(fp=en_bytes, filename=en_filename))
        print(f"[Main] Prepared {en_filename} ({len(en_text.encode('utf-8')):,} bytes)")

        # 3. Deliver via Fresh Message (Avoids 15m Interaction Timeout)
        final_msg = (
            f"### 🏁 Intelligence Report Finalized: {topic}\n"
            f"> **Q:** {safe_q}\n\n"
            f"✅ Analysis Complete! Local-Only: `{local_only}`. English report attached below."
        )
        
        try:
            # We use interaction.channel.send because it is independent of the 15m interaction token
            await interaction.channel.send(content=final_msg, files=files)
            # Update the original "Initializing..." message to show we're done
            await interaction.edit_original_response(content="✅ **Analysis Complete.** Reports delivered as a new message.")
        except Exception as e:
            print(f"[Main] Failed to send final report message: {e}")
            await interaction.channel.send(f"🚨 **Final Report Delivery Failed:** {e}\n*(Note: Data should still be archived in your local knowledge_base)*")
            
    except Exception as e:
        await interaction.channel.send(f"🚨 **CRITICAL SYSTEM ERROR:** ```{e}```")
    finally:
        from agent.checkpoint import clear_soft_stop
        clear_soft_stop()

@bot.tree.command(name="translate", description="Translate a Markdown report into a target language.")
@app_commands.describe(
    report="Upload the .md report to translate.",
    target_language="Target language, e.g. Spanish, French, or Portuguese."
)
async def translate(interaction: discord.Interaction, report: discord.Attachment, target_language: str):
    if not report.filename.lower().endswith(".md"):
        await interaction.response.send_message("❌ **Error:** `report` must be a `.md` Markdown file.")
        return

    safe_lang = (target_language[:100] + "...") if len(target_language) > 100 else target_language
    await interaction.response.send_message(
        f"### 🌐 Report Translation Started\n> **Source:** `{report.filename}`\n> **Target:** `{safe_lang}`\n\n⚙️ **Translating Markdown report...**"
    )

    try:
        report_bytes = await report.read()
        report_text = report_bytes.decode("utf-8-sig")
        archive_topic = infer_report_topic_from_filename(report.filename)
        translated_text = await translate_and_archive_report(report_text, archive_topic, target_language)

        translated_bytes = io.BytesIO(translated_text.encode("utf-8"))
        translated_filename = build_translated_report_filename(report.filename, target_language)
        translated_file = discord.File(fp=translated_bytes, filename=translated_filename)

        await interaction.channel.send(
            content=(
                "### ✅ Translated Report Ready\n"
                f"> **Source:** `{report.filename}`\n"
                f"> **Target:** `{safe_lang}`\n\n"
                "Translated Markdown attached below."
            ),
            file=translated_file,
        )
        await interaction.edit_original_response(content="✅ **Translation Complete.** Translated report delivered as a new message.")
    except UnicodeDecodeError:
        await interaction.channel.send("❌ **Failed to read report:** Markdown files must be valid UTF-8 text.")
    except ValueError as e:
        await interaction.channel.send(f"❌ **Translation Error:** {e}")
    except Exception as e:
        await interaction.channel.send(f"🚨 **Translation Failed:** {e}")

# --- Sync Command (Admin only) ---
@bot.command()
@commands.is_owner()
async def sync(ctx):
    """Run '!sync' in Discord to push the Slash Commands to the server."""
    await bot.tree.sync()
    await ctx.send("✅ **Slash Command Tree Synced!** (Note: It may take a few mins to show up)")

bot.run(DISCORD_TOKEN)
