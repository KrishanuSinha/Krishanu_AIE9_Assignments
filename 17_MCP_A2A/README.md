<p align = "center" draggable="false" ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719"
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 17: Model Context Protocol (MCP) & Agent-to-Agent (A2A) Protocol</h1>

| Session Sheet | Recording     | Slides        | Repo         | Homework      | Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [MCP Servers & A2A](../00_Docs/Session_Sheets/17_MCP_Servers_and_A2A) |[Recording!](https://us02web.zoom.us/rec/share/_iJT-kZiYacyz23fjU3N7w7mZIUFJqGXV48RDqCkCY3avsmngKtzK0SNs0I7k74.xICq6NSv6l6GqAFU) <br> passcode: `fJ9tx4h.`| [Session 17 Slides](https://www.canva.com/design/DAG-ELapG4g/6vDMm63RBwKVsSZvheorVA/edit?utm_content=DAG-ELapG4g&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [(Optional) Session 17 Assignment: MCP Servers & A2A](https://forms.gle/qtjQFfoEF8aykTWy5) | [Feedback 3/12](https://forms.gle/sJwD1a6LLn9NU9s48) |
---

## 📚 Useful Resources

**MCP (Model Context Protocol)**
- [MCP Official Docs](https://modelcontextprotocol.io/) — Spec, tutorials, and guides
- [MCP-UI](https://mcpui.dev/) — Official standard for interactive UI in MCP
- [MCP Auth Guide (Auth0)](https://auth0.com/blog/mcp-specs-update-all-about-auth/) — Deep dive into MCP auth spec updates

**A2A (Agent-to-Agent Protocol)**
- [A2A Official Docs](https://a2a-protocol.org/latest/) — Spec and guides
- [A2A GitHub Repo](https://github.com/a2aproject/A2A) — Protocol spec and implementations
- [Announcing A2A (Google Blog)](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — Protocol vision and motivation

**MCP vs A2A**
- [A2A and MCP (Official)](https://a2a-protocol.org/latest/topics/a2a-and-mcp/) — How they complement each other

---

# Running the MCP Server

### 1. Install dependencies

```bash
uv sync
```

### 2. Set up environment variables

Copy the example env file and fill in your OpenAI API key:

```bash
cp .env.example .env
```

### 3. Run the MCP server locally

```bash
uv run server.py
```

The server will start on `http://localhost:8000`.

### 4. Expose the server with ngrok (for remote/Claude Desktop access)

In a separate terminal, start an ngrok tunnel:

```bash
ngrok http 8000
```

Copy the ngrok forwarding URL (e.g. `https://xxxx-xx-xx-xx-xx.ngrok-free.app`) and restart the server with it:

```bash
ISSUER_URL=https://xxxx-xx-xx-xx-xx.ngrok-free.app uv run server.py
```

> **Note:** The `ISSUER_URL` must match the public URL clients use to reach the server, otherwise OAuth authentication will fail.

---

# Build 🏗️

In today's assignment, we'll be building an MCP server with OAuth authentication — a cat shop application that exposes tools for browsing products, managing a cart, and checking out.

- 🤝 Breakout Room #1
  - Set up the MCP server with OAuth and the product database
  - Explore the MCP tools: `list_products`, `get_product`, `add_to_cart`, `view_cart`, `remove_from_cart`, `checkout`

- 🤝 Breakout Room #2
  - Connect an MCP client to the server
  - Build an end-to-end interaction flow using the MCP tools

# Ship 🚢

The completed MCP server and client integration!

### Deliverables

- A short Loom of either:
  - the MCP server you built and a demo of the client interacting with it; or
  - the notebook you created for the Advanced Build

# Share 🚀

Make a social media post about your final application!

### Deliverables

- Make a post on any social media platform about what you built!

Here's a template to get you started:

```
🚀 Exciting News! 🚀

I am thrilled to announce that I have just built and shipped an MCP server with OAuth authentication! 🎉🤖

🔍 Three Key Takeaways:
1️⃣
2️⃣
3️⃣

Let's continue pushing the boundaries of what's possible in the world of AI and tool integration. Here's to many more innovations! 🚀
Shout out to @AIMakerspace !

#MCP #ModelContextProtocol #OAuth #Innovation #AI #TechMilestone

Feel free to reach out if you're curious or would like to collaborate on similar projects! 🤝🔥
```

# Submitting Your Homework [OPTIONAL]

## Main Homework Assignment

Follow these steps to prepare and submit your homework assignment:

1. Review the MCP server code in `server.py` and the `app/` directory
2. Run the MCP server locally using `uv run server.py`
3. Connect to the server using an MCP client (e.g., Claude Desktop, or a custom client)
4. Test all available tools: browsing products, adding to cart, viewing cart, removing items, and checkout
5. Record a Loom video reviewing what you have learned from this session

## Questions

### ❓ Question #1:

Why is OAuth important for MCP servers, and what security considerations should you keep in mind when exposing tools to AI clients?

#### ✅ Answer:

OAuth is important because it helps the MCP server know who the user is and what they are allowed to do.

In your project, some tools are public, like viewing products. But other tools, like adding items to a cart or checking out, should only happen for a logged-in user. OAuth makes that possible by giving the server a secure way to verify the user before allowing those actions.

Why this matters

Without OAuth:

anyone could call protected tools

one user might access another user’s data

the AI client could use tools without proper permission

### ❓ Question #2:

What is the Agent-to-Agent (A2A) protocol, and how does it differ from MCP in terms of purpose and architecture? When would you choose A2A over MCP?

#### ✅ Answer:

A2A means Agent-to-Agent.

It is a protocol that allows one AI agent to talk to another AI agent.

###### MCP vs A2A

They are related, but they solve different problems.

###### MCP

MCP is used when an AI client, like ChatGPT, wants to connect to:

1. tools

2. resources

3. prompts

4. external systems

So MCP is mainly about:
###### AI client ↔ tool/server

###### Example:

ChatGPT uses an MCP server to view products or add items to a cart

###### A2A

A2A is used when one agent wants to work with another agent.

So A2A is mainly about:
###### agent ↔ agent

###### Example:

1. a shopping agent asks a shipping agent for delivery options

2. a planning agent asks a pricing agent for cost estimates

###### Main difference

MCP = gives an AI access to tools

A2A = lets one agent collaborate with another agent

###### When to choose A2A

Choose A2A when:

1. you have multiple agents

2. different agents have different specialties

3. agents need to delegate work to each other

###### Choose MCP when:

1. you want ChatGPT or another AI client to use your tools directly

2. you are exposing APIs, resources, or actions from a server

## Activity 1: Extend the MCP Server

Add at least one new tool to the cat shop MCP server (e.g., `search_products`, `update_cart_quantity`, or `get_order_history`). Ensure the new tool integrates properly with the existing database and OAuth authentication. Demo the new tool through an MCP client and include it in your Loom video.

###### Answer:
I added a new protected MCP tool called update_cart_quantity. It allows an authenticated user to modify the quantity of an item already in their shopping cart. The tool uses the existing OAuth authentication flow through _get_username() to identify the current user and updates the existing cart_items table in SQLite. If the quantity is set to 0, the item is removed from the cart. This ensures proper integration with both the database and user-specific authorization.”

## Advanced Activity: Build a Custom MCP Client

Build a custom MCP client that connects to the cat shop server over Streamable HTTP, authenticates via OAuth, and orchestrates a multi-step shopping flow (browse → add to cart → checkout). Compare the developer experience of MCP-based tool integration vs. traditional REST API calls.

Include your findings and a demo in your Loom video.
