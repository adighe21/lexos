import express from "express";
import { randomUUID } from "crypto";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json());
app.use(express.static(join(__dirname, "public")));
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  if (req.method === "OPTIONS") return res.status(200).end();
  next();
});

// ── Legal Agent System Prompts ────────────────────────────────────────────────
const SYSTEM_PROMPTS = {
  research: `You are a senior legal research attorney with expertise across federal and state law. Your role is to provide comprehensive, accurate legal research for practicing attorneys.

When responding:
- Lead with the most relevant legal standard or rule
- Cite key cases with jurisdiction and year (e.g. Smith v. Jones, 9th Cir. 2019)
- Identify any circuit splits or jurisdictional variations
- Note recent developments or trends in the law
- Structure responses with clear headings: Legal Standard → Key Cases → Analysis → Practice Notes
- Flag any areas of unsettled law
- End with a concise Summary for busy attorneys

Always remind the attorney to verify citations independently. Be thorough but practical.`,

  email: `You are a senior legal research attorney and contract specialist. Your role is to review contracts, legal documents, and agreements for practicing attorneys.

When reviewing contracts:
- Lead with an Executive Risk Summary (High/Medium/Low risk overall)
- Identify specific risky clauses with exact language quoted
- Flag missing standard protections
- Note non-standard or one-sided provisions
- Suggest specific redline language for problematic clauses
- Structure as: Risk Summary → Red Flags → Missing Provisions → Suggested Revisions
- Note jurisdiction-specific enforceability concerns

Be direct about risks. Attorneys need clear, actionable feedback.`,

  finance: `You are a legal compliance specialist with deep expertise in regulatory frameworks including GDPR, HIPAA, SOX, CCPA, PCI-DSS, FINRA, and industry-specific regulations.

When conducting compliance analysis:
- Identify the applicable regulatory framework(s)
- List specific requirements that apply
- Identify gaps or potential violations with specific regulatory citations
- Rate severity of each gap (Critical/High/Medium/Low)
- Provide specific remediation steps
- Structure as: Applicable Regulations → Compliance Gaps → Risk Assessment → Remediation Plan

Be specific and cite exact regulation sections (e.g. GDPR Article 13(1)(a)).`,

  data: `You are a litigation support specialist and discovery expert. Your role is to help attorneys organize, analyze, and extract value from discovery materials.

When assisting with discovery:
- Identify key facts, dates, parties, and relationships
- Flag potential privilege issues (attorney-client, work product)
- Build chronologies from document excerpts
- Identify inconsistencies or contradictions
- Suggest document requests or areas needing further discovery
- Structure as: Key Facts → Timeline → Issues Identified → Recommended Follow-up

Focus on what matters strategically for the case.`,

  summarization: `You are an expert legal writer specializing in litigation documents, legal memos, and attorney correspondence.

When drafting legal documents:
- Use proper legal document structure and formatting
- Write in a professional, precise legal style
- Include all standard sections for the document type
- Support arguments with legal standards and citations where relevant
- For memos: use IRAC structure (Issue, Rule, Analysis, Conclusion)
- For demand letters: be firm but professional
- For briefs: lead with the strongest arguments
- Always note [INSERT CASE-SPECIFIC FACTS] where attorney input is needed

Produce clean, polished first drafts ready for attorney review.`,

  matching: `You are a deposition preparation specialist and trial attorney coach. Your role is to help attorneys prepare comprehensive, strategic deposition question sets.

When preparing deposition questions:
- Develop questions to establish key facts for the case theory
- Create questions to lock in testimony and prevent later changes
- Design impeachment questions based on known inconsistencies
- Include foundation questions for exhibits
- Anticipate evasive answers and follow-up questions
- Structure as: Background/Foundation → Key Facts → Liability Issues → Damages → Impeachment Topics
- Note strategic objectives for each section

Focus on questions that advance the case theory and limit the witness's ability to equivocate.`,
};

// ── Helpers ───────────────────────────────────────────────────────────────────
async function callClaude(system, user) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": process.env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "claude-sonnet-4-6",
      max_tokens: 4096,
      system,
      messages: [{ role: "user", content: user }],
    }),
  });
  if (!res.ok) throw new Error(`Claude API error ${res.status}: ${await res.text()}`);
  return (await res.json()).content[0].text;
}

async function getEmbedding(text) {
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: { Authorization: `Bearer ${process.env.OPENAI_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({ input: text, model: "text-embedding-3-small" }),
  });
  if (!res.ok) throw new Error(`Embedding error ${res.status}`);
  return (await res.json()).data[0].embedding;
}

async function retrieveRAG(query) {
  if (!process.env.PINECONE_API_KEY || !process.env.PINECONE_INDEX_HOST || !process.env.OPENAI_API_KEY) return "";
  try {
    const vector = await getEmbedding(query);
    const res = await fetch(`https://${process.env.PINECONE_INDEX_HOST}/query`, {
      method: "POST",
      headers: { "Api-Key": process.env.PINECONE_API_KEY, "Content-Type": "application/json" },
      body: JSON.stringify({ vector, topK: 5, namespace: process.env.PINECONE_NAMESPACE || "lexos", includeMetadata: true }),
    });
    const data = await res.json();
    const matches = data.matches || [];
    if (!matches.length) return "";
    let ctx = "RELEVANT DOCUMENTS FROM KNOWLEDGE BASE:\n";
    matches.forEach((m, i) => { ctx += `[${i + 1}] (relevance: ${m.score?.toFixed(2)}) ${m.metadata?.text || ""}\n\n`; });
    return ctx;
  } catch (e) {
    return "";
  }
}

// ── Routes ────────────────────────────────────────────────────────────────────
app.get("/health", (req, res) => res.json({ status: "ok", platform: "LexOS", version: "1.0.0" }));

app.post("/api/agents/run", async (req, res) => {
  const { agent_type, input, use_rag } = req.body;
  if (!agent_type || !input) return res.status(400).json({ error: "agent_type and input are required" });
  const systemPrompt = SYSTEM_PROMPTS[agent_type];
  if (!systemPrompt) return res.status(400).json({ error: `Unknown agent type: ${agent_type}` });
  try {
    const start = Date.now();
    const ragContext = use_rag ? await retrieveRAG(input) : "";
    const userPrompt = ragContext ? `${ragContext}\n---\nATTORNEY REQUEST:\n${input}` : input;
    const result = await callClaude(systemPrompt, userPrompt);
    res.json({ status: "completed", result, latency_ms: Date.now() - start });
  } catch (err) {
    res.status(500).json({ status: "failed", error: err.message });
  }
});

app.post("/api/knowledge/ingest", async (req, res) => {
  const { documents } = req.body;
  if (!documents?.length) return res.status(400).json({ error: "documents[] required" });
  try {
    const ids = [];
    for (const doc of documents) {
      const embedding = await getEmbedding(doc.text);
      const id = randomUUID();
      await fetch(`https://${process.env.PINECONE_INDEX_HOST}/vectors/upsert`, {
        method: "POST",
        headers: { "Api-Key": process.env.PINECONE_API_KEY, "Content-Type": "application/json" },
        body: JSON.stringify({ vectors: [{ id, values: embedding, metadata: { ...(doc.metadata || {}), text: doc.text.slice(0, 2000), ingested_at: new Date().toISOString() } }], namespace: process.env.PINECONE_NAMESPACE || "lexos" }),
      });
      ids.push(id);
    }
    res.json({ ingested: ids.length, ids });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/knowledge/query", async (req, res) => {
  const { query, top_k = 5 } = req.body;
  if (!query) return res.status(400).json({ error: "query required" });
  try {
    const vector = await getEmbedding(query);
    const r = await fetch(`https://${process.env.PINECONE_INDEX_HOST}/query`, {
      method: "POST",
      headers: { "Api-Key": process.env.PINECONE_API_KEY, "Content-Type": "application/json" },
      body: JSON.stringify({ vector, topK: top_k, namespace: process.env.PINECONE_NAMESPACE || "lexos", includeMetadata: true }),
    });
    const data = await r.json();
    res.json({ results: (data.matches || []).map(m => ({ id: m.id, score: m.score, text: m.metadata?.text || "" })) });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log(`LexOS running on port ${PORT}`));
