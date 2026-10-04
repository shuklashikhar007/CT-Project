"""LangChain layer. The LLM never produces numbers that go into the models:
  1. extract_params(): pulls candidate values out of a paper's text; each must carry a verbatim quote,
     and we machine-check that the quote really occurs in the source and contains the number.
  2. explain(): narrates results computed by models.py, restricted to the numbers in its context.
"""
from __future__ import annotations
import json, os, re
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

PROVIDERS = {  # label -> (init_chat_model provider, env var, default model)
    "Groq": ("groq", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "Google Gemini": ("google_genai", "GOOGLE_API_KEY", "gemini-2.5-flash"),
    "Anthropic Claude": ("anthropic", "ANTHROPIC_API_KEY", "claude-sonnet-5-5"),
    "OpenAI": ("openai", "OPENAI_API_KEY", "gpt-4.1-mini"),
}


def get_llm(label: str, model: str, api_key: str | None):
    from langchain.chat_models import init_chat_model
    provider, env, _ = PROVIDERS[label]
    if api_key:
        os.environ[env] = api_key
    return init_chat_model(model, model_provider=provider, temperature=0)


# ------------------------------------------------------------------ extraction
class Param(BaseModel):
    name: str = Field(description="e.g. 'Activation energy', 'Pre-exponential factor', 'Reaction order (H2)', "
                                  "'Cell voltage', 'Faradaic efficiency', 'Exchange current density'")
    value: float = Field(description="Numeric value exactly as printed in the text")
    unit: str
    conditions: str = Field(description="Catalyst / electrolyzer type, T, P, j, composition range the value applies to")
    quote: str = Field(description="VERBATIM sentence or table row from the text containing the value (max 50 words)")


class Extraction(BaseModel):
    params: list[Param]


TARGETS = {
    "Sabatier kinetics (Ni/Al2O3)": "rate-law parameters for CO2 methanation on Ni/alumina: activation energy, "
                                    "pre-exponential factor or rate constant (with reference T), reaction orders, "
                                    "adsorption constants, valid T and P range.",
    "Electrolyzer performance": "electrolyzer cell voltage vs current density, Faradaic efficiency, exchange current "
                                "density, area-specific resistance, operating T and P.",
}

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You extract numerical parameters from scientific text. Extract ONLY values explicitly printed in the "
               "text. Never estimate, convert, or infer. Every item needs a verbatim quote. If nothing relevant is "
               "present, return an empty list."),
    ("human", "Target: {target}\n\n--- TEXT START ---\n{text}\n--- TEXT END ---"),
])


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\u2212", "-").replace("\u2009", " ")).strip().lower()


def _number_in(quote: str, value: float) -> bool:
    nums = re.findall(r"-?\d+(?:[.,]\d+)?", _norm(quote))
    return any(abs(float(n.replace(",", ".")) - value) <= 1e-9 * max(1, abs(value)) for n in nums)


def extract_params(llm, text: str, target: str, max_chars: int = 80_000) -> pd.DataFrame:
    text = text[:max_chars]
    chain = _PROMPT | llm.with_structured_output(Extraction)
    out: Extraction = chain.invoke({"target": TARGETS[target], "text": text})
    src = _norm(text)
    rows = []
    for p in out.params:
        rows.append(dict(name=p.name, value=p.value, unit=p.unit, conditions=p.conditions, quote=p.quote,
                         quote_in_source=_norm(p.quote) in src, number_in_quote=_number_in(p.quote, p.value)))
    df = pd.DataFrame(rows, columns=["name", "value", "unit", "conditions", "quote", "quote_in_source", "number_in_quote"])
    df["grounded"] = df.quote_in_source & df.number_in_quote
    return df


# ------------------------------------------------------------------ analyst
_ANALYST = ChatPromptTemplate.from_messages([
    ("system", "You are a reaction-engineering reviewer for a Mars ISRU Sabatier plant design. Answer ONLY from the "
               "JSON CONTEXT (outputs of deterministic models). Do not introduce numbers that are not in the context; "
               "if the context cannot answer, say what is missing. Flag any item marked unverified. Be concise and "
               "quantitative. Distinguish thermodynamic (equilibrium) results from kinetic ones."),
    ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}"),
])


def explain(llm, context: dict, question: str) -> str:
    return (_ANALYST | llm).invoke({"context": json.dumps(context, indent=1, default=float), "question": question}).content