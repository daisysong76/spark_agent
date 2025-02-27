import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ai_agent.config.settings import settings
from ai_agent.app.logger import logger
from ai_agent.app.retrieval import MilvusRetriever

# Tools
from ai_agent.tools.phone_tool import make_phone_call
from ai_agent.tools.email_tool import send_email
from ai_agent.tools.linkedin_tool import linkedin_action
from ai_agent.tools.google_search_tool import google_search

app = FastAPI(title="Production AI Agent", version="1.0")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

# Load generative model once (gpt2 example)
logger.info(f"Loading generative model: {settings.GENERATIVE_MODEL}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen_tokenizer = AutoTokenizer.from_pretrained(settings.GENERATIVE_MODEL)
gen_model = AutoModelForCausalLM.from_pretrained(settings.GENERATIVE_MODEL).to(device).eval()

retriever = MilvusRetriever()


@app.post("/query")
def query_agent(request: QueryRequest):
    user_query = request.query.lower()

    # 1. Check for tool triggers - naive approach
    if "call phone" in user_query:
        # e.g. "call phone +123456789 message Hello..."
        try:
            # This is a simplified parse
            number_part = user_query.split("call phone")[1].split("message")[0].strip()
            message_part = user_query.split("message")[1].strip()
            result = make_phone_call(number_part, message_part)
            return {"tool_response": result}
        except Exception as e:
            logger.error(f"Tool error: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=str(e))

    if "send email" in user_query:
        # e.g. "send email to bob@example.com subject hi body This is a test"
        # We'll do a simplified parse; real usage should be more robust.
        try:
            # For demonstration, we hardcode or do a minimal parse
            recipient = "bob@example.com"
            subject = "Hello from Production AI Agent"
            body = "This is a test email. Thanks!"
            result = send_email(recipient, subject, body)
            return {"tool_response": result}
        except Exception as e:
            logger.error(f"Tool error: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=str(e))

    if "linkedin" in user_query:
        # e.g. "linkedin post Some interesting update..."
        try:
            content = user_query.replace("linkedin", "").strip()
            result = linkedin_action(action="post", content=content)
            return {"tool_response": result}
        except Exception as e:
            logger.error(f"Tool error: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=str(e))

    if "google search" in user_query:
        # e.g. "google search best pizza in new york"
        try:
            search_term = user_query.replace("google search", "").strip()
            results = google_search(search_term)
            return {"search_results": results}
        except Exception as e:
            logger.error(f"Tool error: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=str(e))

    # 2. Otherwise, do retrieval-augmented generation
    try:
        top_docs = retriever.search(request.query, top_k=request.top_k)
        doc_texts = []
        for doc_match in top_docs:
            # doc_match.id is doc_id, doc_match.score is similarity
            doc_texts.append(f"DocID: {doc_match.id}, Score: {doc_match.score}")

        # Build context for generative model
        context = "\n".join(doc_texts)
        prompt = f"Context:\n{context}\nQuestion: {request.query}\nAnswer:"
        input_ids = gen_tokenizer.encode(prompt, return_tensors='pt').to(device)
        output_ids = gen_model.generate(input_ids, max_length=256, do_sample=True, top_p=0.9, top_k=50)
        answer = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        logger.info(f"RAG answer: {answer}")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"RAG error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Production features:

# Logging to see which path was taken (tool vs. retrieval).
# HTTPException usage for better error responses.
# Real tool calls with Twilio/SendGrid.
# GPU usage if available.
# Minimal “naive approach” to parse user requests; in a real system, you might use NLP or a library like LangChain to handle the logic for choosing tools vs. retrieval.

