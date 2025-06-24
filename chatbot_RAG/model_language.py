from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import torch
import uvicorn
import os
from huggingface_hub import login

# Token HF
os.environ["HF_HUB_TOKEN"] = ""

login(os.environ["HF_HUB_TOKEN"]) 

# Carga del modelo y tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=False,
    token=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map={"": "cpu"},
    torch_dtype=torch.bfloat16,
    token=True
)
model.config.pad_token_id = tokenizer.eos_token_id

app = FastAPI()

class Query(BaseModel):
    query: str
    documents: List[str]

@app.post("/generate")
async def generate_response(query: Query):
    # Construir el prompt prompt
    system  = "Eres un asistente experto en responder preguntas de forma clara.\n\n"
    docs    = "Contexto:\n" + "\n".join(f"- {d}" for d in query.documents) + "\n\n"
    user_in = f"Usuario: {query.query}\nATHENEA:"
    prompt  = system + docs + user_in

     # Tokenizar y mover a dispositivo
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]


    # Generación
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # Decode
    text =  tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()
    return {"response": text}

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000,log_level="info")
