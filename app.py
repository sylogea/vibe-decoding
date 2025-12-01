import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@st.cache_resource(show_spinner=False)
def load_model():
	MODEL_NAME = "gpt2"
	device = "cuda" if torch.cuda.is_available() else "cpu"
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
	model.to(device)
	model.eval()
	with torch.no_grad():
		embedding_weight = model.transformer.wte.weight
		embedding_norms = embedding_weight.norm(dim=1)
		embedding_norms = torch.where(
			embedding_norms == 0,
			torch.ones_like(embedding_norms) * 1e-6,
			embedding_norms,
		)
	return tokenizer, model, embedding_weight, embedding_norms, device


def generate_vanilla(tokenizer, model, device, prompt_text, max_new_tokens):
	if not prompt_text.strip():
		return ""
	input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
	with torch.no_grad():
		for _ in range(max_new_tokens):
			outputs = model(input_ids=input_ids)
			logits = outputs.logits[:, -1, :]
			next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
			input_ids = torch.cat([input_ids, next_token_id], dim=-1)
	text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
	return text


def generate_vibe(tokenizer, model, embedding_weight, embedding_norms, device, prompt_text, alpha, max_new_tokens, top_k_expectation):
	if not prompt_text.strip():
		return ""
	input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
	embedding_weight = embedding_weight.to(device)
	embedding_norms = embedding_norms.to(device)
	with torch.no_grad():
		for _ in range(max_new_tokens):
			outputs = model(input_ids=input_ids)
			logits = outputs.logits[:, -1, :].squeeze(0)
			probs = torch.softmax(logits, dim=-1)

			if top_k_expectation > probs.shape[0]:
				top_k_expectation = probs.shape[0]

			top_probs, top_indices = torch.topk(probs, top_k_expectation)
			top_probs = top_probs / top_probs.sum()

			top_embeddings = embedding_weight[top_indices] / embedding_norms[top_indices].unsqueeze(1)
			vibe_direction = (top_probs.unsqueeze(1) * top_embeddings).sum(dim=0)

			projection = embedding_weight @ vibe_direction
			vibe_bias = projection / embedding_norms

			scores = logits + alpha * vibe_bias
			next_token_id = torch.argmax(scores).view(1, 1)
			input_ids = torch.cat([input_ids, next_token_id.to(device)], dim=-1)
	text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
	return text


def main():
	st.set_page_config(page_title="Vibe decoding vs vanilla (GPT-2)", layout="wide")
	st.title("Vibe decoding vs vanilla")
	st.caption("GPT-2 (~124M) demo comparing greedy decoding with vibe decoding")

	tokenizer, model, embedding_weight, embedding_norms, device = load_model()

	with st.sidebar:
		st.header("Controls")
		alpha = st.slider("α (vibe strength)", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
		max_new_tokens = st.slider("Max new tokens", min_value=5, max_value=80, value=40, step=5)
		top_k_expectation = st.slider("Top-k for vibe direction expectation", min_value=10, max_value=500, value=100, step=10)
		st.markdown(
			"Adjust α to see how strongly the vibe direction biases decoding. "
			"α = 0 reduces to vanilla greedy decoding."
		)

	default_prompt = "In a future where small language models are everywhere,"
	prompt_text = st.text_area("Prompt", value=default_prompt, height=120)

	generate = st.button("Generate")

	if generate:
		with st.spinner("Running both decoders..."):
			vanilla_text = generate_vanilla(
				tokenizer=tokenizer,
				model=model,
				device=device,
				prompt_text=prompt_text,
				max_new_tokens=max_new_tokens,
			)
			vibe_text = generate_vibe(
				tokenizer=tokenizer,
				model=model,
				embedding_weight=embedding_weight,
				embedding_norms=embedding_norms,
				device=device,
				prompt_text=prompt_text,
				alpha=alpha,
				max_new_tokens=max_new_tokens,
				top_k_expectation=top_k_expectation,
			)

		col_vanilla, col_vibe = st.columns(2)
		with col_vanilla:
			st.subheader("Vanilla greedy decoding")
			st.write(vanilla_text)
		with col_vibe:
			st.subheader("Vibe decoding")
			st.write(vibe_text)


if __name__ == "__main__":
	main()
