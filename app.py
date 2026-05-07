import gradio as gr
from hf_engine import HuggingFaceEngine

def create_demo():
    engine = HuggingFaceEngine()

    with gr.Blocks(title="EE471 - Hugging Face Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 EE471 Week 10: Hugging Face Transformers Demo")
        gr.Markdown("Interactive demonstrations for various NLP, Vision, and Audio tasks implemented in OOP style.")

        with gr.Tab("1. Sentiment Analysis"):
            sa_input = gr.Textbox(label="Input Text", value="I've been waiting for a EE471 course my whole life.", lines=2)
            sa_btn = gr.Button("Analyze", variant="primary")
            sa_output = gr.Textbox(label="Result")
            sa_btn.click(fn=engine.sentiment_analysis, inputs=sa_input, outputs=sa_output)

        with gr.Tab("2. Zero-Shot Classification"):
            zs_text = gr.Textbox(label="Input Text", value="Berkshire keeps their cash reserves at an extremely high level.")
            zs_labels = gr.Textbox(label="Candidate Labels (comma separated)", value="finance, sports, technology")
            zs_btn = gr.Button("Classify", variant="primary")
            zs_output = gr.Label(label="Result")
            zs_btn.click(fn=engine.zero_shot_classification, inputs=[zs_text, zs_labels], outputs=zs_output)

        with gr.Tab("3. Text Generation"):
            tg_input = gr.Textbox(label="Prompt", value="If I continue to successfully complete all in-class exercises in EE471 course,")
            tg_len = gr.Slider(minimum=10, maximum=100, value=35, step=1, label="Max Length")
            tg_btn = gr.Button("Generate", variant="primary")
            tg_output = gr.Textbox(label="Generated Alternatives", lines=5)
            tg_btn.click(fn=engine.text_generation, inputs=[tg_input, tg_len], outputs=tg_output)

        with gr.Tab("4. Mask Filling"):
            mf_input = gr.Textbox(label="Input with Mask (<mask>)", value="To understand generative AI, one must study <mask> well.")
            mf_btn = gr.Button("Fill Mask", variant="primary")
            mf_output = gr.Textbox(label="Results", lines=3)
            mf_btn.click(fn=engine.mask_filling, inputs=mf_input, outputs=mf_output)

        with gr.Tab("5. NER"):
            ner_input = gr.Textbox(label="Input Text", value="I am Nate, a research assistant in Izmir Institute of Technology, and currently living and working in beautiful city İzmir in Türkiye.", lines=3)
            ner_btn = gr.Button("Extract Entities", variant="primary")
            ner_output = gr.Textbox(label="Results", lines=5)
            ner_btn.click(fn=engine.named_entity_recognition, inputs=ner_input, outputs=ner_output)

        with gr.Tab("6. Question Answering"):
            qa_context = gr.Textbox(label="Context", value="I am Nate, a research assistant in Izmir Institute of Technology, and currently living and working in beautiful city İzmir in Türkiye.", lines=3)
            qa_question = gr.Textbox(label="Question", value="Where does he live?")
            qa_btn = gr.Button("Answer", variant="primary")
            qa_output = gr.Textbox(label="Result")
            qa_btn.click(fn=engine.question_answering, inputs=[qa_context, qa_question], outputs=qa_output)

        with gr.Tab("7. Summarization"):
            sum_input = gr.Textbox(label="Long Text", value="The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century, often compared to the Great Depression of the 1930s. Triggered by the bursting of the United States housing bubble, its effects rippled across the globe, leading to the collapse of major financial institutions and a deep international recession. The crisis began with the subprime mortgage market. In the early 2000s, low interest rates and a push for homeownership led banks to issue high-risk loans to borrowers with poor credit.", lines=7)
            sum_btn = gr.Button("Summarize", variant="primary")
            sum_output = gr.Textbox(label="Summary", lines=4)
            sum_btn.click(fn=engine.text_summarization, inputs=sum_input, outputs=sum_output)

        with gr.Tab("8. Translation"):
            tr_input = gr.Textbox(label="English Text", value="The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century, often compared to the Great Depression.", lines=2)
            tr_btn = gr.Button("Translate (EN->FR)", variant="primary")
            tr_output = gr.Textbox(label="French Translation", lines=2)
            tr_btn.click(fn=engine.text_translation, inputs=tr_input, outputs=tr_output)

        with gr.Tab("9. Image Classification"):
            ic_input = gr.Image(type="pil", label="Upload an Image")
            ic_btn = gr.Button("Classify Image", variant="primary")
            ic_output = gr.Label(label="Predictions")
            ic_btn.click(fn=engine.image_classification, inputs=ic_input, outputs=ic_output)

        with gr.Tab("10. Speech Recognition"):
            asr_input = gr.Audio(type="filepath", label="Upload or Record Audio")
            asr_btn = gr.Button("Transcribe", variant="primary")
            asr_output = gr.Textbox(label="Transcription")
            asr_btn.click(fn=engine.automatic_speech_recognition, inputs=asr_input, outputs=asr_output)

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
