import gradio as gr
from hf_engine import HuggingFaceEngine

def create_demo():
    engine = HuggingFaceEngine()

    # 1. Sentiment Analysis
    iface1 = gr.Interface(
        fn=engine.sentiment_analysis,
        inputs=gr.Textbox(label="name", lines=5, value="I've been waiting for a EE471 course my whole life."),
        outputs=gr.Textbox(label="output", lines=5)
    )

    # 2. Zero-Shot
    iface2 = gr.Interface(
        fn=engine.zero_shot_classification,
        inputs=[
            gr.Textbox(label="text", lines=3, value="Berkshire keeps their cash reserves at an extremely high level."),
            gr.Textbox(label="candidate_labels", value="finance, sports, technology")
        ],
        outputs="label"
    )
    
    # 3. Text Generation
    iface3 = gr.Interface(
        fn=engine.text_generation,
        inputs=[
            gr.Textbox(label="prompt", lines=3, value="If I continue to successfully complete all in-class exercises in EE471 course,"),
            gr.Slider(minimum=10, maximum=100, value=35, step=1, label="max_length")
        ],
        outputs=gr.Textbox(label="output", lines=5)
    )
    
    # 4. Mask Filling
    iface4 = gr.Interface(
        fn=engine.mask_filling,
        inputs=gr.Textbox(label="text", lines=5, value="To understand generative AI, one must study <mask> well."),
        outputs=gr.Textbox(label="output", lines=5)
    )

    # 5. NER
    iface5 = gr.Interface(
        fn=engine.named_entity_recognition,
        inputs=gr.Textbox(label="text", lines=5, value="I am Nate, a research assistant in Izmir Institute of Technology, and currently living and working in beautiful city İzmir in Türkiye."),
        outputs=gr.Textbox(label="output", lines=5)
    )

    # 6. Question Answering
    iface6 = gr.Interface(
        fn=engine.question_answering,
        inputs=[
            gr.Textbox(label="context", lines=3, value="I am Nate, a research assistant in Izmir Institute of Technology, and currently living and working in beautiful city İzmir in Türkiye."),
            gr.Textbox(label="question", value="Where does he live?")
        ],
        outputs=gr.Textbox(label="output", lines=5)
    )

    # 7. Summarization
    iface7 = gr.Interface(
        fn=engine.text_summarization,
        inputs=gr.Textbox(label="text", lines=7, value="The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century, often compared to the Great Depression of the 1930s. Triggered by the bursting of the United States housing bubble, its effects rippled across the globe, leading to the collapse of major financial institutions and a deep international recession. The crisis began with the subprime mortgage market. In the early 2000s, low interest rates and a push for homeownership led banks to issue high-risk loans to borrowers with poor credit."),
        outputs=gr.Textbox(label="output", lines=5)
    )

    # 8. Translation
    iface8 = gr.Interface(
        fn=engine.text_translation,
        inputs=gr.Textbox(label="text", lines=5, value="The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century, often compared to the Great Depression."),
        outputs=gr.Textbox(label="output", lines=5)
    )

    # 9. Image Classification
    iface9 = gr.Interface(
        fn=engine.image_classification,
        inputs=gr.Image(type="filepath", label="image"),
        outputs="label"
    )

    # 10. Speech Recognition
    iface10 = gr.Interface(
        fn=engine.automatic_speech_recognition,
        inputs=gr.Audio(type="filepath", label="audio"),
        outputs=gr.Textbox(label="output", lines=5)
    )

    demo = gr.TabbedInterface(
        [iface1, iface2, iface3, iface4, iface5, iface6, iface7, iface8, iface9, iface10],
        ["1. Sentiment", "2. Zero-Shot", "3. Text Gen", "4. Mask Filling", "5. NER", "6. QA", "7. Summarize", "8. Translation", "9. Image Class", "10. Speech Rec"]
    )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
