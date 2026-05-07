import torch
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

class HuggingFaceEngine:
    def __init__(self):
        # We will use lazy initialization to save memory (load only when requested)
        self.pipelines = {}

    def get_pipeline(self, task_name, **kwargs):
        # Generate a unique key if specific model is provided
        model_name = kwargs.get('model', task_name)
        if model_name not in self.pipelines:
            print(f"Loading pipeline for {model_name}...")
            self.pipelines[model_name] = pipeline(task_name, **kwargs)
        return self.pipelines[model_name]

    def sentiment_analysis(self, text):
        if not text: return "Please enter text."
        pipe = self.get_pipeline("sentiment-analysis")
        result = pipe(text)
        return f"Label: {result[0]['label']}, Score: {result[0]['score']:.3f}"

    def zero_shot_classification(self, text, labels_str):
        if not text or not labels_str: return "Please enter text and labels."
        pipe = self.get_pipeline("zero-shot-classification")
        labels = [label.strip() for label in labels_str.split(",")]
        result = pipe(text, candidate_labels=labels)
        return {label: score for label, score in zip(result['labels'], result['scores'])}

    def text_generation(self, text, max_length=35):
        if not text: return "Please enter text."
        pipe = self.get_pipeline("text-generation")
        results = pipe(text, max_length=int(max_length), num_return_sequences=2)
        return "\n".join([f"Option {i+1}: {res['generated_text']}" for i, res in enumerate(results)])

    def mask_filling(self, text):
        if not text: return "Please enter text."
        pipe = self.get_pipeline("fill-mask")
        if "<mask>" not in text and "[MASK]" not in text:
            return "Error: Please include <mask> or [MASK] in the text."
        try:
            results = pipe(text)
        except Exception:
            if "<mask>" in text:
                results = pipe(text.replace("<mask>", "[MASK]"))
            else:
                results = pipe(text.replace("[MASK]", "<mask>"))
        
        return "\n".join([f"Filled: {res['sequence']} (Score: {res['score']:.3f})" for res in results[:2]])

    def named_entity_recognition(self, text):
        if not text: return "Please enter text."
        pipe = self.get_pipeline("ner", aggregation_strategy="simple")
        results = pipe(text)
        entities = [f"Entity: {res['word']}, Label: {res['entity_group']}, Score: {res['score']:.3f}" for res in results]
        return "\n".join(entities) if entities else "No entities found."

    def question_answering(self, context, question):
        if not context or not question: return "Please enter context and question."
        pipe = self.get_pipeline("question-answering")
        result = pipe(question=question, context=context)
        return f"Answer: {result['answer']} (Score: {result['score']:.3f})"

    def text_summarization(self, text):
        if not text: return "Please enter text."
        pipe = self.get_pipeline("summarization")
        input_len = len(text.split())
        max_len = min(50, max(10, input_len // 2))
        result = pipe(text, max_length=max_len, min_length=10, do_sample=False)
        return result[0]['summary_text']

    def text_translation(self, text):
        if not text: return "Please enter text."
        pipe = self.get_pipeline("translation_en_to_fr", model="t5-small")
        result = pipe(text)
        return result[0]['translation_text']

    def image_classification(self, image):
        if image is None: return "Please upload an image."
        pipe = self.get_pipeline("image-classification", model="google/vit-base-patch16-224")
        results = pipe(image)
        return {res['label']: res['score'] for res in results[:3]}

    def automatic_speech_recognition(self, audio_path):
        if audio_path is None: return "Please upload an audio file."
        # Note: Using whisper-large-v3 as per PDF. This might take some time to download.
        pipe = self.get_pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
        result = pipe(audio_path)
        return result['text']
