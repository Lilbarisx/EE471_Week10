import torch
from transformers import pipeline
import requests
from PIL import Image
import urllib.request

def main():
    print("Task 1: Basic Implementations of Hugging Face Transformers")
    
    # 1. Sentiment Analysis
    print("\n--- 1. Sentiment Analysis ---")
    sentiment_pipeline = pipeline("sentiment-analysis")
    sentences = ["I've been waiting for a EE471 course my whole life.", "I hate EE471 course"]
    results = sentiment_pipeline(sentences)
    for sent, res in zip(sentences, results):
        print(f"'{sent}' -> {res}")

    # 2. Zero-shot classification
    print("\n--- 2. Zero-shot Classification ---")
    zero_shot_pipeline = pipeline("zero-shot-classification")
    sequence_to_classify = "Berkshire keeps their cash reserves at an extremely high level."
    candidate_labels = ['finance', 'sports', 'technology']
    zero_shot_result = zero_shot_pipeline(sequence_to_classify, candidate_labels)
    print(f"Text: '{sequence_to_classify}'")
    print(f"Labels: {zero_shot_result['labels']}")
    print(f"Scores: {zero_shot_result['scores']}")

    # 3. Text Generation
    print("\n--- 3. Text Generation ---")
    generator = pipeline("text-generation")
    prompt = "If I continue to successfully complete all in-class exercises in EE471 course,"
    gen_results = generator(prompt, max_length=35, num_return_sequences=2)
    print(f"Prompt: '{prompt}'")
    for i, res in enumerate(gen_results):
        print(f"Option {i+1}: {res['generated_text']}")

    # 4. Mask Filling
    print("\n--- 4. Mask Filling ---")
    fill_mask = pipeline("fill-mask")
    # For some models, the mask token is <mask>, for BERT it is [MASK]. 
    # fill-mask uses the default model, usually distilroberta which uses <mask>
    mask_prompt = "To understand generative AI, one must study <mask> well."
    try:
        mask_results = fill_mask(mask_prompt)
        print(f"Prompt: '{mask_prompt}'")
        for res in mask_results[:2]:
            print(f"Filled: {res['sequence']} (score: {res['score']:.3f})")
    except Exception:
        # Fallback to [MASK] if <mask> fails
        mask_prompt = "To understand generative AI, one must study [MASK] well."
        mask_results = fill_mask(mask_prompt)
        print(f"Prompt: '{mask_prompt}'")
        for res in mask_results[:2]:
            print(f"Filled: {res['sequence']} (score: {res['score']:.3f})")

    # 5. Named Entity Recognition (NER)
    print("\n--- 5. Named Entity Recognition (NER) ---")
    ner_pipeline = pipeline("ner", aggregation_strategy="simple")
    ner_text = "I am Nate, a research assistant in Izmir Institute of Technology, and currently living and working in beautiful city İzmir in Türkiye."
    ner_results = ner_pipeline(ner_text)
    print(f"Text: '{ner_text}'")
    for entity in ner_results:
        print(f"Entity: {entity['word']}, Label: {entity['entity_group']}, Score: {entity['score']:.3f}")

    # 6. Question Answering
    print("\n--- 6. Question Answering ---")
    qa_pipeline = pipeline("question-answering")
    context = ner_text
    questions = ["What is the person's name?", "Where does he work?", "Where does he live?"]
    print(f"Context: '{context}'")
    for q in questions:
        ans = qa_pipeline(question=q, context=context)
        print(f"Q: {q} -> A: {ans['answer']}")

    # 7. Summarization
    print("\n--- 7. Summarization ---")
    summarizer = pipeline("summarization")
    long_text = "The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century, often compared to the Great Depression of the 1930s. Triggered by the bursting of the United States housing bubble, its effects rippled across the globe, leading to the collapse of major financial institutions and a deep international recession. The crisis began with the subprime mortgage market. In the early 2000s, low interest rates and a push for homeownership led banks to issue high-risk loans to borrowers with poor credit."
    summary = summarizer(long_text, max_length=50, min_length=20, do_sample=False)
    print(f"Original Length: {len(long_text)} characters")
    print(f"Summary: {summary[0]['summary_text']}")

    # 8. Translation
    print("\n--- 8. Translation ---")
    # Using t5-small for en to fr
    translator = pipeline("translation_en_to_fr", model="t5-small")
    sentence_to_translate = "The 2008 Global Financial Crisis stands as the most severe economic collapse of the 21st century, often compared to the Great Depression."
    translation = translator(sentence_to_translate)
    print(f"English: '{sentence_to_translate}'")
    print(f"French: '{translation[0]['translation_text']}'")

    # 9. Image Classification
    print("\n--- 9. Image Classification ---")
    try:
        img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        image = Image.open(requests.get(img_url, stream=True).raw)
        
        image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
        img_results = image_classifier(image)
        print("Image URL: sample cat image")
        for res in img_results[:3]:
            print(f"Label: {res['label']}, Score: {res['score']:.3f}")
    except Exception as e:
        print(f"Image classification failed: {e}")

    # 10. Automatic Speech Recognition
    print("\n--- 10. Automatic Speech Recognition ---")
    try:
        # Use a short sample audio to save time
        audio_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
        urllib.request.urlretrieve(audio_url, "sample_audio.flac")
        
        # NOTE: Using a smaller model here to avoid massive download time during class,
        # but the PDF specifies openai/whisper-large-v3. We'll use the required one.
        asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
        asr_result = asr_pipeline("sample_audio.flac")
        print(f"Transcribed Text: {asr_result['text']}")
    except Exception as e:
        print(f"ASR failed: {e}")

if __name__ == "__main__":
    main()
