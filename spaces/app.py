import os
import contextlib
import logging
import random
import re
import time
from pathlib import Path

import gradio as gr
import nltk
from cleantext import clean

from summarize import load_model_and_tokenizer, summarize_via_tokenbatches
from utils import load_example_filenames, truncate_word_count, saves_summary
from textrank import get_summary

example_path = "./"
nltk.download("stopwords")  # TODO=find where this requirement originates from

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def proc_submission(
    input_text: str,
    model_size: str,
    num_beams,
    token_batch_length,
    length_penalty,
    repetition_penalty,
    no_repeat_ngram_size,
    max_input_length: int = 1024,
):

    settings = {
        "length_penalty": float(length_penalty),
        "repetition_penalty": float(repetition_penalty),
        "no_repeat_ngram_size": int(no_repeat_ngram_size),
        "encoder_no_repeat_ngram_size": 4,
        "num_beams": int(num_beams),
        "min_length": 4,
        "max_length": int(token_batch_length // 4),
        "early_stopping": True,
        "do_sample": False,
    }
    st = time.perf_counter()
    history = {}
    clean_text = clean(input_text, lower=False)
    max_input_length = 1024 if "base" in model_size.lower() else max_input_length
    clean_text = get_summary(clean_text)
    processed = truncate_word_count(clean_text, max_input_length)

    if processed["was_truncated"]:
        tr_in = processed["truncated_text"]
        # create elaborate HTML warning
        input_wc = re.split(r"\s+", input_text)
        msg = f"""
        <div style="background-color: #FFA500; color: white; padding: 20px;">
        <h3>Warning</h3>
        <p>Input text was truncated to {max_input_length} words. That's about {100*max_input_length/len(input_wc):.2f}% of the submission.</p>
        </div>
        """
        logging.warning(msg)
        history["WARNING"] = msg
    else:
        tr_in = input_text
        msg = None

    if len(input_text) < 50:
        # this is essentially a different case from the above
        msg = f"""
        <div style="background-color: #880808; color: white; padding: 20px;">
        <h3>Warning</h3>
        <p>Input text is too short to summarize. Detected {len(input_text)} characters.
        Please load text by selecting an example from the dropdown menu or by pasting text into the text box.</p>
        </div>
        """
        logging.warning(msg)
        logging.warning("RETURNING EMPTY STRING")
        history["WARNING"] = msg

        return msg, "", []

    _summaries = summarize_via_tokenbatches(
        tr_in,
        model,
        tokenizer,
        batch_length=token_batch_length,
        **settings,
    )
    sum_text = [f"Section {i}: " + s["summary"][0] for i, s in enumerate(_summaries)]
    sum_scores = [
        f" - Section {i}: {round(s['summary_score'],4)}"
        for i, s in enumerate(_summaries)
    ]

    sum_text_out = "\n".join(sum_text)
    history["Summary Scores"] = "<br><br>"
    scores_out = "\n".join(sum_scores)
    rt = round((time.perf_counter() - st) / 60, 2)
    print(f"Runtime: {rt} minutes")
    html = ""
    html += f"<p>Runtime: {rt} minutes on CPU</p>"
    if msg is not None:
        html += msg

    html += ""

    # save to file
    saved_file = saves_summary(_summaries)

    return html, sum_text_out, scores_out, saved_file


def load_single_example_text(
    example_path: str or Path="./example.txt",
    max_pages=20,
):
    """
    load_single_example - a helper function for the gradio module to load examples
    Returns:
        list of str, the examples
    """
    global name_to_path
    full_ex_path = name_to_path[example_path]
    full_ex_path = Path(full_ex_path)

    if full_ex_path.suffix == ".txt":
        with open(full_ex_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
        text = clean(raw_text, lower=False)
    else:
        logging.error(f"Unknown file type {full_ex_path.suffix}")
        text = "ERROR - check example path"

    return text

if __name__ == "__main__":
    logging.info("Starting app instance")
    os.environ[
        "TOKENIZERS_PARALLELISM"
    ] = "false"  # parallelism on tokenizers is buggy with gradio
    logging.info("Loading summ models")
    with contextlib.redirect_stdout(None):
        model, tokenizer = load_model_and_tokenizer(
            "SmartPy/bart-large-cnn-finetuned-scientific_summarize"
        )
        
    name_to_path = load_example_filenames(example_path)
    logging.info(f"Loaded {len(name_to_path)} examples")
    demo = gr.Blocks()
    _examples = list(name_to_path.keys())
    with demo:

        gr.Markdown("# Document Summarization with Long-Document Transformers")
        gr.Markdown(
            "This is an example use case for fine-tuned long document transformers. The model is trained on Scientific Article summaries (via the Yale Scientific Article Summarization Dataset). The models in this demo are [Bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)."
        )
        with gr.Column():

            gr.Markdown("## Load Inputs & Select Parameters")
            gr.Markdown(
                "Enter text below in the text area. The text will be summarized [using the selected parameters](https://huggingface.co/blog/how-to-generate). "
            )
            with gr.Row(variant="compact"):
                with gr.Column(scale=0.5, variant="compact"):

                    model_size = gr.Radio(
                        choices=["bart-large-cnn"],
                        label="Model Variant",
                        value="bart-large-cnn",
                    )
                    num_beams = gr.Radio(
                        choices=[2, 3, 4],
                        label="Beam Search: # of Beams",
                        value=2,
                    )
                with gr.Column(variant="compact"):
                    example_name = gr.Dropdown(
                        _examples,
                        label="Examples",
                        value=random.choice(_examples),
                    )

            with gr.Row():
                input_text = gr.Textbox(
                    lines=4,
                    label="Input Text (for summarization)",
                    placeholder="Enter text to summarize, the text will be cleaned and truncated on Spaces. Narrative, academic (both papers and lecture transcription), and article text work well. May take a bit to generate depending on the input text :)",
                )
                with gr.Column(min_width=100, scale=0.5):
                    load_examples_button = gr.Button(
                        "Load Example",
                    )

        with gr.Column():
            gr.Markdown("## Generate Summary")
            gr.Markdown(
                "Summarization should take ~1-2 minutes for most settings, but may extend up to 5-10 minutes in some scenarios."
            )
            summarize_button = gr.Button(
                "Summarize!",
                variant="primary",
            )

            output_text = gr.HTML("<p><em>Output will appear below:</em></p>")
            gr.Markdown("### Summary Output")
            summary_text = gr.Textbox(
                label="Summary", placeholder="The generated summary will appear here"
            )
            gr.Markdown(
                "The summary scores can be thought of as representing the quality of the summary. less-negative numbers (closer to 0) are better:"
            )
            summary_scores = gr.Textbox(
                label="Summary Scores", placeholder="Summary scores will appear here"
            )

            text_file = gr.File(
                label="Download Summary as Text File",
                file_count="single",
                type="file",
                interactive=False,
            )

        gr.Markdown("---")
        with gr.Column():
            gr.Markdown("### Advanced Settings")
            with gr.Row(variant="compact"):
                length_penalty = gr.inputs.Slider(
                    minimum=0.5,
                    maximum=1.0,
                    label="length penalty",
                    default=0.7,
                    step=0.05,
                )
                token_batch_length = gr.Radio(
                    choices=[512, 768, 1024, 1536],
                    label="token batch length",
                    value=1024,
                )

            with gr.Row(variant="compact"):
                repetition_penalty = gr.inputs.Slider(
                    minimum=1.0,
                    maximum=5.0,
                    label="repetition penalty",
                    default=3.5,
                    step=0.1,
                )
                no_repeat_ngram_size = gr.Radio(
                    choices=[2, 3, 4],
                    label="no repeat ngram size",
                    value=3,
                )
        with gr.Column():
            gr.Markdown("### About the Model")
            gr.Markdown(
                "These models are fine-tuned on the [1000 most cited papers in the ACL Anthology Network (AAN)](http://arxiv.org/pdf/1909.01716.pdf).The goal was to create a model that can generalize well and is useful in summarizing lots of text in academic and daily usage."
            )
            gr.Markdown("---")

        load_examples_button.click(
            fn=load_single_example_text, inputs=[example_name], outputs=[input_text]
        )

        summarize_button.click(
            fn=proc_submission,
            inputs=[
                input_text,
                model_size,
                num_beams,
                token_batch_length,
                length_penalty,
                repetition_penalty,
                no_repeat_ngram_size,
            ],
            outputs=[output_text, summary_text, summary_scores, text_file],
        )

    demo.launch(share=True,enable_queue=True, debug=True)
