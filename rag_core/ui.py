import gradio as gr


STARTER_QUESTIONS = [
    "Tell me about your work experience.",
    "What is your education background?",
    "Tell me about your research experience.",
    "What kind of projects have you worked on?",
    "What did you do at Juniper Networks?",
    "What was your role at CoRAL Lab?",
    "Explain PersonaRAG.",
]


def build_demo(runtime) -> gr.Blocks:
    """Construct the Gradio UI around the provided runtime object."""
    with gr.Blocks(title="Ask Ritam (Career QA Bot)") as demo:
        gr.Markdown(
            "# Ask Ritam\n"
            "A RAG-powered career assistant over my resume, website, and projects.\n"
            "Ask anything about my experience, projects, research, or education."
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Conversation", height=500)

                with gr.Row():
                    starter_dropdown = gr.Dropdown(
                        choices=STARTER_QUESTIONS,
                        label="Try A Starter Question",
                        value=STARTER_QUESTIONS[0],
                        allow_custom_value=False,
                        scale=4,
                    )
                    use_question_button = gr.Button("Use", scale=1)

                with gr.Row():
                    message_box = gr.Textbox(
                        placeholder="Ask anything about my career, projects, or research...",
                        lines=2,
                        scale=4,
                        show_label=False,
                    )
                    send_button = gr.Button("Send", variant="primary", scale=1)

                clear_button = gr.Button("Clear chat")

                send_button.click(
                    runtime.respond,
                    inputs=[message_box, chatbot],
                    outputs=[message_box, chatbot],
                )
                message_box.submit(
                    runtime.respond,
                    inputs=[message_box, chatbot],
                    outputs=[message_box, chatbot],
                )
                use_question_button.click(
                    lambda question: question,
                    inputs=[starter_dropdown],
                    outputs=[message_box],
                )
                starter_dropdown.change(
                    lambda question: question,
                    inputs=[starter_dropdown],
                    outputs=[message_box],
                )
                clear_button.click(lambda: ([], ""), outputs=[chatbot, message_box])

    return demo
