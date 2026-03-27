import gradio as gr


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
                clear_button.click(lambda: ([], ""), outputs=[chatbot, message_box])

    return demo
