from dotenv import load_dotenv
from rag_core.runtime import CareerQARuntime
from rag_core.ui import build_demo


def main() -> None:
    load_dotenv()
    runtime = CareerQARuntime()
    demo = build_demo(runtime)
    demo.launch()


if __name__ == "__main__":
    main()
