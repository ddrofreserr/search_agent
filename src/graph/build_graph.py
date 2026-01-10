from src.agent import SearchAgent

if __name__ == "__main__":
    q = input("User query> ").strip()
    SearchAgent().run_cli(q)
