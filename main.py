from sentence_transformers import SentenceTransformer
from news_data_manager import NewsDataManager

# paraphrase-multilingual-MiniLM-L12-v2 -> Rápido
# paraphrase-multilingual-mpnet-base-v2 -> Preciso
# all-mpnet-base-v2 -> Em inglês

def search(model, manager, doc_embeddings):
    while True:
        query = input("\nBusca (ou 'sair' para sair): ").strip()

        if query.lower() == "sair":
            return

        if not query:
            print("Digite algo para buscar.")
            continue

        query_embedding = model.encode_query(query)
        similarities = model.similarity(query_embedding, doc_embeddings)[0]
        ranked = sorted(zip(similarities.tolist(), manager.data), key=lambda x: x[0], reverse=True)
        top3 = ranked[:3]

        print("\nTop 3 resultados:")
        for i, (score, notice) in enumerate(top3, 1):
            print(f"  [{i}] ({score:.4f}) {notice['titulo']}")

        while True:
            choice = input("\nEscolha um artigo (1-3), nova busca (b) ou sair (s): ").strip().lower()

            if choice == "s":
                return
            elif choice == "b":
                break
            elif choice in ("1", "2", "3"):
                _, notice = top3[int(choice) - 1]
                print(f"\n{'=' * 50}")
                print(f"Título: {notice['titulo']}")
                print(f"Data: {notice['data']} | Fonte: {notice['fonte']}")
                print(f"{'=' * 50}")
                print(notice['texto'])
                print(f"{'=' * 50}")

                back = input("\nVoltar para (b) busca, (r) resultados ou (s) sair: ").strip().lower()
                if back == "s":
                    return
                elif back == "b":
                    break
                elif back == "r":
                    continue
                else:
                    print("Opção inválida, voltando aos resultados.")
            else:
                print("Opção inválida. Digite 1, 2, 3, 'b' ou 's'.")


def main():
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", cache_folder="modelos/")

    manager = NewsDataManager("dados/noticias_brutas.json")

    corpus = [f"{n['titulo']}. {n['texto']}" for n in manager.data]
    doc_embeddings = model.encode_document(corpus)

    search(model, manager, doc_embeddings)


if __name__ == "__main__":
    main()