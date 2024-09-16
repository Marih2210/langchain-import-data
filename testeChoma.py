import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Carregar variáveis de ambiente
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Definir documentos manualmente
documents_content = [
    "Data de Submissão: 2024-01-01\nID do Revisor: 123\nID do Produto: 456\nNome do Produto: Produto A\nMarca do Produto: Marca X\nCategoria do Site LV1: Eletrônicos\nCategoria do Site LV2: Smartphones\nTítulo da Revisão: Excelente Produto\nAvaliação Geral: 5\nRecomendaria a um Amigo: Sim\nTexto da Revisão: Muito bom!\nAno de Nascimento do Revisor: 1985\nGênero do Revisor: Feminino\nEstado do Revisor: SP",
    "Data de Submissão: 2024-01-02\nID do Revisor: 124\nID do Produto: 457\nNome do Produto: Produto B\nMarca do Produto: Marca Y\nCategoria do Site LV1: Eletrodomésticos\nCategoria do Site LV2: Refrigeradores\nTítulo da Revisão: Bom\nAvaliação Geral: 4\nRecomendaria a um Amigo: Não\nTexto da Revisão: A entrega foi lenta.\nAno de Nascimento do Revisor: 1990\nGênero do Revisor: Masculino\nEstado do Revisor: RJ"
]

# Criar documentos
documents = [Document(page_content=content) for content in documents_content]

# Dividir os documentos em chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_chunks = text_splitter.split_documents(documents)

# Remover duplicatas dos chunks
unique_chunks = []
seen_texts = set()
for chunk in text_chunks:
    if chunk.page_content not in seen_texts:
        seen_texts.add(chunk.page_content)
        unique_chunks.append(chunk)

# Criar embeddings e vetorstore
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

# Adicionar documentos ao Chroma
vectorstore.add_documents(unique_chunks)  # Adiciona todos os documentos únicos

# Teste a recuperação
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# Definir template e prompt
template = """Você é um assistente treinado para responder perguntas com base nas seguintes informações:
- Data de Submissão
- ID do Revisor
- ID do Produto
- Nome do Produto
- Marca do Produto
- Categoria do Site LV1
- Categoria do Site LV2
- Título da Revisão
- Avaliação Geral
- Recomendaria a um Amigo
- Texto da Revisão
- Ano de Nascimento do Revisor
- Gênero do Revisor
- Estado do Revisor

Use os pedaços de texto retornados como base para suas respostas. Se a informação não estiver disponível ou não souber a resposta, diga "Não é Possível Retornar esse Dado".

Pergunta: {question}
Contexto: {context}
Resposta:
"""

prompt = PromptTemplate.from_template(template)
output_parser = StrOutputParser()

# Inicializar modelo e cadeia de processamento
llm_model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.2)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | output_parser
)

# Loop para receber e processar perguntas do usuário
while True:
    user_question = input("Digite sua Pergunta (ou 'Sair' para encerrar): ")
    if user_question.strip().lower() == 'sair':
        print("Saindo do programa.")
        break

    print("Pergunta recebida:", user_question)
    responseQuestion = rag_chain.invoke(user_question)
    print("Resposta gerada:", responseQuestion)
    print('-' * 80)
