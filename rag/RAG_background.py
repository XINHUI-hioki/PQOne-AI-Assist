import os, re
from langchain_community.document_loaders import (
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader, 
    DirectoryLoader,   
)
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader


DOCX_PATH = "data/620.docx"
TXT_PATH = "rag/readme.txt"
MD_PATH   = "data/pqone_user_guide_preview.md"
CHROMA_DB_DIR = "./chroma_db_ollama"
CHROMA_MD_DIR = "./chroma_md"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True}
)

docx_loader = Docx2txtLoader(DOCX_PATH)
readme_loader = TextLoader(TXT_PATH, encoding="utf-8")
md_loader     = UnstructuredMarkdownLoader(MD_PATH)
documents = (
    docx_loader.load()
    + readme_loader.load()
    + md_loader.load()            
)



def parse_markdown_by_heading_tree(md_path: str, min_level: int = 2) -> list:
    """
    将 Markdown 文档按层级结构组织，将每个 min_level 标题及其子内容组成一个 chunk。
    例如：min_level=2 → 每个 `##` 标题块包含其下所有 `###`、`####` 内容。
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    headings = list(re.finditer(r"^(#{1,6})\s+(.*)", content, flags=re.MULTILINE))

    docs = []
    for idx, match in enumerate(headings):
        level = len(match.group(1))  # 1~6
        title = match.group(2).strip()
        if level != min_level:
            continue

        start = match.end()
        end = headings[idx + 1].start() if idx + 1 < len(headings) else len(content)
        chunk_text = content[start:end].strip()

        full_chunk = f"{match.group(0)}\n{chunk_text}"
        docs.append(Document(page_content=full_chunk))

    return docs

if not os.path.exists(os.path.join(CHROMA_MD_DIR, "index")):
    md_docs = parse_markdown_by_heading_tree(md_path=MD_PATH, min_level=2)
    md_vectorstore = Chroma.from_documents(md_docs, embeddings, persist_directory=CHROMA_MD_DIR)
    md_vectorstore.persist()


else:
    md_vectorstore = Chroma(persist_directory=CHROMA_MD_DIR, embedding_function=embeddings)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = text_splitter.split_documents(documents)
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

if not os.path.exists(os.path.join(CHROMA_DB_DIR, "index")):
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_DIR)
    vectorstore.persist()
else:

    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

retriever = vectorstore.as_retriever()

llm = Ollama(model="llama3", base_url="http://localhost:11434")
#llm = Ollama(model="llama3:8b", base_url="http://localhost:11434")



def build_prompt_template(language: str = "English"):
    lang = language.lower()
    if "chinese" in lang or "中文" in lang:
        instruction = (
            "你是 PQ ONE 软件助手，请根据提供的文档内容完整回答问题。\n"
            "如果涉及多个区域，请逐条列出并详细解释。\n"
            "如果原文包含图片（如 ![](xxx) 或 <img src=...>），请保留原始图片标签，以便用户界面显示。"
        )
    elif "japanese" in lang or "日本" in lang:
        instruction = (
            "あなたは PQ ONE ソフトウェアの説明アシスタントです。文書の情報に基づいてすべての項目を丁寧に説明してください。\n"
            "画像が含まれている場合、<img>タグやMarkdown形式を残してください。"
        )
    else:
        instruction = (
            "Answer the user's question strictly in English using the provided documentation. if have image please show\n\n"
            "You are a helpful assistant for explaining the PQ ONE software.\n\n"
            "Use the following documentation snippets to answer the user's question thoroughly and completely.\n\n"
            "If the documentation describes multiple items (such as 5 areas), list and explain each one clearly.\n"
            "If the documentation contains image syntax (e.g., ![](...) or <img src=...>), preserve the tag so the UI can render the image.\n\n"
            "If the answer is not found in the documents, say: \"I can't find it in the file.\""
        )

    return ChatPromptTemplate.from_template(
        f"""{instruction}

Question: {{input}}

Relevant Info: {{context}}
"""
    )


def convert_md_images_to_html(md_text: str) -> str:
    return re.sub(r'!\[\]\((.*?)\)', r'<img src="\1" width="500"/>', md_text)

def answer_from_rag_with_lang(question: str, language: str = "English") -> str:
    try:
        prompt = build_prompt_template(language)
        document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": question})
        answer = response["answer"]
        answer = convert_md_images_to_html(answer)
        return answer
    except Exception as e:
        return f"[RAG Error] {e}"

def build_conversation_prompt(language: str = "English") -> PromptTemplate:
    lang = language.lower()
    if "chinese" in lang or "中文" in lang:
        instruction = "你是 PQ ONE 软件的操作助手，请逐步指导用户完成请求，回答请分步骤描述。"
    elif "japanese" in lang or "日本" in lang:
        instruction = "あなたはPQ ONEソフトウェアのナビゲーターです。ユーザーを一歩ずつ案内してください。回答はステップごとに記述してください。"
    else:
        instruction = "You are a PQ ONE software assistant. Please walk the user through the task step-by-step."

    return PromptTemplate.from_template(
        f"""{instruction}

Conversation history:
{{history}}

User: {{input}}
Assistant:"""
    )

memory = ConversationBufferMemory(return_messages=True)

conversation_prompt = build_conversation_prompt(language="English")  
conversation_chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=conversation_prompt,
    verbose=False
)

def chat_from_memory(prompt_with_context: str, language: str = "English") -> str:
    try:
        response = conversation_chain.run(prompt_with_context)
        return response
    except Exception as e:
        return f"[Memory Chat Error] {e}"
    
def generate_followup_question(user_question: str, ai_answer: str, language: str = "English") -> str:
    prompts = {
        "English": (
            "You're a helpful assistant. Based on the user's question and your answer, suggest a follow-up question "
            "that guides the user to explore related data or actions."
        ),
        "中文": (
            "你是一个智能助手，请根据用户的问题和你的回答，提出一个有意义的后续问题，"
            "帮助用户进一步探索相关内容或采取下一步行动。"
        ),
        "日本語": (
            "あなたは親切なアシスタントです。ユーザーの質問とあなたの回答に基づいて、関連データや次のステップを提案するフォローアップ質問を生成してください。"
        )
    }
    instruction = prompts.get(language, prompts["English"])

    prompt = f"""{instruction}

User's question:
{user_question}

AI's answer:
{ai_answer}

Please provide one helpful follow-up question in {language}.
"""
    
#    print("========== Prompt for Debugging(follow_up) ==========")
#    print(prompt)
#    print("==========================================")    

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        return ""

def answer_query(query: str, language: str = "English") -> str:
    return answer_from_rag_with_lang(query, language)
def chat_query(query: str, language: str = "English") -> str:
    return chat_from_memory(query, language)
def followup_query(user_q: str, ai_a: str, language: str = "English") -> str:
    return generate_followup_question(user_q, ai_a, language)
def build_qa_chain(language: str = "English"):

    prompt = build_prompt_template(language)
    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    return create_retrieval_chain(retriever, doc_chain)

def load_md_documents(data_dir="data"):
    loader = DirectoryLoader(data_dir, glob="**/*.md", loader_cls=TextLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    return split_docs

md_docs = load_md_documents()
md_vectorstore = Chroma.from_documents(md_docs, embeddings, persist_directory="./chroma_md")

def search_md_only_context(query, k=3):
    retriever = md_vectorstore.as_retriever(search_kwargs={"k": k + 5})  # 多取一些
    raw_results = retriever.get_relevant_documents(query)

    seen_contents = set()
    results = []
    for doc in raw_results:
        content = doc.page_content.strip()
        if content not in seen_contents:
            results.append(doc)
            seen_contents.add(content)
        if len(results) >= k:
            break

    print(f" [Beginner Mode] 去重后返回 {len(results)} 条")
    return results


qa_chain = build_qa_chain()   
__all__ = [
    "answer_query",
    "chat_query",
    "followup_query",
    "memory",
    "qa_chain",
    "search_md_only_context", 
]
