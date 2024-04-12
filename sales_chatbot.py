import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter

embeddings = OpenAIEmbeddings()


def prepare_vectors():
    text_splitter = CharacterTextSplitter(
        separator=r'\d+\.',
        chunk_size=100,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=True
    )
    with open("sales_data.txt", mode="r", encoding="utf8") as f:
        cellphone_sales_data = f.read()

    docs = text_splitter.create_documents([cellphone_sales_data])
    print(f"[docs count]{len(docs)}")

    db = FAISS.from_documents(docs, embeddings)
    db.save_local("sales_data")


def initialize_sales_bot(vector_store_dir: str = "sales_data"):
    db = FAISS.load_local(vector_store_dir, embeddings, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    global sales_bot_chain
    sales_bot_chain = RetrievalQA.from_chain_type(llm,
                                                  retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                            search_kwargs={"score_threshold": 0.8})
                                                  )
    # 返回向量数据库的检索结果
    sales_bot_chain.return_source_documents = True
    return sales_bot_chain


def sales_chat(message, histories):
    # TODO: 从命令行参数中获取
    enable_chat = True

    if len(histories) > 0:
        # 只记录最近的3条问答
        histories = histories[-3:] if len(histories) > 3 else histories
        # 补充历史问答记录
        message += "\n以下是历史问答记录，按时间倒序排列，如果检索不到数据时，你可以利用这些记录：\n"
        for i in range(len(histories), 0, -1):
            message += f"[历史问题]{histories[i-1][0]}\n[历史回答]{histories[i-1][1]}\n\n"

    print(f"[message]{message}")
    ans = sales_bot_chain.invoke({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"].split("[销售回答]")[-1]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"


def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="房产和手机销售",
        retry_btn=None,
        undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    # 准备向量数据
    prepare_vectors()
    # 初始化销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
