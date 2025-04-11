import os
import re
from typing import List, Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from tavily import TavilyClient
import unicodedata

# 環境変数の確認
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY が設定されていません。")
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY が設定されていません。")

# 1. LLMと埋め込みモデルの設定
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings()

# 2. 検索ツール
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query: str) -> List[Dict]:
    try:
        response = tavily_client.search(query, max_results=5)
        results = [
            {
                "url": result["url"],
                "content": unicodedata.normalize("NFKC", result["content"])
            }
            for result in response.get("results", [])
        ]
        return results
    except Exception as e:
        print(f"ウェブ検索エラー: {e}")
        return []

# 3. 検索結果のフィルタリングと検証
def filter_and_verify_results(results: List[Dict]) -> List[Dict]:
    filtered_results = []
    trusted_domains = [
        "wikipedia.org",
        "researchgate.net",
        "arxiv.org",
        ".edu",
        ".gov",
        "technologyreview.com",
        "ibm.com",
        "microsoft.com",
        "forbes.com",
        "idc.com",  # IDC 追加
        "mri.co.jp",  # 三菱総合研究所 追加
        "nikkei.com",  # 日本経済新聞 追加
    ]
    keywords = [
        "aiエージェント",
        "人工知能エージェント",
        "自律性",
        "自律型ai",
        "artificial intelligence",
        "autonomous agent",
        "agentic ai",
        "generative ai",
        "multimodal ai",
        "ai agent",
    ]

    print(f"生の検索結果数: {len(results)}")
    for i, result in enumerate(results):
        content = result["content"].lower()
        url = result["url"]
        print(f"結果 {i+1}: URL={url}, コンテンツプレビュー={content[:100]}...")

        # キーワードチェック
        matched_keywords = [kw for kw in keywords if kw in content]
        if matched_keywords:
            print(f"結果 {i+1}: マッチしたキーワード={matched_keywords}")
            # 信頼できるドメインかチェック
            if any(domain in url for domain in trusted_domains):
                filtered_results.append(result)
                print(f"結果 {i+1} を含む: キーワードと信頼できるドメインに一致")
            else:
                print(f"結果 {i+1} を除外: 信頼できるドメインに一致しない")
        else:
            print(f"結果 {i+1} を除外: キーワードに一致しない")

    print(f"フィルタリング後の結果数: {len(filtered_results)}")
    return filtered_results

# 4. ベクトルストアの構築と検索
def build_vector_store(search_results: List[Dict]) -> FAISS:
    documents = [
        {"page_content": result["content"], "metadata": {"source": result["url"]}}
        for result in search_results
    ]
    if not documents:
        raise ValueError("検索結果が空です。ベクトルストアを構築できません。")
    
    texts = [doc["page_content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

def search_vector_store(vector_store: FAISS, query: str, k: int = 2) -> List[Dict]:
    try:
        results = vector_store.similarity_search_with_score(query, k=k)
        return [
            {"content": doc.page_content, "url": doc.metadata["source"], "score": score}
            for doc, score in results
        ]
    except Exception as e:
        print(f"ベクトル検索エラー: {e}")
        return []

# 5. レポート生成テンプレート
report_template = PromptTemplate(
    input_variables=["query", "results", "conclusions"],
    template="""
# 調査レポート: {query}

## 概要
以下のクエリに基づき、最新の情報を収集し、分析しました。

## 調査結果
{results}

## 結論
収集した情報を基に、{query}に関する主要なポイントを以下にまとめます：
{conclusions}

## 参考文献
{results}
    """
)

def generate_report(query: str, search_results: List[Dict]) -> str:
    if not search_results:
        return f"# 調査レポート: {query}\n\n## 概要\n関連する情報が見つかりませんでした。別のクエリを試してください。"
    
    results_text = ""
    for i, result in enumerate(search_results, 1):
        content = result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
        results_text += f"{i}. [{result['url']}]({result['url']})\n   - {content}\n"
    
    conclusions = []
    for result in search_results:
        content_lower = result["content"].lower()
        if "aiエージェント" in content_lower or "agentic ai" in content_lower or "ai agent" in content_lower:
            conclusions.append("AIエージェントの技術は急速に進化しており、業務の自動化や意思決定支援に活用されている。")
        elif "自律性" in content_lower or "autonomous agent" in content_lower or "自律型ai" in content_lower:
            conclusions.append("自律型エージェントは、複雑なタスクを独立して処理する能力が向上している。")
        elif "generative ai" in content_lower:
            conclusions.append("生成AIは、コンテンツ生成だけでなく、ワークフローの最適化にも利用されている。")
    
    conclusion_text = "\n".join(f"- {c}" for c in conclusions) if conclusions else "- 具体的な結論を導き出せませんでした。"
    
    return report_template.format(query=query, results=results_text, conclusions=conclusion_text)

# 6. カスタムツールの定義
search_tool = Tool(
    name="WebSearch",
    func=web_search,
    description="クエリに基づいてウェブから最新の情報を検索し、信頼性の高い結果を返します。AIエージェントや技術動向の調査に適しています。"
)

# 7. エージェントの設定
tools = [search_tool]
prompt = PromptTemplate(
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
    template="""
あなたは熟練したリサーチャーです。以下のツールを使用して、ユーザーのクエリに答えるために最新の情報を収集し、詳細なレポートを生成してください。

利用可能なツール：
{tools}

ツール名：
{tool_names}

以下の手順を厳密に実行してください：
1. クエリを理解し、必要に応じて WebSearch ツールを使用して情報を収集する。
2. 収集した情報を検証し、信頼性の高いもののみを使用する（例: 信頼できるドメイン、最新のデータ）。
3. 関連性の高い情報を抽出し、構造化されたレポートを生成する。
4. レポートには概要、調査結果、結論、参考文献を含める。

回答は以下の形式で記述してください：
- [THOUGHT] あなたの思考プロセスを説明
- [ACTION] 使用するツールとその入力（例: [WebSearch] query)
- [OBSERVATION] ツールからの結果
- [FINAL ANSWER] 最終的なレポート

クエリ: {input}
作業メモ: {agent_scratchpad}
"""
)
agent = create_react_agent(llm, tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=os.getenv("VERBOSE", "False").lower() == "true")

# 8. エージェントの実行フロー
def run_research_agent(query: str) -> str:
    try:
        raw_results = web_search(query)
        if not raw_results:
            return f"# 調査レポート: {query}\n\n## 概要\nウェブ検索で結果が見つかりませんでした。"
        
        filtered_results = filter_and_verify_results(raw_results)
        if not filtered_results:
            print("警告: 信頼性の高い結果が見つかりませんでした。フィルタリングなしの結果を使用します。")
            filtered_results = raw_results  # フォールバック
            
        vector_store = build_vector_store(filtered_results)
        vector_results = search_vector_store(vector_store, query)
        report = generate_report(query, vector_results)
        return report
    
    except Exception as e:
        return f"# 調査レポート: {query}\n\n## 概要\nエラーが発生しました: {str(e)}\nクエリを再確認し、再度お試しください。"

# 9. 実行例
if __name__ == "__main__":
    try:
        query = "AIエージェントの最新動向について調査し、レポートを作成してください"
        report = run_research_agent(query)
        print(report)
    except Exception as e:
        print(f"実行エラー: {e}")
