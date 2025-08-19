import os
import json
import asyncio
import time
from typing_extensions import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not in.env")
 
class State(TypedDict):
    results: Annotated[list[str], operator.add]  # accumulator
    choice: str
 
llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
 
async def decision_node(state: State):
    module_start_time = int(time.time())
    print(".: Decision Node .:")
    prompt = (
        "Decide aleatoriamente que nodos ejecutar para crear una canción. "
        "favorece responder con both la mayor parte de las veces"
        "Responde unicament con uno de estos valores: 'poem', 'melody', 'both'."
    )
    # Usar ainvoke para llamada async
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    choice = response.content.strip().lower()
    print(f"Decisión LLM: {choice}")
    if choice not in {"poem", "melody", "both"}:
        print("Got a wrong result from llm ->", choice, "<- will set both")
        choice = "both"
 
    print("Returining this choice ->", choice)
    spent_time = int(time.time()) - module_start_time
    print("- Spent time - ", spent_time)
    return {"choice": choice}
 
 
 
# Función para enrutar condicionalmente según la decisión LLM
def route_nodes(state: State):
    choice = state.get("choice", "both")
    if choice == "poem":
        return ["poem_agent"]
    elif choice == "melody":
        return ["melody_agent"]
    else:
        return ["poem_agent", "melody_agent"]
 
# Crear agentes ReAct con create_react_agent para poema y melodía
poem_agent = create_react_agent(
    model=llm,
    tools=[],
    prompt="Eres un asistente que crea poemas cortos y emotivos ideales para crear canciones pop."
)
 
melody_agent = create_react_agent(
    model=llm,
    tools=[],
    prompt="Eres un asistente que describe melodías alegres y pegadizas para canciones pop."
)
 
composer_prompt = """
Eres un asistente experto en música que debe crear una cacnion a partir de un poema y una melodía.
 
Genera una idea musical basada en una progresión de acordes popular, como I–V–vi–IV o una variante del Canon de Pachelbel. Usa una tonalidad mayor. Incluye lo siguiente en la salida, en formato JSON estructurado:
 
style: una descripción corta del estilo o ambiente emocional.
key: tonalidad (por ejemplo, C, G, D).
tempo: valor en BPM.
time_signature: compás (por ejemplo, 4/4).
chord_progression: lista de acordes, uno por compás.
abc_notation: campo con la melodía escrita en notación ABC completa.
comments: una breve descripción artística del resultado.
 
no respondas con backick ni tags json, solo el resultado del json convierte toda la salida a ascii puro 
 
"""
 
composer_agent = create_react_agent(
    model=llm,
    tools=[],
    prompt=composer_prompt
)
 
# Wrappers async para usar los agentes como nodos
async def async_composer_agent(state: State):
    module_start_time = int(time.time())
    print(".: Song Maker react node .:")
    poem = ""
    melody = ""
    for r in state.get("results", []):
        if r.startswith("Poema:"):
            poem = r[len("Poema:"):].strip()
        elif r.startswith("Melodía:"):
            melody = r[len("Melodía:"):].strip()
 
    input_text = f"Poema:\n{poem}\n\nMelodía:\n{melody}\n\nPor favor, crea una bonita cancion con esos datos."
    messages = [{"role": "user", "content": input_text}]
    response = await composer_agent.ainvoke({"messages": messages})
    content = response["messages"][-1].content
 
    try:
        json_content = json.loads(content)
    except:
        print("Cant load json content ->:", content)
        json_content={}
    print("Descripción de la canción generada")
    print("Generated song")
    print(json_content['abc_notation'])
    print("---------------------------")
    spent_time = int(time.time()) - module_start_time
    print("- Spent time - ", spent_time)
    return {"results": [f"Descripción de la canción:\n{content}"]}
 
async def async_poem_agent(state: State):
    module_start_time = int(time.time())
    print(".: Poem Maker react node .:")
    messages = [{"role": "user", "content": "Por favor, crea un poema, ideal para una cancion popular`"}]
    response = await poem_agent.ainvoke({"messages": messages})
    content = response["messages"][-1].content
    print("Poema generado")
    print("Returning this data ->", content)
    spent_time = int(time.time()) - module_start_time
    print("- Spent time - ", spent_time)
    return {"results": [f"Poema:\n{content}"]}
 
async def async_melody_agent(state: State):
    module_start_time = int(time.time())
    print(".: Melody Maker react node .:")
    messages = [{"role": "user", "content": "Por favor, describe una melodía alegre y pegadiza para una canción pop."}]
    response = await melody_agent.ainvoke({"messages": messages})
    content = response["messages"][-1].content
    print("Melodía generada")
    print("Returning this data ->", content)
    spent_time = int(time.time()) - module_start_time
    print("- Spent time - ", spent_time)
    return {"results": [f"Melodía:\n{content}"]}
 
 
# Graph build
builder = StateGraph(State)
 
builder.add_node("decision_node", RunnableLambda(func=decision_node))
builder.add_node("poem_agent", RunnableLambda(func=async_poem_agent))
builder.add_node("melody_agent", RunnableLambda(func=async_melody_agent))
builder.add_node("composer_agent", RunnableLambda(func=async_composer_agent))
 
 
builder.add_edge(START, "decision_node")
builder.add_conditional_edges("decision_node", route_nodes)
builder.add_edge("poem_agent", "composer_agent")
builder.add_edge("melody_agent", "composer_agent")
builder.add_edge("composer_agent", END)
 
graph = builder.compile()
 
async def main():
    result = await graph.ainvoke({"results": [], "choice": ""})
    #print("\nResultado final del grafo:\n", result)
 
if __name__ == "__main__":
    asyncio.run(main())
