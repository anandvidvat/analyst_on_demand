
from get_started_with_pdf import query_engine

print(query_engine)

import gradio as gr

def messaging_interface(message ="Greetings!", history = []):    
    try:
        response = query_engine.query(message)
        print(response)
        print(f"for question : {message} \n following answer : {response}")
        if len(history) != 0 :
            print(f"while the history is {len(history)} and last msg is {history[-1]}")

    except ValueError as e:
        print(e)
        response = "unable to continue without additional context or training"


    return str(response)

demo = gr.ChatInterface(messaging_interface)

if __name__ == "__main__":
    demo.launch()