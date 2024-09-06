import gradio as gr
import requests

def send_query(query):
    url = 'http://127.0.0.1:8001/chat'  # URL of the FastAPI endpoint
    json_data = {"query": query}
    response = requests.post(url, json=json_data)

    if response.status_code == 200:
        data = response.json()
        return data["response"]["output"]
    else:
        return "Failed to get response from server."

def main():
    interface = gr.Interface(
        fn=send_query,
        inputs="text",
        outputs="text",
        title="Query Interface",
        description="Enter your query to get a response from the backend API."
    )
    interface.launch()

# if __name__ == "__main__":
#     main()
