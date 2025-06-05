import ollama

response = ollama.chat(model='llama3:2.3b', messages=[{
    'role': 'user', 
    'content': "what if I destroy all Apple product?"}])
print(response["message"]["content"])