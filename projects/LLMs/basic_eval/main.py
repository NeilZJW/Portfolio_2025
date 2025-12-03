# Username:3036378489
# Password:uChLp1kRYpi5wh7FQ1L41qstPvE40cju


import pprint
from modelhub import Modelhub
from openai import OpenAI


# Modelhub is a hub that integrates multiple LLM models from
# various providers. The MODELHUB_HOST refers to the server-side
# address of Modelhub.
MODELHUB_HOST = "https://modelhub.puyuan.tech/api/"
USERNAME = "3036378489"
PASSWORD = "uChLp1kRYpi5wh7FQ1L41qstPvE40cju"

def test_modelhub():
    """
    You may call the modelhub server using modelhub-client.
    See: https://pypi.org/project/puyuan_modelhub/
    """
    mh = Modelhub(host=MODELHUB_HOST, username=USERNAME, password=PASSWORD)

    # List models
    print("Show supported models:")
    pprint.pprint(mh.supported_models)

    # Generate text
    print("\n\nGenerate text sync:")
    print(mh.generate("hello, who are you?", model="gpt-4o").
          choices[0].message.content)

    # Streaming text
    print("\n\nGenerate text async:")
    for t in mh.stream("hello, who are you?", model="gpt-4o"):
        print(t.token, end="")

    # Embedding
    print("\n\nEmbedding:")
    print(mh.embedding("hello", model="m3e"))

    # Rerank
    print("\n\nRerank:")
    print(mh.rerank(
        [["hello", "world"], ["hello", "morning"]],
        model="bge-reranker-base"
    ))


def test_openai():
    """
    At the same time, modelhub server is OpenAI API compatible.
    """
    client = OpenAI(
        api_key=f"{USERNAME}:{PASSWORD}",
        base_url=f"{MODELHUB_HOST}/v1/"
    )
    print("Generate text async using openai:")
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": "hello, who are you?"
        }],
        stream=True
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content, end="")

    print("\n\nEmbedding using openai:")
    response = client.embeddings.create(
        input="hello",
        model="m3e"
    )
    print(response.data[0].embedding)


if __name__ == '__main__':
    test_modelhub()
    print("--------------------------------\n\n")
    test_openai()









