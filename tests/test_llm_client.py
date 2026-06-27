from unittest.mock import MagicMock, patch


@patch("google.genai.Client")
def test_send_retorna_texto(mock_genai):
    mock_chat = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "  Resposta do tutor.  "
    mock_chat.send_message.return_value = mock_response
    mock_genai.return_value.chats.create.return_value = mock_chat

    from api.llm_client import LLMClient

    client = LLMClient(api_key="fake-key", model="gemma-fake")
    resposta = client.send("quanto é 2 mais 2?")

    assert resposta == "Resposta do tutor."
    mock_chat.send_message.assert_called_once()


@patch("google.genai.Client")
def test_resetar_sessao_cria_novo_chat(mock_genai):
    chat1 = MagicMock()
    chat2 = MagicMock()

    mock_genai.return_value.chats.create.side_effect = [
        chat1,
        chat2,
    ]

    from api.llm_client import LLMClient

    client = LLMClient(api_key="fake-key", model="gemma-fake")

    assert client._chat is chat1

    client.resetar_sessao()

    assert client._chat is chat2
    assert mock_genai.return_value.chats.create.call_count == 2


@patch("google.genai.Client")
def test_send_strip_espacos(mock_genai):
    mock_chat = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "\n\n  texto com espaços  \n"
    mock_chat.send_message.return_value = mock_response
    mock_genai.return_value.chats.create.return_value = mock_chat

    from api.llm_client import LLMClient

    client = LLMClient(api_key="fake-key", model="gemma-fake")

    assert client.send("teste") == "texto com espaços"
