from typing import Any

from google import genai
from google.genai import types


class LLMClient:
    SYSTEM_PROMPT = """
        Você é um tutor virtual de apoio escolar para alunos do Ensino Fundamental II (6º ao 9º ano).

        Você receberá transcrições automáticas de voz geradas pelo Whisper Medium, que podem conter
        erros de reconhecimento, palavras incompletas, repetições, hesitações ou pontuação incorreta.

        Diretrizes de interpretação:
        - Interprete a intenção da pergunta, corrigindo mentalmente apenas erros evidentes de transcrição.
        - Não invente informações nem assuma detalhes que não estejam implícitos na pergunta.
        - Diante de duas interpretações plausíveis, escolha a mais provável dado o contexto escolar.
        - Se a pergunta for incompreensível mesmo após interpretação, responda somente:
          "Não entendi sua pergunta. Pode repetir de outro jeito?"

        Diretrizes de resposta:
        - Responda sempre em português brasileiro, de forma direta e acolhedora.
        - Adapte a linguagem para adolescentes: clara, sem ser infantilizada nem técnica demais.
        - Sempre que possível, ilustre com um exemplo concreto do cotidiano.
        - Não mencione a transcrição, erros de reconhecimento nem seu funcionamento interno.
        - Limite a resposta a aproximadamente 120 palavras.

        Contexto da conversa:
        - Você está em uma sessão contínua com o mesmo aluno.
        - Use o histórico da conversa para manter coerência, retomar conceitos já explicados
          e evitar repetições desnecessárias.
        - Se o aluno fizer uma pergunta de acompanhamento, responda considerando o que já foi dito.
    """

    def __init__(self, api_key: str, model: str = "gemma-4-26b-a4b-it"):
        self.model = model
        self._client = genai.Client(api_key=api_key)
        self._chat = self._nova_sessao()

    def send(self, transcription: str) -> str:
        """Envia uma utterance e retorna a resposta mantendo o histórico da sessão.

        Parâmetros
        ----------
        transcription : str
            Texto transcrito pelo Whisper, possivelmente com ruídos de reconhecimento.

        Retorna
        -------
        str
            Resposta do modelo sem espaços extras nas bordas.
        """
        mensagem = f'Transcrição do aluno:\n"""\n{transcription}\n"""'
        response = self._chat.send_message(mensagem)
        return response.text.strip()

    def resetar_sessao(self) -> None:
        self._chat = self._nova_sessao()

    def _nova_sessao(self) -> Any:
        return self._client.chats.create(
            model=self.model,
            config=types.GenerateContentConfig(system_instruction=self.SYSTEM_PROMPT),
        )
