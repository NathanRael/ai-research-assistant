import sys

from dotenv import load_dotenv

from app.chat_opencode import ChatOpenCode, OpenCodeModel
from app.config import settings

load_dotenv()


def ask():
    model = ChatOpenCode(api_key=settings.opencode_api_key, model_name=OpenCodeModel.KIMI_K26)
    response = model.invoke('Hello ! What can you do for me as a dev assistant?')
    print(response.content.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))


if __name__ == "__main__":

    ask()

